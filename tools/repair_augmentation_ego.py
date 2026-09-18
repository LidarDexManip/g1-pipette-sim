"""Immutable camera-only rendering revisions from accepted full simulation states."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,copy,json,multiprocessing,shutil,sys,tempfile,time
import concurrent.futures
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import cv2,mujoco,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.augmentation.contracts import atomic_json,bind_run,dependencies,digest,file_hash,lock,seal,verify
from g1_pipette_sim.augmentation.export import CAPTURE_FILES,catalog,verify_catalog
from g1_pipette_sim.camera_views import apply_camera_views,assert_camera_views
from g1_pipette_sim.camera_streams import CameraStreams
from g1_pipette_sim.dynamics import build_dynamic_xml


def without_cameras(xml):
    tree=ET.fromstring(xml)
    for parent in tree.iter():
        for camera in list(parent.findall('camera')):parent.remove(camera)
    return ET.tostring(tree)


def repair(spec):
    started=time.monotonic();entry=spec['entry'];source=Path(entry['folder']);dest=Path(spec['destination'])
    verify_catalog({'entries':[entry]})
    inputs={'entry':entry,'camera_config':spec['config'],'implementation':spec['implementation']}
    with lock(dest.parent/('lock-'+dest.name)):
        if dest.exists():
            if not verify(dest,'repair',inputs):raise ValueError('Unverified existing repair')
            return json.loads((dest/'render-audit.json').read_text())
        stage=Path(tempfile.mkdtemp(prefix='.'+dest.name+'-',dir=dest.parent))
        try:
            oldcfg=json.loads((source/'config.json').read_text());cfg=spec['config']
            for name in set(oldcfg)|set(cfg):
                if name not in ['cameras','camera_view_correction']:assert oldcfg.get(name)==cfg.get(name)
            for name in oldcfg['cameras']:
                if name!='rgb':assert oldcfg['cameras'][name]==cfg['cameras'][name]
            assert without_cameras(build_dynamic_xml(oldcfg))==without_cameras(build_dynamic_xml(cfg))
            model=mujoco.MjModel.from_xml_string(build_dynamic_xml(cfg));data=mujoco.MjData(model)
            assert model.nmocap==0, 'Saved states do not include mocap positions'
            for name in ['commands.npz','trajectory.npz','verification.json','wrist_left.mp4','wrist_right.mp4']:
                os.link(source/name,stage/name)
            atomic_json(stage/'config.json',cfg)
            rows=np.load(source/'trajectory.npz');n=len(rows['timestamp']);assert model.nq==rows['qpos'].shape[1]
            poses=json.loads((source/'camera-poses.json').read_text());assert len(poses)==n
            writer=cv2.VideoWriter(str(stage/'rgb.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(512,288))
            if not writer.isOpened():raise RuntimeError('Cannot open video writer')
            max_nozzle_error=0.;max_tip_error=0.
            try:
                with CameraStreams(model,['rgb'],512,288) as camera:
                    for i in range(n):
                        data.qpos[:]=rows['qpos'][i];data.qvel[:]=rows['qvel'][i];data.time=float(rows['simulation_time'][i])
                        # Recompute transforms at the saved state without integrating physics.
                        mujoco.mj_forward(model,data)
                        pixels=camera.render(data)['rgb']
                        max_nozzle_error=max(max_nozzle_error,float(np.linalg.norm(data.site('nozzle').xpos-rows['nozzle_position'][i])))
                        max_tip_error=max(max_tip_error,float(np.linalg.norm(data.body('tip').xpos-rows['tip_position'][i])))
                        poses[i]['rgb']=camera.poses(data)['rgb']
                        writer.write(cv2.cvtColor(pixels,cv2.COLOR_RGB2BGR))
            finally:writer.release()
            atomic_json(stage/'camera-poses.json',poses)
            # Full decode, dimensions, frame count and encoded duration check.
            video=cv2.VideoCapture(str(stage/'rgb.mp4'));decoded=0;fps=video.get(cv2.CAP_PROP_FPS)
            while True:
                ok,pixels=video.read()
                if not ok:break
                assert pixels.shape==(288,512,3);decoded+=1
            video.release();assert decoded==n and abs(fps-30)<1e-6
            for name in ['commands.npz','trajectory.npz','verification.json','wrist_left.mp4','wrist_right.mp4']:
                assert file_hash(stage/name)==entry['files'][name]
            revision={'method':'saved_state_camera_rerender','camera':'rgb','source_capture_id':entry['capture_id'],
                      'source_folder':str(source),'source_config_sha256':entry['files']['config.json'],
                      'camera_correction':cfg['camera_view_correction'],'physics_verdict':'inherited from original independent rollout; physics XML excluding cameras identical',
                      'render_revision_id':digest(inputs)}
            candidate=copy.deepcopy(entry['candidate']);candidate.update(folder=str(dest),config=str(dest/'config.json'),
                commands=str(dest/'commands.npz'),config_sha256=file_hash(stage/'config.json'),number=spec['episode_index'],
                capture_id=entry['capture_id'],render_revision=revision)
            candidate['lineage']=[*candidate.get('lineage',[]),revision]
            record=json.loads((source/'accepted.json').read_text());record['candidate']=candidate;record['render_revision']=revision
            record['artifact_hashes']={name:file_hash(stage/name) for name in CAPTURE_FILES if name!='accepted.json'}
            atomic_json(stage/'candidate.json',candidate);atomic_json(stage/'accepted.json',record)
            result={'episode_index':spec['episode_index'],'capture_id':entry['capture_id'],'frames':n,'decoded_frames':decoded,
                    'physics_xml_equal_except_cameras':True,'unchanged':['trajectory.npz','commands.npz','verification.json','wrist_left.mp4','wrist_right.mp4'],
                    'max_saved_nozzle_position_difference_m':max_nozzle_error,'max_saved_tip_position_difference_m':max_tip_error,
                    'forward_method':'mj_forward at saved qpos/qvel; no physics integration','seconds':time.monotonic()-started}
            atomic_json(stage/'render-audit.json',result)
            verify_catalog({'entries':[entry]})
            seal(stage,'repair',inputs,[stage/name for name in [*CAPTURE_FILES,'candidate.json','render-audit.json']])
            os.rename(stage,dest)
            return result
        finally:
            if stage.exists():shutil.rmtree(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--pilot-only',action='store_true');a=p.parse_args()
    root=a.output.resolve();value=json.loads(a.catalog.read_text());implementation=dependencies()
    with lock(root):
        verify_catalog(value)
        plans=[]
        for index,entry in enumerate(value['entries']):
            cfg=json.loads((Path(entry['folder'])/'config.json').read_text())
            ep=SimpleNamespace(root=Path(entry['candidate']['dataset_snapshot']),index=entry['candidate']['parent_episode'])
            corrected=apply_camera_views(cfg,ep);assert_camera_views(corrected,ep)
            if corrected['cameras']!=cfg['cameras']:
                plans.append({'episode_index':index,'entry':entry,'config':corrected,'destination':str(root/'captures/candidates'/f'{index:06}'),'implementation':digest(implementation)})
        contract={'catalog':value,'plans':plans,'dependencies':implementation,'method':'saved_state_camera_rerender'}
        bind_run(root,contract);(root/'captures/candidates').mkdir(parents=True,exist_ok=True)
        atomic_json(root/'repair-plan.json',{'total':len(value['entries']),'affected':len(plans),'unchanged':len(value['entries'])-len(plans)})
        selected=plans[:1] if a.pilot_only else plans
        audits=[];started=time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as pool:
            for result in pool.map(repair,selected):
                audits.append(result);progress={'repaired':len(audits),'target':len(plans),'elapsed_s':time.monotonic()-started,'last':result}
                atomic_json(root/'progress.json',progress);print(json.dumps(progress),flush=True)
        if a.pilot_only:return
        assert dependencies()==implementation,'Code or assets changed during rendering'
        verify_catalog(value)
        changed=catalog([root/'captures']);by_id={e['capture_id']:e for e in changed['entries']}
        assert len(by_id)==len(plans) and not changed['duplicates']
        updated={**value,'entries':[by_id.get(e['capture_id'],e) for e in value['entries']],
                 'render_revision':'ego-camera-correction-v2','source_catalog_sha256':file_hash(a.catalog)}
        assert [e['capture_id'] for e in updated['entries']]==[e['capture_id'] for e in value['entries']]
        verify_catalog(updated);atomic_json(root/'accepted-catalog.json',updated)
        atomic_json(root/'audit.json',{'complete':True,'episodes':len(value['entries']),'repaired':len(plans),'unchanged':len(value['entries'])-len(plans),'frames':sum(e['frames'] for e in value['entries']),'details':audits})
if __name__=='__main__':main()
