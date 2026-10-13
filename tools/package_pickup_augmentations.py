"""Package accepted simulator captures as a separate LeRobot v2.1 motor dataset.

Only accepted.json captures are exported; no 35-D high-level labels are copied.
Videos are hard-linked when possible. This tool can refresh a partial collection.
"""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.pickup_layouts import ROOT,sha256
from tools.collect_pickup_augmentations import SIZES,FPS
from tools.fit_pickup_layouts import write_json

TASK='Attach a green pipette tip from the rack and lift it with the pipette.'


def stats(values):
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[:,None]
    return {'min':x.min(axis=0).tolist(),'max':x.max(axis=0).tolist(),'mean':x.mean(axis=0).tolist(),'std':x.std(axis=0).tolist(),'count':[len(x)]}


def image_stats(path):
    import cv2
    cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));means=[];seconds=[];low=[];high=[]
    for index in np.unique(np.linspace(0,n-1,min(n,12),dtype=int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read()
        if not ok:raise ValueError('Undecodable accepted video')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);mean,std=cv2.meanStdDev(rgb);mean=mean.ravel()/255;std=std.ravel()/255
        means.append(mean);seconds.append(std**2+mean**2);low.append(rgb.min(axis=(0,1))/255);high.append(rgb.max(axis=(0,1))/255)
    cap.release();mean=np.mean(means,axis=0);var=np.maximum(0,np.mean(seconds,axis=0)-mean**2)
    def shape(x):return np.asarray(x).reshape(3,1,1).tolist()
    return {'min':shape(np.min(low,axis=0)),'max':shape(np.max(high,axis=0)),'mean':shape(mean),'std':shape(np.sqrt(var)),'count':[len(means)]}


def aggregate(episodes):
    result={}
    for key in episodes[0]:
        parts=[e[key] for e in episodes];counts=np.array([e['count'][0] for e in parts]);weights=counts/counts.sum()
        means=np.array([e['mean'] for e in parts]);stds=np.array([e['std'] for e in parts]);w=weights.reshape((-1,)+(1,)*(means.ndim-1))
        mean=np.sum(w*means,axis=0);var=np.maximum(0,np.sum(w*(stds**2+means**2),axis=0)-mean**2)
        result[key]={'min':np.min([e['min'] for e in parts],axis=0).tolist(),'max':np.max([e['max'] for e in parts],axis=0).tolist(),'mean':mean.tolist(),'std':np.sqrt(var).tolist(),'count':[int(counts.sum())]}
    return result


def link(src,dst):
    import shutil
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists():raise ValueError('Staged video already exists')
    shutil.copy2(src,dst)


def package(collection,output):
    from g1_pipette_sim.augmentation.export import catalog,publish
    return publish(catalog([collection]),output)


def _package(collection,output,entries=None):
    collection=Path(collection) if collection is not None else None;output=Path(output);meta=output/'meta';meta.mkdir(parents=True,exist_ok=True)
    if entries is None:raise ValueError('Explicit catalog entries required')
    accepted=list(range(len(entries)))
    if not accepted:return 0
    from g1_pipette_sim.augmentation.sources import load_episode
    ep=load_episode(entries[0]['candidate']);joint_names=ep.names
    features={key:{'dtype':'float32','shape':[29],'names':joint_names} for key in ['observation.state','action']}
    for side in ['left','right']:features['action.hand_'+side]={'dtype':'float32','shape':[6],'names':['pinky','ring','middle','index','thumb_bend','thumb_rotation']}
    from g1_pipette_sim.dynamics import build_dynamic_xml
    import xml.etree.ElementTree as ET
    first_config=json.loads((Path(entries[0]['folder'])/'config.json').read_text())
    xml=ET.fromstring(build_dynamic_xml(first_config))
    hand_names=[node.get('joint') for node in xml.find('actuator') if node.get('joint','').startswith(('L_','R_'))]
    if len(hand_names)!=12:raise ValueError('Unexpected hand actuator order')
    features['observation.hand_joint_positions']={'dtype':'float32','shape':[12],'names':hand_names}
    for key in ['diagnostics.tip_position','diagnostics.nozzle_position']:features[key]={'dtype':'float32','shape':[3],'names':['x','y','z']}
    for key in ['timestamp','diagnostics.simulation_time']:features[key]={'dtype':'float32','shape':[1],'names':None}
    for key in ['episode_index','frame_index','index','task_index']:features[key]={'dtype':'int64','shape':[1],'names':None}
    for key in ['next.done','next.success','diagnostics.tip_attached']:features[key]={'dtype':'bool','shape':[1],'names':None}
    for name,(w,h) in SIZES.items():features['observation.images.'+name]={'dtype':'video','shape':[h,w,3],'names':['height','width','channels'],'video_info':{'video.fps':FPS,'video.codec':'mpeg4','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}
    episodes=[];episode_stats=[];provenance=[];global_index=0
    mapping={}
    existing=meta/'augmentation.jsonl'
    if existing.exists():mapping={r['candidate_number']:r['episode_index'] for r in [json.loads(line) for line in existing.read_text().splitlines()]}
    # Acceptance order is append-only, so dataset episode indices remain stable.
    for index,number in enumerate(accepted):
        if number in mapping and mapping[number]!=index:raise ValueError('Acceptance order changed')
        folder=Path(entries[index]['folder']);record=json.loads((folder/'accepted.json').read_text());candidate=record['candidate']
        if not record['validation']['pickup'] or not record['independently_replayed']:raise ValueError('Unvalidated candidate')
        data=np.load(folder/'trajectory.npz');n=len(data['timestamp']);chunk=index//1000
        if n!=record['validation']['frame_count']:raise ValueError('Trajectory/video frame count mismatch')
        columns={key:data[key].astype(np.float32) for key in ['observation.state','action','action.hand_left','action.hand_right','observation.hand_joint_positions']}
        columns.update({'timestamp':data['timestamp'].astype(np.float32),'diagnostics.simulation_time':data['simulation_time'].astype(np.float32),'diagnostics.tip_position':data['tip_position'].astype(np.float32),'diagnostics.nozzle_position':data['nozzle_position'].astype(np.float32),'diagnostics.tip_attached':data['tip_attached'].astype(bool),'episode_index':np.full(n,index,dtype=np.int64),'frame_index':np.arange(n,dtype=np.int64),'index':np.arange(global_index,global_index+n,dtype=np.int64),'task_index':np.zeros(n,dtype=np.int64),'next.done':np.arange(n)==n-1,'next.success':np.arange(n)==n-1})
        path=output/f'data/chunk-{chunk:03}/episode_{index:06}.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise ValueError('Staging output already exists')
        if True:
            table=pa.table({key:pa.array(value.tolist(),type=pa.list_(pa.float32(),value.shape[1])) if value.ndim==2 else pa.array(value) for key,value in columns.items()});pq.write_table(table,path,compression='zstd')
        if pq.read_metadata(path).num_rows!=n:raise ValueError('Bad parquet length')
        statpath=meta/'episode_stats_cache'/f'{index:06}.json';statpath.parent.mkdir(exist_ok=True)
        if statpath.exists():st=json.loads(statpath.read_text())
        else:st={key:stats(value) for key,value in columns.items()}
        for name in SIZES:
            src=folder/(name+'.mp4');dst=output/f'videos/chunk-{chunk:03}/observation.images.{name}/episode_{index:06}.mp4';link(src,dst)
            if 'observation.images.'+name not in st:st['observation.images.'+name]=image_stats(src)
        write_json(statpath,st);episode_stats.append(st)
        episodes.append({'episode_index':index,'tasks':[TASK],'length':n})
        provenance.append({'episode_index':index,'capture_id':entries[index]['capture_id'],'lineage':candidate.get('lineage',candidate),'candidate_number':candidate['number'],'parent_episode':candidate['parent_episode'],'split_group':record['training_split_group'],'kind':candidate['kind'],'cell':candidate.get('cell',json.loads((folder/'config.json').read_text())['rack']['active_cell']),'target_shift_m':candidate.get('target_shift_m'),'world_lift_mm':record['validation']['sim_lift_mm'] if 'sim_lift_mm' in record['validation'] else record['validation']['lift_m']*1000,'reference_lift_mm':record['validation'].get('lift_target_m')*1000 if record['validation'].get('lift_target_m') is not None else None,'source_capture':str(folder.resolve()),'config_sha256':sha256(folder/'config.json'),'commands_sha256':sha256(folder/'commands.npz'),'validation_sha256':sha256(folder/'verification.json')})
        global_index+=n
    for name,rows in [('episodes.jsonl',episodes),('episodes_stats.jsonl',[{'episode_index':i,'stats':st} for i,st in enumerate(episode_stats)]),('tasks.jsonl',[{'task_index':0,'task':TASK}]),('augmentation.jsonl',provenance)]:
        tmp=meta/(name+'.tmp');tmp.write_text(''.join(json.dumps(r)+'\n' for r in rows));tmp.replace(meta/name)
    write_json(meta/'stats.json',aggregate(episode_stats))
    info={'codebase_version':'v2.1','robot_type':'unitree_g1_inspire_sim_motor_targets','total_episodes':len(episodes),'total_frames':global_index,'total_tasks':1,'total_videos':3*len(episodes),'total_chunks':math_ceil(len(episodes)/1000),'chunks_size':1000,'fps':FPS,'splits':{'train':f'0:{len(episodes)}'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet','video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features}
    partitions=[entry['candidate'].get('partition') for entry in entries]
    if all(p in ('train','val') for p in partitions):
        # Storage order must group each partition into one contiguous span.
        spans={}
        for membership in dict.fromkeys(partitions):
            indices=[i for i,p in enumerate(partitions) if p==membership]
            if indices!=list(range(indices[0],indices[-1]+1)):
                raise ValueError('Export catalog must be grouped by partition')
            spans[membership]=f'{indices[0]}:{indices[-1]+1}'
        info['splits']=spans
        write_json(meta/'parent-splits.json',{'episodes':[{'episode_index':i,'parent_episode':e['candidate']['parent_episode'],'partition':partitions[i],'capture_id':e['capture_id'],'family':e['candidate'].get('family')} for i,e in enumerate(entries)]})
    write_json(meta/'modality.json',{
        'state':{'body':{'start':0,'end':29,'original_key':'observation.state'}},
        'action':{'body':{'start':0,'end':29,'original_key':'action'},
                  'hand_left':{'start':0,'end':6,'original_key':'action.hand_left'},
                  'hand_right':{'start':0,'end':6,'original_key':'action.hand_right'}},
        'video':{name:{'original_key':'observation.images.'+name} for name in SIZES},
        'annotation':{'human.task_description':{'original_key':'task_index'}}})
    write_json(meta/'info.json',info)
    write_json(meta/'training-contract.json',{'action':'29 motor targets in radians; NOT 35-D high-level goals','hand_actions':'Two six-register command vectors; separate from main action','state':'29 simulated measured robot joints in radians','hand_state':'12 simulated actuated finger angles in radians; not real frozen hand-register feedback','camera_roles':{'rgb':'fixed ego','wrist_right':'fixed side','wrist_left':'right wrist'},'fps':FPS,'command_replay':'Dense original-clock augmented motor targets remain in each source_capture/commands.npz; exported actions sample those targets at the recorded simulation time. Do not assume 30 Hz zero-order hold exactly reproduces the dense replay.','image_stats':'Measured from up to 12 evenly spaced frames per episode/view; numeric stats use all rows.','split_warning':'The train span is a storage index, not a held-out split. Group splits by parent_episode across real and simulated data.','privileged_fields':'diagnostics.* must be excluded from image/state policy inputs unless explicitly intended.','simulation_limits':['Approximate camera calibration and wrist/tool/hand mechanics','Fixed pelvis','Geometric tip attachment constraint','Success establishes this simulator task, not real-world transfer']})
    print(json.dumps({'episodes':len(episodes),'frames':global_index,'output':str(output.resolve())}),flush=True)
    return len(episodes)


def math_ceil(x):return int(np.ceil(x))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,default=ROOT/'outputs/hil194/augmentation600');p.add_argument('--output',type=Path,default=ROOT/'outputs/hil194/augmentation600/lerobot-v21');a=p.parse_args();package(a.collection,a.output)
