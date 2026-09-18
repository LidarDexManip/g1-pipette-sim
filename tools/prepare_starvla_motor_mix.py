"""Prepare grouped, synchronized 30 Hz real/sim LeRobot v2.1 motor training copies."""
import argparse, concurrent.futures, hashlib, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode, DEFAULT_DATASET, resolve_dataset_root

ROOT=Path(__file__).resolve().parents[1]
FPS=30
SIZES={'rgb':(512,288),'wrist_right':(384,384),'wrist_left':(384,384)}
TASK='Attach a green pipette tip from the tip rack to the pipette.'

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def lines(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))

def nearest_indices(times,grid):
    right=np.searchsorted(times,grid).clip(0,len(times)-1);left=(right-1).clip(0)
    return np.where(abs(times[left]-grid)<=abs(times[right]-grid),left,right)

def interpolate(times,values,grid):
    return np.stack([np.interp(grid,times,values[:,i]) for i in range(values.shape[1])],axis=1)

def convert_video(src,dst,indices,size):
    import cv2
    cv2.setNumThreads(1)
    dst.parent.mkdir(parents=True,exist_ok=True);temp=dst.with_name(dst.stem+'.tmp.mp4')
    cap=cv2.VideoCapture(str(src));writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),FPS,size)
    if not writer.isOpened():raise RuntimeError(f'Cannot write {temp}')
    out=0;source_index=0
    try:
        while out<len(indices):
            ok,frame=cap.read()
            if not ok:raise ValueError(f'Truncated source video {src} at {source_index}')
            if source_index==indices[out]:
                frame=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
                while out<len(indices) and indices[out]==source_index:writer.write(frame);out+=1
            source_index+=1
    finally:cap.release();writer.release()
    # Decode the entire output before accepting this conversion.
    cap=cv2.VideoCapture(str(temp));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame.shape[:2]!=(size[1],size[0]):raise ValueError('Video shape mismatch')
        count+=1
    cap.release()
    if count!=len(indices):raise ValueError(f'Video count {count} != {len(indices)}')
    temp.replace(dst)

def job(spec):
    root=Path(spec['output']);index=spec['episode_index'];folder=root/'conversion_records';done=folder/f'{index:06}.json'
    if done.exists():raise ValueError('Unexpected conversion cache in fresh staging directory')
    if spec['domain']=='real':
        ep=Episode(spec['source'],spec['parent_episode']);times=np.arange(int(np.floor(ep.duration*FPS+1e-7))+1)/FPS
        picks=nearest_indices(ep.times,times)
        state=interpolate(ep.times,ep.arrays['observation.state'][:,:29],times)
        motor=interpolate(ep.action_times,ep.arrays['action.low_level'],times)
        hands=[interpolate(ep.action_times,ep.arrays['action.hand_'+side],times) for side in ['left','right']]
        videos={name:ep.video_path(name) for name in SIZES}
        clock={'source_image_time_s':ep.times[picks], 'sample_time_s':times,'source_frame_index':picks}
        timing_error=float(np.max(abs(ep.times[picks]-times)))
    else:
        src=Path(spec['source']);d=np.load(src/'trajectory.npz');times=d['timestamp'];picks=np.arange(len(times))
        state=d['observation.state'];motor=d['action'];hands=[d['action.hand_'+side] for side in ['left','right']]
        videos={name:src/(name+'.mp4') for name in SIZES}
        clock={'source_image_time_s':d['simulation_time'],'sample_time_s':times,'source_frame_index':picks}
        timing_error=float(np.max(abs(d['simulation_time']-times)))
    action=np.concatenate([motor,*hands],axis=1).astype(np.float32);state=state.astype(np.float32);n=len(times)
    if not np.isfinite(action).all() or not np.isfinite(state).all():raise ValueError('Nonfinite numeric data')
    if np.any(action[:,29:]<0) or np.any(action[:,29:]>1000):raise ValueError('Hand commands outside 0..1000')
    for name,size in SIZES.items():
        dst=root/f'videos/chunk-{index//1000:03}/observation.images.{name}/episode_{index:06}.mp4'
        if spec['domain']=='real':convert_video(videos[name],dst,picks,size)
        else:
            dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():
                shutil.copy2(videos[name],dst)
    columns={'observation.state':pa.array(state.tolist(),pa.list_(pa.float32(),29)),'action':pa.array(action.tolist(),pa.list_(pa.float32(),41)),
             'timestamp':pa.array(np.arange(n,dtype=np.float32)/FPS),'frame_index':pa.array(np.arange(n,dtype=np.int64)),
             'episode_index':pa.array(np.full(n,index,dtype=np.int64)),'task_index':pa.array(np.zeros(n,dtype=np.int64))}
    path=root/f'data/chunk-{index//1000:03}/episode_{index:06}.parquet';path.parent.mkdir(parents=True,exist_ok=True);pq.write_table(pa.table(columns),path,compression='zstd')
    folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/f'{index:06}-clock.npz',**clock)
    record={**spec,'length':n,'maximum_image_alignment_error_s':timing_error,'numeric_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    write(done,record);return record

def metadata(root,records,joints):
    records=sorted(records,key=lambda r:r['episode_index']);offset=0
    for r in records:
        path=root/f"data/chunk-{r['episode_index']//1000:03}/episode_{r['episode_index']:06}.parquet";table=pq.read_table(path)
        if 'index' in table.column_names:table=table.drop(['index'])
        table=table.append_column('index',pa.array(np.arange(offset,offset+r['length'],dtype=np.int64)));pq.write_table(table,path,compression='zstd');offset+=r['length']
        r['numeric_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        write(root/'conversion_records'/f"{r['episode_index']:06}.json",{k:v for k,v in r.items() if k!='output'})
    hands=[f'{side}_{name}' for side in ['left','right'] for name in ['pinky','ring','middle','index','thumb_bend','thumb_rot']]
    features={'observation.state':{'dtype':'float32','shape':[29],'names':joints},'action':{'dtype':'float32','shape':[41],'names':joints+hands}}
    for k in ['timestamp','frame_index','episode_index','task_index','index']:features[k]={'dtype':'float32' if k=='timestamp' else 'int64','shape':[1],'names':None}
    for name,(w,h) in SIZES.items():features['observation.images.'+name]={'dtype':'video','shape':[h,w,3],'names':['height','width','channels'],'info':{'video.fps':FPS,'video.codec':'mpeg4','video.pix_fmt':'yuv420p','video.is_depth_map':False,'has_audio':False}}
    info={'codebase_version':'v2.1','robot_type':'unitree_g1_inspire_motor_targets','fps':FPS,'total_episodes':len(records),'total_frames':offset,'total_tasks':1,'total_videos':len(records)*3,'total_chunks':(len(records)+999)//1000,'chunks_size':1000,'splits':{'train':f'0:{len(records)}'},'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet','video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4','features':features}
    modality={'state':{'body':{'start':0,'end':29,'original_key':'observation.state'}},'action':{'body':{'start':0,'end':29,'original_key':'action'},'hand_left':{'start':29,'end':35,'original_key':'action'},'hand_right':{'start':35,'end':41,'original_key':'action'}},'video':{name:{'original_key':'observation.images.'+name} for name in SIZES},'annotation':{'human.task_description':{'original_key':'task_index'}}}
    write(root/'meta/info.json',info);write(root/'meta/modality.json',modality)
    lines(root/'meta/episodes.jsonl',[{'episode_index':r['episode_index'],'tasks':[TASK],'length':r['length']} for r in records]);lines(root/'meta/tasks.jsonl',[{'task_index':0,'task':TASK}]);lines(root/'meta/provenance.jsonl',[{k:v for k,v in r.items() if k!='output'} for r in records])
    # Standard LeRobot statistics; starVLA builds its own versioned cache in this copy.
    stats={}
    for k in ['observation.state','action']:
        x=np.concatenate([np.asarray(pq.read_table(root/f"data/chunk-{r['episode_index']//1000:03}/episode_{r['episode_index']:06}.parquet",columns=[k])[k].to_pylist(),dtype=np.float32) for r in records])
        stats[k]={'min':x.min(0).tolist(),'max':x.max(0).tolist(),'mean':x.mean(0).tolist(),'std':x.std(0).tolist(),'q01':np.quantile(x,.01,axis=0).tolist(),'q99':np.quantile(x,.99,axis=0).tolist(),'count':[len(x)]}
    write(root/'meta/stats.json',stats)
    return {'episodes':len(records),'frames':offset,'maximum_image_alignment_error_s':max(r['maximum_image_alignment_error_s'] for r in records)}

def main():
    from g1_pipette_sim.augmentation.contracts import dependencies,digest,file_hash,lock,seal,verify,tree_files
    from g1_pipette_sim.augmentation.export import verify_catalog
    from g1_pipette_sim.augmentation.sources import source_files
    from g1_pipette_sim.augmentation.splits import parent_splits
    import tempfile
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--catalog',type=Path,required=True)
    p.add_argument('--real-dataset',type=Path,default=DEFAULT_DATASET)
    p.add_argument('--split-manifest',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260912)
    a=p.parse_args();a.output=a.output.resolve()
    real=resolve_dataset_root(a.real_dataset).resolve()
    value=json.loads(a.catalog.read_text());verify_catalog(value)
    count=json.loads((real/'meta/info.json').read_text())['total_episodes']
    split=parent_splits(a.split_manifest,real,count,a.seed)
    source_hashes={}
    for parent in range(count):
        ep=Episode(real,parent);source_hashes.update(source_files(ep))
        source_hashes.update({str(ep.video_path(name)):file_hash(ep.video_path(name)) for name in SIZES})
    for entry in value['entries']:
        c=entry['candidate']
        if Path(c['dataset_snapshot']).resolve()!=real or str(c['parent_episode']) not in split['parents']:
            raise ValueError('Simulation parent is outside the real snapshot/split contract')
    contract={'schema':2,'catalog':value,'real_files':source_hashes,'splits':split,
              'dependencies':dependencies(),'fps':FPS,'action':'29 motor radians + 12 hand registers',
              'hand_provenance':'Preserved source commands, including reconstructed/imputed taped-grasp collection labels'}
    with lock(a.output.parent/('.'+a.output.name+'-publish')):
        if a.output.exists():
            if not verify(a.output,'publication',contract):raise ValueError('Existing training output is unversioned; use a new version')
            print('Verified existing training export',a.output);return
        stage=Path(tempfile.mkdtemp(prefix='.'+a.output.name+'-',dir=a.output.parent))
        try:
            jobs=[];counts={}
            for domain,items in [('real',list(range(count))),('sim',value['entries'])]:
                for item in items:
                    parent=item if domain=='real' else item['candidate']['parent_episode']
                    membership=split['parents'][str(parent)];name=f'{domain}_{membership}'
                    index=counts.get(name,0);counts[name]=index+1
                    spec={'domain':domain,'parent_episode':parent,'split':membership,'episode_index':index,
                          'output':str(stage/name),'source':str(real) if domain=='real' else item['folder']}
                    if domain=='sim':spec['capture_id']=item['capture_id'];spec['lineage']=item['candidate'].get('lineage',item['candidate'])
                    jobs.append(spec)
            results=[]
            with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
                for result in pool.map(job,jobs):
                    results.append(result)
                    if len(results)%10==0:print('CONVERTED',len(results),'/',len(jobs),flush=True)
            summaries={};joints=Episode(real,0).names
            for name in counts:
                summaries[name]=metadata(stage/name,[r for r in results if Path(r['output']).name==name],joints)
            verify_catalog(value)
            for path,expected in source_hashes.items():
                if file_hash(path)!=expected:raise ValueError('Real source changed during conversion')
            write(stage/'preparation-contract.json',contract)
            write(stage/'progress.json',{'status':'prepared','converted':len(results),'datasets':summaries})
            write(stage/'suggested-mixture.json',{'real_train':0.5,'sim_train':0.5,
                  'note':'Suggested sampling weights, not an enforced trainer setting. Compare against real-only and natural-frequency mixtures. Validation stays separate.'})
            seal(stage,'publication',contract,tree_files(stage));os.rename(stage,a.output)
            print(json.dumps(summaries,indent=2),flush=True)
        except BaseException:
            shutil.rmtree(stage,ignore_errors=True);raise
if __name__=='__main__':main()
