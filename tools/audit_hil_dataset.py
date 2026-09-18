"""Audit every numeric row and video endpoint without altering the source dataset."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import resolve_dataset_root


def array(column):
    c=column.combine_chunks()
    return c.values.to_numpy().reshape(len(c),-1) if pa.types.is_list(c.type) or pa.types.is_fixed_size_list(c.type) else c.to_numpy().reshape(-1,1)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=resolve_dataset_root(a.dataset);info=json.loads((root/'meta/info.json').read_text());episodes=[];global_stats={};errors=[]
    for i in range(info['total_episodes']):
        path=root/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i)
        tab=pq.read_table(path);entry={'episode':i,'frames':len(tab),'channels':{},'videos':{}}
        for key in tab.column_names:
            v=array(tab[key]);finite=np.isfinite(v);lo=np.min(v,axis=0);hi=np.max(v,axis=0)
            if not finite.all():errors.append({'episode':i,'column':key,'nonfinite':int((~finite).sum())})
            frozen=bool(np.array_equal(lo,hi));entry['channels'][key]={'frozen':frozen,'nonfinite':int((~finite).sum())}
            if v.shape[1]<50:entry['channels'][key].update(min=lo.tolist(),max=hi.tolist())
            if key not in global_stats:global_stats[key]={'min':lo,'max':hi,'frozen_episodes':[]}
            s=global_stats[key];s['min']=np.minimum(s['min'],lo);s['max']=np.maximum(s['max'],hi)
            if frozen:s['frozen_episodes'].append(i)
        image=array(tab['observation.t_img_ms']).ravel();action=array(tab['observation.t_action_ms']).ravel();delta=np.diff(image)
        entry['timing']={'duration_s':float((image[-1]-image[0])/1000),'image_dt_ms_quantiles':np.quantile(delta,[0,.5,.95,1]).tolist(),'nonincreasing_images':int(np.sum(delta<=0)),'nonincreasing_actions':int(np.sum(np.diff(action)<=0)),'action_age_ms_quantiles':np.quantile(image-action,[0,.5,.95,1]).tolist(),'nominal_duration_s':float((len(tab)-1)/info['fps'])}
        state=array(tab['observation.state'])[:,:29]
        for source in ['action','action.low_level']:
            difference=array(tab[source])[:,:29]-state;entry[source+'_minus_state_right_arm_rmse_rad']=float(np.sqrt(np.mean(difference[:,22:29]**2)))
        for view in ['rgb','wrist_left','wrist_right']:
            path=root/info['video_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i,video_key='observation.images.'+view)
            cap=cv2.VideoCapture(str(path));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));wh=[int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))];fps=cap.get(cv2.CAP_PROP_FPS)
            first,_=cap.read();cap.set(cv2.CAP_PROP_POS_FRAMES,max(n-1,0));last,_=cap.read();cap.release()
            entry['videos'][view]={'frames':n,'wh':wh,'fps':fps,'first_decodable':first,'last_decodable':last}
            if n!=len(tab) or not first or not last:errors.append({'episode':i,'video':view,'details':entry['videos'][view]})
        episodes.append(entry)
        if i%10==0:print(f'audited {i+1}/{info["total_episodes"]}',flush=True)
    for s in global_stats.values():s['min']=s['min'].tolist();s['max']=s['max'].tolist()
    result={'dataset':str(root),'episodes':episodes,'channels':global_stats,'errors':errors,'total_frames':sum(e['frames'] for e in episodes),'video_check':'Container frame counts, resolution, and first/last decoded frames; not full bitstream decode.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print('Saved',a.output,'errors',len(errors),flush=True)
if __name__=='__main__':main()
