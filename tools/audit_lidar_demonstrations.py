"""Audit replay channels of the two-view five-task dataset without modifying data."""
import argparse,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    info=json.loads((a.dataset/'meta/info.json').read_text());notes=json.loads((a.dataset/'meta/source_notes.json').read_text());rows=[]
    for i in range(info['total_episodes']):
        keys=['observation.state','action.low_level','action','observation.t_img_ms','observation.t_action_ms','observation.state.hand_left','observation.state.hand_right','action.hand_left','action.hand_right']
        table=pq.read_table(a.dataset/info['data_path'].format(episode_chunk=i//info['chunks_size'],episode_index=i),columns=keys)
        d={k:np.asarray(table[k].to_pylist()) for k in keys};t=d['observation.t_img_ms'].ravel();at=d['observation.t_action_ms'].ravel()
        rows.append({'episode':i,'task':notes['task_types']['by_episode_index'][str(i)],'frames':len(t),'duration_s':float((t[-1]-t[0])/1000),
                     'finite':all(np.isfinite(v).all() for v in d.values()),'image_times_strict':bool(np.all(np.diff(t)>0)),
                     'action_clock_frozen':bool(np.ptp(at)==0),'motor_target_vs_state_rmse_rad':float(np.sqrt(np.mean((d['action.low_level']-d['observation.state'][:,:29])**2))),
                     'high_level_vs_motor_rmse_rad':float(np.sqrt(np.mean((d['action'][:,:29]-d['action.low_level'])**2))),
                     'frozen_hand_feedback':{s:bool(np.max(np.ptp(d[f'observation.state.hand_{s}'],axis=0))==0) for s in ['left','right']}})
    summary={'dataset':str(a.dataset.resolve()),'revision':a.dataset.name,'episodes':len(rows),'frames':sum(r['frames'] for r in rows),'fps':info['fps'],
             'task_counts':{task:sum(r['task']==task for r in rows) for task in sorted(set(r['task'] for r in rows))},
             'views':{k:v['shape'] for k,v in info['features'].items() if v['dtype']=='video'},
             'nonfinite_episodes':[r['episode'] for r in rows if not r['finite']],
             'nonmonotonic_image_episodes':[r['episode'] for r in rows if not r['image_times_strict']],
             'frozen_action_clock_episodes':sum(r['action_clock_frozen'] for r in rows),
             'frozen_hand_feedback_episodes':{s:sum(r['frozen_hand_feedback'][s] for r in rows) for s in ['left','right']},
             'motor_target_vs_state_median_rmse_rad':float(np.median([r['motor_target_vs_state_rmse_rad'] for r in rows])),
             'high_level_vs_motor_median_rmse_rad':float(np.median([r['high_level_vs_motor_rmse_rad'] for r in rows])),
             'episodes_detail':rows}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='episodes_detail'},indent=2))

if __name__=='__main__':main()
