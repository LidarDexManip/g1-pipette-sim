"""Check every exported episode against its capture and native-loader boundaries."""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('HF_DATASETS_OFFLINE','1')
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.fit_pickup_layouts import write_json


def audit(collection=None, v2=None, v3=None, output=None):
 root=Path(collection) if collection is not None else None
 v2=Path(v2) if v2 is not None else root/'lerobot-v21'
 v3=Path(v3) if v3 is not None else root/'lerobot-v3'
 info=json.loads((v2/'meta/info.json').read_text());rows=[json.loads(line) for line in (v2/'meta/augmentation.jsonl').read_text().splitlines()]
 if len(rows)!=info['total_episodes']:raise ValueError('Episode metadata count differs')
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 dataset=LeRobotDataset('local/g1-pipette-sim-motor-aug',root=v3,video_backend='pyav',delta_timestamps={'action':[0,1/30,2/30]})
 numeric_names=['action','observation.state','action.hand_left','action.hand_right','observation.hand_joint_positions']
 native_numeric=dataset.hf_dataset.select_columns(numeric_names).with_format(None)
 checked=[];total=0
 for item in rows:
  index=item['episode_index'];capture=Path(item['source_capture'])
  p=v2/info['data_path'].format(episode_chunk=index//info['chunks_size'],episode_index=index)
  table=pq.read_table(p);n=table.num_rows
  with np.load(capture/'trajectory.npz') as source, np.load(capture/'commands.npz') as commands:
   if n!=len(source['timestamp']):raise ValueError('Frame count differs from validated capture')
   for name in numeric_names:
    values=np.asarray(table[name].to_pylist(),dtype=np.float32)
    if not np.array_equal(values,source[name].astype(np.float32)):raise ValueError(f'{index}: exported {name} differs from capture')
   sampled=np.stack([np.interp(source['simulation_time'],commands['times'],commands['motor_targets'][:,j]) for j in range(29)],axis=1)
   if not np.allclose(sampled,source['action'],atol=1e-10,rtol=1e-10):raise ValueError('Action labels do not match executed dense commands')
   meta=dataset.meta.episodes[index];start=int(meta['dataset_from_index']);end=int(meta['dataset_to_index'])
   if end-start!=n or start!=total:raise ValueError('Native dataset offsets differ')
   native_rows=native_numeric[start:end]
   for name in numeric_names:
    if not np.array_equal(np.asarray(native_rows[name],dtype=np.float32),source[name].astype(np.float32)):raise ValueError(f'{index}: native full-episode {name} differs from capture')
   for pos,local in [(start,0),(end-1,n-1)]:
    frame=dataset[pos]
    if tuple(frame['action'].shape)!=(3,29):raise ValueError('Wrong native action chunk')
    if not np.allclose(frame['observation.state'].numpy(),source['observation.state'][local],atol=1e-6):raise ValueError('Native state differs')
    if not np.allclose(frame['action'][0].numpy(),source['action'][local],atol=1e-6):raise ValueError('Native current action differs')
    for key in ['rgb','wrist_right','wrist_left']:
     name='observation.images.'+key;pixels=frame[name].numpy();shape=info['features'][name]['shape']
     if pixels.shape!=(3,shape[0],shape[1]) or not np.isfinite(pixels).all():raise ValueError('Invalid native camera frame')
     import cv2
     cap=cv2.VideoCapture(str(capture/(key+'.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,local);ok,original=cap.read();cap.release()
     if not ok:raise ValueError('Cannot decode reference capture frame')
     reference=cv2.cvtColor(original,cv2.COLOR_BGR2RGB).astype(np.float32)/255
     pixel_error=float(np.mean(abs(pixels.transpose(1,2,0)-reference)))
     if pixel_error>1/255:raise ValueError(f'{index}: native {key} view/time does not match capture (MAE {pixel_error})')
    if local==n-1:
     if not np.allclose(frame['action'].numpy(),np.broadcast_to(source['action'][-1],(3,29)),atol=1e-6):raise ValueError('Future action chunk crosses an episode boundary')
   checked.append(index);total+=n
  if len(checked)%25==0:print('AUDITED EPISODES',len(checked),flush=True)
 if total!=len(dataset) or total!=info['total_frames']:raise ValueError('Dataset frame totals differ')
 result={'complete':True,'episodes_checked':len(checked),'frames_checked_numerically':total,'native_camera_frames_checked':6*len(checked),'all_accepted_videos_previously_decoded_in_full':True,'checks':['all numeric v2.1 and native v3.0 exported labels versus validated captures','all motor labels versus dense replay commands','every native episode first/last state and all three camera images versus source pixels','future action padding at every episode end','global frame offsets and totals']}
 write_json(Path(output) if output is not None else root/'dataset-audit.json',result);return result

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',nargs='?',default='outputs/hil194/augmentation600');a=p.parse_args();print(json.dumps(audit(a.collection),indent=2))
