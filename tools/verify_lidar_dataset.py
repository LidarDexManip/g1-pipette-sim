"""Verify all downloaded trajectory/video files against the LeRobot descriptor."""
import argparse,json
from pathlib import Path
import cv2
import pyarrow.parquet as pq


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    info=json.loads((a.dataset/'meta/info.json').read_text());episodes=[json.loads(line) for line in (a.dataset/'meta/episodes.jsonl').read_text().splitlines()]
    errors=[];videos=0;frames=0;trajectory_bytes=0;video_bytes=0
    for ep in episodes:
        i=ep['episode_index'];fmt={'episode_chunk':i//info['chunks_size'],'episode_index':i}
        path=a.dataset/info['data_path'].format(**fmt)
        if not path.is_file():errors.append(f'Missing {path}');continue
        n=pq.ParquetFile(path).metadata.num_rows;frames+=n;trajectory_bytes+=path.stat().st_size
        if n!=ep['length']:errors.append(f'Episode {i} rows {n} != {ep["length"]}')
        for key,feature in info['features'].items():
            if feature['dtype']!='video':continue
            path=a.dataset/info['video_path'].format(**fmt,video_key=key)
            if not path.is_file():errors.append(f'Missing {path}');continue
            video_bytes+=path.stat().st_size;cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
            got=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),round(cap.get(cv2.CAP_PROP_FPS),4))
            expected=(n,feature['shape'][1],feature['shape'][0],float(info['fps']))
            if not cap.isOpened() or got!=expected:errors.append(f'{path}: {got} != {expected}')
            cap.release();videos+=1
    if frames!=info['total_frames']:errors.append(f'Total frames: {frames}')
    if videos!=info['total_videos']:errors.append(f'Total videos: {videos}')
    result={'dataset':str(a.dataset.resolve()),'revision':a.dataset.name,'episodes':len(episodes),'frames':frames,'videos':videos,
            'trajectory_bytes':trajectory_bytes,'video_bytes':video_bytes,'verification':'Parquet row counts and video headers (resolution, FPS, frame count); not full video decode','errors':errors}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));raise SystemExit(bool(errors))

if __name__=='__main__':main()
