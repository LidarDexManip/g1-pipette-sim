"""Decode every video frame with FFmpeg; record errors without changing source files."""
import argparse,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.dataset import Episode


def check(item):
    index,view,path=item
    try:
        r=subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(path),'-map','0:v:0','-f','null','-'],capture_output=True,text=True,timeout=300)
        return {'episode':index,'view':view,'ok':r.returncode==0,'returncode':r.returncode,'error':r.stderr[-2500:]}
    except Exception as error:return {'episode':index,'view':view,'ok':False,'error':repr(error)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();ep=Episode();jobs=[]
    for i in range(ep.info['total_episodes']):
        ep.index=i
        jobs.extend((i,v,ep.video_path(v)) for v in ['rgb','wrist_left','wrist_right'])
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(check,j) for j in jobs]):
            results.append(future.result())
            if len(results)%30==0:print(f'{len(results)}/{len(jobs)} videos, {sum(not r["ok"] for r in results)} failed',flush=True)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'checked':len(results),'expected':len(jobs),'results':sorted(results,key=lambda r:(r['episode'],r['view']))},indent=2))
if __name__=='__main__':main()
