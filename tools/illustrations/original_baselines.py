"""Render the 12 original real-command baseline replays, never augmented commands."""
import argparse,concurrent.futures,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from g1_pipette_sim.camera_views import apply_camera_views
EPISODES=[6,13,14,24,30,35,40,41,45,64,68,144]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2)+'\n');t.replace(p)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=4);p.add_argument('--physics',choices=['dynamic','kinematic'],default='dynamic');p.add_argument('--source',choices=['low_level','state'],default='low_level');p.add_argument('--output',type=Path);a=p.parse_args();out=(a.output or ROOT/('outputs/hil194/original-baseline-comparisons'+('-kinematic-state' if a.physics=='kinematic' else ('-state' if a.source=='state' else '')))).resolve();out.mkdir(parents=True,exist_ok=True)
 if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text()).get('source','low_level')!=a.source:raise ValueError('Output directory belongs to another replay source')
 if a.physics=='kinematic' and a.source!='state':raise ValueError('Kinematic diagnostics require --source state')
 if (out/'progress.json').exists() and json.loads((out/'progress.json').read_text()).get('physics','dynamic')!=a.physics:raise ValueError('Output belongs to another physics mode')
 manifest_path=ROOT/'config/hil194-pickup-layouts.json';manifest=json.loads(manifest_path.read_text());jobs=[]
 for i in EPISODES:
  e=manifest['episodes'][str(i)];source_cfg=ROOT/e['config'];source_report=ROOT/e['report'];assert sha(source_cfg)==e['sha256']
  cfg=json.loads(source_cfg.read_text());report=json.loads(source_report.read_text());assert report['source']=='low_level' and report['episode']==i
  cfg=apply_camera_views(cfg,SimpleNamespace(root=Path(report['dataset']),index=i));folder=out/f'episode-{i:03d}';folder.mkdir(exist_ok=True)
  report['video_frames']=math.floor(report['time']*30+1e-6)+1
  write(folder/'workspace.json',cfg);write(folder/'report.json',report)
  provenance={'episode':i,'source':a.source,'physics':a.physics,'source_semantics':('Direct measured joint-position playback; prescribed tool mount; geometric attachment' if a.physics=='kinematic' else ('Measured body positions as servo targets with original hand commands' if a.source=='state' else 'Original recorded motor targets; no augmentation')),'baseline_config':str(source_cfg),'baseline_config_sha256':sha(source_cfg),'baseline_report':str(source_report),'baseline_report_sha256':sha(source_report),'manifest_sha256':sha(manifest_path),'camera_view_correction':cfg['camera_view_correction'],'rack_contact':bool(report['contact_diagnostics'].get('pipette/rack',{}).get('samples',0))}
  write(folder/'baseline-provenance.json',provenance);jobs.append((i,folder,provenance))
 results={};errors={};started=time.time()
 def progress():write(out/'progress.json',{'source':a.source,'physics':a.physics,'complete':len(results)==len(jobs) and not errors,'completed':len(results),'requested':EPISODES,'errors':errors,'recordings':list(results.values()),'elapsed_s':time.time()-started})
 def render(job):
  i,folder,prov=job;receipt=folder/'highres-render.json';video=folder/'simulation.mp4'
  if receipt.exists() and video.exists():
   r=json.loads(receipt.read_text())
   if r.get('workspace_sha256')==sha(folder/'workspace.json') and r.get('video_sha256')==sha(video) and r.get('source','low_level')==a.source and r.get('physics','dynamic')==a.physics and r.get('reference_report_sha256')==sha(folder/'report.json'):return {'episode':i,'video':str(video.relative_to(out)),'frames':r['frames_verified'],'rack_contact':(bool(r['replay_report']['kinematic_diagnostics']['overlap_frames']) if a.physics=='kinematic' else bool(r['replay_report']['contact_diagnostics'].get('pipette/rack',{}).get('samples',0))),'source':a.source}
  env=dict(os.environ,MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
  with (folder/'render.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'tools/illustrations/real_sim_comparison.py'),'--folder',str(folder),'--source',a.source,'--physics',a.physics],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=3600)
  r=json.loads(receipt.read_text());return {'episode':i,'video':str(video.relative_to(out)),'frames':r['frames_verified'],'rack_contact':(bool(r['replay_report']['kinematic_diagnostics']['overlap_frames']) if a.physics=='kinematic' else bool(r['replay_report']['contact_diagnostics'].get('pipette/rack',{}).get('samples',0))),'source':a.source}
 progress()
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  futures={pool.submit(render,j):j[0] for j in jobs}
  for f in concurrent.futures.as_completed(futures):
   i=futures[f]
   try:results[i]=f.result();print(f'{len(results)}/12 episode {i} verified',flush=True)
   except Exception as error:errors[str(i)]=repr(error);print(f'ERROR episode {i}: {error}',flush=True)
   progress()
 links=''.join(f'<tr><td>{i}</td><td>{"Tool/rack overlap or contact" if results.get(i,{}).get('rack_contact') else "None observed"}</td><td><a href="episode-{i:03d}/simulation.mp4">Video</a></td></tr>' for i,folder,prov in jobs)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Original baseline comparisons</title><style>body{font:18px sans-serif;max-width:1000px;margin:40px auto;background:#14202c;color:#edf2f6}td,th{padding:12px;text-align:left}a{color:#80d9df}</style><h1>Original replay baselines</h1><p>'+('Direct measured-position playback (kinematic/state). Verdict means sampled geometric overlap, not dynamic contact or physical task success.' if a.physics=='kinematic' else ('Measured joint positions used as dynamic servo targets (source=state).' if a.source=='state' else 'Original recorded motor commands (source=low_level).'))+' Top: real; bottom: simulator. Columns: ego, right wrist, fixed side. Corrected camera poses. 3240×2160, 30 fps; no video labels.</p><table><tr><th>Real episode</th><th>Simulation contact verdict</th><th>Comparison</th></tr>'+links+'</table>')
 if errors:raise SystemExit(1)

if __name__=='__main__':main()
