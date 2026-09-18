"""Bounded layout refit for the shared torque-bias hypothesis; never auto-publish."""
import argparse,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from g1_pipette_sim.pickup_layouts import ROOT,MANIFEST,model_fingerprints,sha256
from g1_pipette_sim.dataset import Episode
from tools.diagnose_replay_divergence import EffortBiasSimulation
from tools.free_motion_proposals import free_trace
from tools.retry_world_pickups import window_proposals,move_well,allowed,placement_key
from tools.publish_world_pickup_layouts import config_for_row
from tools.fit_pickup_layouts import write_json


def run_trial(job):
    cfg,index=job
    import tools.fit_pickup_layouts as fit
    fit.candidate_simulation=lambda c,e,full=False:EffortBiasSimulation(c,e,source='low_level')
    return fit.rollout(cfg,Episode(index=index),full=True,lift_frame='world')


def prepare(index,out):
    entry=json.loads(MANIFEST.read_text())['episodes'][str(index)]
    current=json.loads((ROOT/entry['config']).read_text());base=json.loads((ROOT/f'outputs/hil194/pickup-layouts/episode-{index:03}/baseline.json').read_text())
    origin={'delta':current['layout_search']['delta_xy_z_yaw'],'active_cell':current['rack']['active_cell']}
    folder=Path(out)/f'episode-{index:03}';folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps({'config':current,'model':model_fingerprints(),'regression':sha256(ROOT/'outputs/hil194/pickup-layouts/servo-gain-check.json')},sort_keys=True).encode()).hexdigest()
    stamp=folder/'proposal-provenance.json'
    if (folder/'free-motion-proposal.npy').exists() and (not stamp.exists() or json.loads(stamp.read_text()).get('signature')!=signature):raise ValueError('Proposal provenance changed or missing; use a new output directory')
    trace=free_trace(current,Episode(index=index),folder,lambda c,e:EffortBiasSimulation(c,e,source='low_level'))
    write_json(stamp,{'signature':signature})
    seeds=window_proposals(trace,base,origin)
    pending=seeds[:4]
    for cell in [(0,0),(0,base['rack']['cols']-1),(base['rack']['rows']-1,0),(base['rack']['rows']-1,base['rack']['cols']-1)]:
        pending.append(move_well(base,seeds[0] if seeds else origin,cell))
    pending.extend(seeds[4:]);unique=[];seen=set()
    for row in pending:
        if allowed(base,row) and placement_key(row) not in seen:
            unique.append(row);seen.add(placement_key(row))
        if len(unique)>=8:break
    return base,unique


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episodes',nargs='+',type=int,default=[46,52,160,193]);p.add_argument('--workers',type=int,default=4);p.add_argument('--output',type=Path,default=ROOT/'outputs/hil194/bias-corrected-layouts');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text())
    if manifest['model_fingerprints']!=model_fingerprints():raise ValueError('Stale manifest')
    write_json(a.output/'input-manifest.json',manifest)
    write_json(a.output/'experiment.json',{'script_sha256':sha256(Path(__file__)),'regression_sha256':sha256(ROOT/'outputs/hil194/pickup-layouts/servo-gain-check.json'),'episodes':a.episodes,'runtime_intervention':'EffortBiasSimulation; saved trial config is only the scene input','published':False})
    jobs=[];summary={'results':[],'errors':[],'scope':'Experimental shared torque bias plus layout refit; no default changes. Existing insertion, contact and world-lift criteria retained.'}
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(prepare,i,str(a.output)):i for i in a.episodes}
        for f in as_completed(futures):
            index=futures[f];base,rows=f.result()
            for trial,row in enumerate(rows):jobs.append((index,trial,row,config_for_row(base,row)))
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(run_trial,(cfg,index)):(index,trial,row,cfg) for index,trial,row,cfg in jobs}
        for f in as_completed(futures):
            index,trial,row,cfg=futures[f]
            try:
                result,trace,report=f.result();folder=a.output/f'episode-{index:03}'
                record={'episode':index,'trial':trial,'placement':row,**result}
                write_json(folder/f'trial-{trial:02}-config.json',cfg);write_json(folder/f'trial-{trial:02}-report.json',report);write_json(folder/f'trial-{trial:02}-result.json',record)
                summary['results'].append(record);print(json.dumps(record),flush=True)
            except Exception as error:summary['errors'].append({'episode':index,'trial':trial,'error':repr(error)})
            write_json(a.output/'summary.json',summary)
    if summary['errors']:raise SystemExit(1)
if __name__=='__main__':main()
