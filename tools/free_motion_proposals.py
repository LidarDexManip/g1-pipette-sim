"""Generate placement proposals from a tool trajectory without rack/tip contacts.

This is a proposal heuristic only, never validation. Saved candidate configs
retain every collision and attachment setting from the real replay model.
"""
import copy,json
from pathlib import Path
import numpy as np

def free_trace(base,episode,directory,factory):
    directory=Path(directory);path=directory/'free-motion-proposal.npy'
    if path.exists():return np.load(path)
    cfg=copy.deepcopy(base);cfg['attachment']['dwell_seconds']=1e9
    sim=factory(cfg,episode);m,d=sim.model,sim.data
    objects=[m.body(name).id for name in ['rack','waste','tip']]
    mask=np.isin(m.geom_bodyid,objects);m.geom_contype[mask]=0;m.geom_conaffinity[mask]=0
    rows=[];stride=max(1,round(.005/m.opt.timestep));count=0
    while sim.time<episode.duration-1e-9:
        sim.step();count+=1
        if count%stride:continue
        tip=d.body('tip');R=tip.xmat.reshape(3,3);mouth=tip.xpos+R[:,2]*base['pipette']['tip_length']/2
        nozzle=d.site('nozzle');angle=float(np.degrees(np.arccos(np.clip(nozzle.xmat.reshape(3,3)[2,2],-1,1))))
        rows.append([sim.time,*nozzle.xpos,*mouth,angle])
    trace=np.asarray(rows);np.save(path,trace)
    (directory/'free-motion-proposal.json').write_text(json.dumps({'scope':'Proposal only: rack, tip and lid collisions disabled; attachment disabled. Never a validated demonstration.','episode':episode.index,'dataset':str(episode.root),'rows':len(rows)},indent=2))
    return trace

def proposals(trace,base):
    from g1_pipette_sim.scene import rack_target
    target=base['pickup_fit_criteria']['target_m'];mouth=rack_target(base)
    future=np.maximum.accumulate(trace[::-1,3])[::-1]-trace[:,3]
    delta=np.c_[trace[:,1:3]-mouth[:2],trace[:,3]+.011-mouth[2],np.zeros(len(trace))]
    speed=np.linalg.norm(np.gradient(trace[:,1:3],axis=0),axis=1)/.005
    rank=np.argsort(abs(future-target)+.002*np.maximum(trace[:,7]-5,0)+.03*speed)
    result=[]
    for i in rank:
        v=delta[i]
        if abs(v[2])>.02 or np.linalg.norm(v[:2])>.12:continue
        if all(np.linalg.norm(v[:3]-old[:3])>.002 for old in result):result.append(v)
        if len(result)>=8:break
    return result
