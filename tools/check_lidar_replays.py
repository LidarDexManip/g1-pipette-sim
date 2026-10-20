"""Check measured-joint playback against the calibrated LIDAR scene (not task success)."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.environments.robot import build_robot_scene,load_registration,MeasuredReplay
from g1_pipette_sim.environments import ENVIRONMENTS,environment_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--episodes',type=int,nargs='+',default=[0,126,188,127,1]);p.add_argument('--output',type=Path,default=Path('outputs/lidar-teleop/replay-audit.json'));a=p.parse_args()
    cfg=load_registration();m=mujoco.MjModel.from_xml_string(build_robot_scene(registration=cfg));d=mujoco.MjData(m)
    manifest=environment_manifest('lidar');table=next(o for o in manifest['objects'] if o['name']=='Table_new');top=table['bounds_m'][2][1]
    rows=[]
    for i in a.episodes:
        e=Episode(a.dataset,i);r=MeasuredReplay(m,d,e);max_error=0.;heights=[];ranges=[];table_contact_frames=0;max_penetration=0.;penetration_by_body={}
        adr=np.array([m.joint(n).qposadr[0] for n in e.names])
        for frame,t in enumerate(e.times):
            r.pose(t);max_error=max(max_error,float(np.max(np.abs(d.qpos[adr]-e.arrays['observation.state'][frame,:29]))))
            palms=[]
            for side in ['left','right']:
                body=d.body(f'{side}_wrist_yaw_link');palms.append(body.xpos+body.xmat.reshape(3,3)@np.array([.09,0,0]))
            heights.append([palm[2] for palm in palms]);ranges.append(palms)
            touching=False
            for contact in d.contact:
                ga,gb=contact.geom1,contact.geom2
                table_id=m.body('env_07_table_new').id
                if ((m.geom_bodyid[ga]==table_id and m.geom_contype[gb]==2) or (m.geom_bodyid[gb]==table_id and m.geom_contype[ga]==2)) and contact.dist<-.001:
                    touching=True;max_penetration=max(max_penetration,float(-contact.dist))
                    robot_geom=gb if m.geom_bodyid[ga]==table_id else ga
                    body_name=m.body(m.geom_bodyid[robot_geom]).name
                    penetration_by_body[body_name]=max(penetration_by_body.get(body_name,0.),float(-contact.dist)*1000)
            table_contact_frames+=int(touching)
        heights=np.asarray(heights);ranges=np.asarray(ranges)
        rows.append({'episode':i,'duration_s':e.duration,'frames_replayed':len(e.times),'finite':bool(np.isfinite(ranges).all()),
                     'measured_joint_assignment_error_rad':max_error,'hand_sources':r.hand_sources,
                     'estimated_palm_height_range_m':{'left':[float(heights[:,0].min()),float(heights[:,0].max())],'right':[float(heights[:,1].min()),float(heights[:,1].max())]},
                     'tabletop_above_highest_palm_m':float(top-heights.max()),'robot_table_penetration_frames_over_1mm':table_contact_frames,'max_robot_table_penetration_mm':max_penetration*1000,'table_penetration_by_robot_body_mm':penetration_by_body,'task_success':None})
        print(json.dumps(rows[-1]),flush=True)
    report={'registration':cfg,'tabletop_z_m':top,'floor_z_m':manifest['floor']['z_m'],'table_height_m':table['dimensions_world_m'][2],
            'replays':rows,'task_success_evaluated':False,'limitations':['Measured joints prescribed exactly, not a dynamic tracking result.',
            'No measured base translation/yaw in dataset; fixed base is an estimate.',
            'Palm height uses a 9 cm offset from wrist, not a calibrated grasp point.',
            'All objects static; pick, place, aim and contact success cannot be evaluated.',
            'Scene scale is calibrated; task and contact success still require validation.']}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
