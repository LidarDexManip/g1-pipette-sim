"""Read-only geometric checks for prescribed motion; not force/contact validation."""
import mujoco
import numpy as np

class KinematicDiagnostics:
    def __init__(self,sim):
        self.sim=sim;m=sim.model
        self.qadr=np.array([m.jnt_qposadr[m.joint(n).id] for n in sim.episode.names])
        self.tool=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3 and m.body(int(m.geom_bodyid[i])).name in ['pipette','ejector']],dtype=int)
        self.rack=np.array([i for i in range(m.ngeom) if m.geom_group[i]==3 and m.body(int(m.geom_bodyid[i])).name=='rack'],dtype=int)
        if not len(self.tool) or not len(self.rack):raise ValueError('Missing diagnostic collision geometry')
        self.rows=[]
    def sample(self,frame):
        s=self.sim;m=s.model;d=s.data
        expected=s.episode.sample(s.time,'state')[0]
        error=float(np.max(abs(d.qpos[self.qadr]-expected)))
        # Bounding spheres are only a conservative broad phase. Signed geometry
        # distances are queried even when normal collision masks are disabled.
        centers=d.geom_xpos[self.tool,None,:]-d.geom_xpos[None,self.rack,:]
        bounds=np.linalg.norm(centers,axis=2)-m.geom_rbound[self.tool,None]-m.geom_rbound[None,self.rack]
        possible=np.argwhere(bounds<.002)
        minimum=.002
        for ti,ri in possible:
            minimum=min(minimum,float(mujoco.mj_geomDistance(m,d,int(self.tool[ti]),int(self.rack[ri]),.002,None)))
        self.rows.append({'frame':frame,'simulation_time_s':s.time,'max_body_joint_error_rad':error,'minimum_tool_rack_distance_m_capped_at_2mm':minimum,'geometric_overlap':minimum<0.,'nozzle_world_m':d.site('nozzle').xpos.tolist(),'geometric_attachment_state':s.state})
    def report(self):
        hits=[r for r in self.rows if r['geometric_overlap']]
        return {'sample_rate_hz':30,'samples':len(self.rows),'overlap_frames':len(hits),'maximum_sampled_penetration_mm':max([0.]+[-r['minimum_tool_rack_distance_m_capped_at_2mm']*1000 for r in hits]),'first_overlap_time_s':hits[0]['simulation_time_s'] if hits else None,'max_body_joint_error_rad':max(r['max_body_joint_error_rad'] for r in self.rows),'method':'Signed distance between tool and rack collision geoms, independent of disabled collision masks, at rendered frames.','limitations':['30 Hz samples can miss contacts between frames.','Geometry is approximate; overlap is not force or real-world contact.','Attachment in kinematic mode is scripted/geometric, not physically validated.','Real-image pipette positions have not been independently annotated.'],'frames':self.rows}
