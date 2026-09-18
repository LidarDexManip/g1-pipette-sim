"""Initialize unknown finger positions at first tool contact, preserving targets.

This is an initial-pose estimate, not recovered hand feedback. It neither moves
nor constrains the pipette and never changes poses during physics stepping.
"""
import numpy as np
import mujoco


def initialize_fingers(model, data, tolerance=.00002):
    def penetration(prefix=None):
        result=0.
        for contact in data.contact:
            names=[model.body(int(model.geom_bodyid[g])).name for g in [contact.geom1,contact.geom2]]
            if 'pipette' in names and any(n.startswith(prefix or 'R_') for n in names):
                result=max(result,-float(contact.dist))
        return result
    report={'method':'open-to-first-contact initialization; recorded servo targets unchanged',
            'before_max_penetration_mm':penetration()*1000,'fingers':{}}
    for finger in ['index','middle','ring','pinky','thumb']:
        prefix='R_'+finger
        ids=[j for j in range(model.njnt) if model.joint(j).name.startswith(prefix) and 'yaw' not in model.joint(j).name]
        q=model.jnt_qposadr[ids];closed=data.qpos[q].copy()
        def set_fraction(fraction):
            data.qpos[q]=closed*fraction
            mujoco.mj_forward(model,data)
            return penetration(prefix)
        if set_fraction(1)<=tolerance:
            continue
        if set_fraction(0)>tolerance:
            set_fraction(1)
            report['fingers'][finger]={'resolved':False,'reason':'Even open finger intersects tool; mounting pose needs calibration'}
            continue
        low=0.;high=1.
        for fraction in np.linspace(.02,1,50):
            if set_fraction(fraction)>tolerance:
                high=float(fraction);break
            low=float(fraction)
        for _ in range(12):
            middle=(low+high)/2
            if set_fraction(middle)>tolerance:high=middle
            else:low=middle
        set_fraction(low)
        report['fingers'][finger]={'resolved':True,'initial_closure_fraction':low}
    report['after_max_penetration_mm']=penetration()*1000
    return report
