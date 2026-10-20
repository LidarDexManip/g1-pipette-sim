"""Historical calibrated workspace geometry, shared by pipette task variants."""
import numpy as np

def add_workspace(root, config):
    from ..scene import element
    asset=root.find('asset');world=root.find('worldbody')
    element(asset,'texture',name='floor_tex',type='2d',builtin='checker',rgb1=[.11,.14,.17],rgb2=[.14,.17,.20],width=256,height=256)
    element(asset,'material',name='floor_mat',texture='floor_tex',texrepeat=[8,8],texuniform='true')
    element(world,'light',pos=[.4,-.5,2],dir=[0,0,-1],diffuse=[.8,.8,.8],directional='true')
    element(world,'geom',name='floor',type='plane',size=[3,3,.02],material='floor_mat',friction=[.8,.005,.0001])
    element(world,'geom',name='backdrop',type='box',pos=[2.5,0,1.2],size=[.02,3,1.2],rgba=[.035,.055,.075,1])
    t=config['table'];cx,cy=t['center_xy'];sx,sy=np.asarray(t['size_xy'])/2;top=t['top_z'];th=t['thickness']
    element(world,'geom',name='table',type='box',pos=[cx,cy,top-th/2],size=[sx,sy,th/2],rgba=[.57,.43,.25,1],friction=[.8,.005,.0001])
    for dx in [-sx+.04,sx-.04]:
        for dy in [-sy+.04,sy-.04]:
            element(world,'geom',type='box',pos=[cx+dx,cy+dy,(top-th)/2],size=[.018,.018,(top-th)/2],rgba=[.22,.24,.26,1])
