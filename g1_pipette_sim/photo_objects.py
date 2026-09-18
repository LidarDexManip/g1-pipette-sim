"""Photo-guided workspace shapes; independent visual and contact geometry."""
import math
import numpy as np
from .scene import element, rack_target, vec
from .object_meshes import generate


def add_objects(config, world, pelvis, asset):
    meshes=generate(config)
    for name,path in meshes.items():element(asset,'mesh',name='object_'+name,file=str(path.resolve()))
    def visual(body,mesh,**kw):
        return element(body,'geom',type='mesh',mesh='object_'+mesh,group=2,mass=0,contype=0,conaffinity=0,**kw)
    clear=[.73,.84,.88,.32];yellow=[.94,.79,.06,1];blue=[.04,.39,.64,1]
    r=config['rack'];top=config['table']['top_z'];h=r['height']
    rx,ry=np.asarray(r.get('size_xy',[.128,.088]))/2
    rack=element(world,'body',name='rack',pos=[*r['center_xy'],top],euler=[0,0,math.radians(r['yaw_deg'])])
    element(rack,'geom',name='rack_bottom',type='box',pos=[0,0,.005],size=[rx,ry,.002],rgba=clear)
    for x in [-rx,rx]:element(rack,'geom',type='box',pos=[x,0,h/2+.002],size=[.0015,ry,h/2-.002],rgba=clear)
    for y in [-ry,ry]:element(rack,'geom',type='box',pos=[0,y,h/2+.002],size=[rx,.0015,h/2-.002],rgba=clear)
    for x in [-rx+.006,rx-.006]:
        for y in [-ry+.006,ry-.006]:element(rack,'geom',type='box',pos=[x,y,.002],size=[.005,.005,.002],rgba=clear)
    # External ribs have no extra collision surfaces or mass.
    for y in [-ry-.001,ry+.001]:
        for x in np.linspace(-rx+.01,rx-.01,9):element(rack,'geom',type='box',pos=[x,y,h*.48],size=[.001,.0015,h*.32],rgba=[.78,.87,.89,.6],mass=0,contype=0,conaffinity=0)
    visual(rack,'rack_plate',pos=[0,0,h],rgba=yellow)
    gx=r['cols']*r['pitch']/2;gy=r['rows']*r['pitch']/2
    for x in [-1,1]:element(rack,'geom',type='box',pos=[x*(rx+gx)/2,0,h],size=[(rx-gx)/2,ry,.001],rgba=yellow)
    for y in [-1,1]:element(rack,'geom',type='box',pos=[0,y*(ry+gy)/2,h],size=[gx,(ry-gy)/2,.001],rgba=yellow)
    for row in range(r['rows']):
        for col in range(r['cols']):
            pos=[(col-(r['cols']-1)/2)*r['pitch'],(row-(r['rows']-1)/2)*r['pitch'],h]
            visual(rack,'well_rim',pos=pos,rgba=yellow)
            for i in range(16):element(rack,'geom',type='mesh',mesh=f'object_well_sector_{i}',pos=pos,group=3,mass=.03/(r['rows']*r['cols']*16))
    # Lid uses its own free body: hinge articulation is not calibrated yet.
    w=config['waste'];wx,wy=np.asarray(w['size_xy'])/2
    waste=element(world,'body',name='waste',pos=[*(np.array(r['center_xy'])+w['offset_from_rack']),top],euler=[0,0,math.radians(w.get('yaw_deg',0))])
    element(waste,'geom',name='waste_bottom',type='box',pos=[0,0,.002],size=[wx,wy,.002],rgba=clear,friction=[.9,.01,.0001])
    for x in [-wx,wx]:element(waste,'geom',type='box',pos=[x,0,w['height']/2],size=[.0015,wy,w['height']/2],rgba=clear)
    for y in [-wy,wy]:element(waste,'geom',type='box',pos=[0,y,w['height']/2],size=[wx,.0015,w['height']/2],rgba=clear)
    for x in ([] if config['waste'].get('hinge',{}).get('enabled') else [-wx*.65,wx*.65]):element(waste,'geom',type='box',pos=[x,wy+.002,.008],size=[.009,.003,.004],rgba=clear)
    element(waste,'geom',type='box',pos=[0,-wy-.003,.012],size=[.012,.003,.006],rgba=clear)
    from .box_lid import enabled, attach
    if enabled(config):attach(config, world, rack, waste, element)
    p=config['pipette'];wrist=pelvis.find(".//body[@name='right_wrist_yaw_link']")
    pipe=element(wrist,'body',name='pipette',pos=p['mount_position'],quat=p['mount_quaternion'])
    if p.get('photo_revision'):
        from .pipette_mesh import add_details
        add_details(pipe,p,element)
    else:
        for name,color,mass in [('pipette_handle',[.91,.93,.92,1],.060),('pipette_lower',blue,.02)]:
            visual(pipe,name,rgba=color)
            element(pipe,'geom',name=name,type='mesh',mesh='object_'+name,group=3,mass=mass,contype=0,conaffinity=0)
        element(pipe,'geom',type='cylinder',pos=[0,0,.070 if p.get('product_model')=='onilab_0.5_10ul' else .064],size=[.0035,.014] if p.get('product_model')=='onilab_0.5_10ul' else [.006,.007],rgba=[.7,.75,.75,1],contype=0,conaffinity=0)
        element(pipe,'geom',type='cylinder',pos=[0,0,.087 if p.get('product_model')=='onilab_0.5_10ul' else .080],size=[.0095,.003] if p.get('product_model')=='onilab_0.5_10ul' else [.013,.010],rgba=blue,contype=0,conaffinity=0)
        element(pipe,'geom',type='cylinder',pos=[0,0,p['nozzle_z']/p.get('detail_length_scale',1)+.018],size=[.003,.006],rgba=[.60,.62,.64,1] if p.get('product_model')=='onilab_0.5_10ul' else [.50,.16,.16,1],mass=0,contype=0,conaffinity=0)
        element(pipe,'geom',type='cylinder',pos=[0,0,p['nozzle_z']/p.get('detail_length_scale',1)+.003],size=[.0015,.003],rgba=[.08,.09,.10,1],mass=0,contype=0,conaffinity=0)
        if p.get('product_model')!='onilab_0.5_10ul':element(pipe,'geom',type='box',pos=[0,-.0113,.015],size=[.005,.001,.016],rgba=[.11,.17,.20,1],mass=0,contype=0,conaffinity=0)
        if p.get('product_model')=='onilab_0.5_10ul':
            # Curved finger rest, plunger collar and metal tip-ejector sleeve.
            for start,end in [([0,-.006,.052],[0,-.018,.065]),([0,-.018,.065],[0,-.030,.062]),([0,-.030,.062],[0,p.get('hand_rest_outer_y',-.039)+.003,.052])]:
                element(pipe,'geom',name='finger_rest_'+str(start[1]),type='capsule',fromto=[*start,*end],size=[.003],rgba=[.91,.93,.92,1],mass=.0005,contype=0,conaffinity=0)
            element(pipe,'geom',type='cylinder',pos=[0,0,-.137],size=[.0028,.004],rgba=[.65,.67,.69,1],mass=.001,contype=0,conaffinity=0)
            # Recessed three-digit volume window on the front face.
            element(pipe,'geom',type='box',pos=[0,-.0104*p.get('display_face_scale',1),-.014],size=[.003,.0005,.012],rgba=[.035,.04,.045,1],mass=0,contype=0,conaffinity=0)
            for z in [-.021,-.014,-.007]:
                for dz in [-.002, .002]:
                    element(pipe,'geom',type='box',pos=[0,-.011*p.get('display_face_scale',1),z+dz],size=[.0013,.0001,.00025],rgba=[.95,.95,.95,1],mass=0,contype=0,conaffinity=0)
                for x in [-.0013,.0013]:
                    element(pipe,'geom',type='box',pos=[x,-.011*p.get('display_face_scale',1),z],size=[.00025,.0001,.002],rgba=[.95,.95,.95,1],mass=0,contype=0,conaffinity=0)
        button=element(pipe,'body',name='ejector',pos=[0,.017,.060]);element(button,'joint',name='ejector_slide',type='slide',axis=[0,0,-1],range=[0,.012],damping=.1)
        element(button,'geom',type='cylinder',size=[.007,.005],mass=.005,rgba=blue,contype=0,conaffinity=0)
        # Scale non-mesh detail lengths consistently with the measured main profiles.
        scale=p.get('detail_length_scale',1.)
        for node in pipe.iter():
            if node is pipe or node.tag not in ['geom','body']:continue
            if node.get('type')=='mesh':continue
            if node.get('pos'):
                pos=list(map(float,node.get('pos').split()));pos[2]*=scale;node.set('pos',vec(pos))
            if node.get('fromto'):
                points=list(map(float,node.get('fromto').split()));points[2]*=scale;points[5]*=scale;node.set('fromto',vec(points))
            if node.tag=='geom' and node.get('type') in ['box','cylinder']:
                size=list(map(float,node.get('size').split()));size[-1]*=scale;node.set('size',vec(size))
    element(pipe,'site',name='nozzle',pos=[0,0,p['nozzle_z']],size=.002,rgba=[0,0,0,0])
    length=p['tip_length'];tip=element(world,'body',name='tip',pos=rack_target(config)-[0,0,length/2]);element(tip,'freejoint',name='tip_free')
    visual(tip,'tip_visual',rgba=[.29,.70,.08,1])
    from .object_meshes import tip_shell_parts
    pieces=tip_shell_parts(p);total_volume=sum(v for _,_,v in pieces)
    for i,(_,_,volume) in enumerate(pieces):
        element(tip,'geom',name='tip_geom' if i==0 else f'tip_geom_{i}',type='mesh',mesh=f'object_tip_collision_{i}',group=3,mass=.0006*volume/total_volume,friction=[.7,.005,.0001],solref=[.003,1],solimp=[.99,.999,.0001])
