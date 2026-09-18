"""Photo-derived pipette components; dimensions in metres, faces zero-indexed."""
import numpy as np


def loft(rings):
    vertices=np.asarray(rings).reshape(-1,3).tolist();n=len(rings[0]);faces=[]
    for j in range(len(rings)-1):
        for i in range(n):
            a=j*n+i;b=j*n+(i+1)%n;faces.extend([[a,b,b+n],[a,b+n,a+n]])
    for i in range(1,n-1):
        faces.append([0,i+1,i]);start=(len(rings)-1)*n;faces.append([start,start+i,start+i+1])
    return vertices,faces


def beveled_profile(profile,depth_scale=.9):
    # Flat broad faces and narrow diagonal bevels, visible in the September 10 photos.
    ring=np.array([[1,.65],[.65,1],[-.65,1],[-1,.65],[-1,-.65],[-.65,-1],[.65,-1],[1,-.65]])
    return loft([np.c_[ring*[r,r*depth_scale],np.full(8,z)] for z,r in profile])


def parts(p):
    from scipy.interpolate import CubicSpline
    from .object_meshes import lathe
    result={'pipette_handle':beveled_profile(p['handle_profile'],p['handle_y_scale']),
            'pipette_lower':lathe(p['shaft_profile']),
            'pipette_nozzle':lathe(p.get('nozzle_profile',[[p['nozzle_z'],.001],[-.153,.0013],[-.1478,.0019]]))}
    # Separate convex axial frusta preserve taper changes without a full-shaft hull.
    for name,profile in [('pipette_lower',p['shaft_profile']),('pipette_nozzle',p['nozzle_profile'])]:
        for i in range(len(profile)-1):result[f'{name}_contact_{i}']=lathe(profile[i:i+2])
    tape=p.get('red_tape')
    if tape and tape.get('enabled',True):
        start=p['nozzle_z']+tape.get('start_offset_from_nozzle_m',.0127)
        end=start+tape.get('length_m',.020)
        shaft=np.asarray(p['shaft_profile'])
        levels=np.unique(np.r_[start,np.linspace(start,end,7),shaft[(shaft[:,0]>start)&(shaft[:,0]<end),0],end])
        # A color-bearing skin 20 microns above the shaft avoids z-fighting.
        # It is not a measured tape-thickness or additional collision shape.
        offset=tape.get('visual_offset_m',.00002)
        profile=[[z,float(np.interp(z,shaft[:,0],shaft[:,1]))+offset] for z in levels]
        result['pipette_red_tape']=lathe(profile,segments=96,caps=False)
    # A broad moulded ribbon; preserve its concavity with separate convex sectors.
    controls=np.array([[-.007,.072],[-.018,.076],[-.029,.082],[-.0355,.081],[-.0375,.072]])
    t=np.linspace(0,1,len(controls));curve=CubicSpline(t,controls,axis=0)(np.linspace(0,1,21))
    # Bound the outer edge by the measured 63.5 mm overall width (opposite tab ends at +24 mm).
    curve[:,0]=-.007+(curve[:,0]+.007)*(.0305/abs(curve[:,0].min()+.007))
    tangent=np.gradient(curve,axis=0)
    normal=np.c_[tangent[:,1],-tangent[:,0]]
    normal/=np.linalg.norm(normal,axis=1)[:,None]
    rings=[]
    for center,n in zip(curve,normal):
        low=center-.002*n;high=center+.002*n
        rings.append([[-.009,*low],[.009,*low],[.009,*high],[-.009,*high]])
    rings=np.asarray(rings)
    factor=.0325/abs(rings[:,:,1].min()+.007)
    rings[:,:,1]=-.007+(rings[:,:,1]+.007)*factor
    rings=rings.tolist()
    result['pipette_finger_rest']=loft(rings)
    for i in range(len(rings)-1):result[f'pipette_rest_contact_{i}']=loft(rings[i:i+2])
    # Blue side paddle, in ejector-local coordinates.
    result['pipette_ejector']=beveled_profile([[-.014,.0065],[-.012,.007],[.004,.007]],.8)
    # Separate broad thumb pad over a narrower sliding stem, seen in the new head close-ups.
    result['pipette_ejector_pad']=beveled_profile([[.004,.009],[.005,.011],[.008,.011],[.009,.009]],.6363636364)
    top=p['nozzle_z']+p['measured_dimensions_m']['length']
    vertices,faces=lathe([[top-.009,.0088],[top-.008,.0095],[top-.001,.0095],[top,.0088]],segments=64)
    vertices=np.asarray(vertices)
    # Sixteen shallow moulded flutes, retaining the measured outer envelope.
    for i in range(256):
        factor=1-.045*(1-np.cos(16*np.arctan2(vertices[i,1],vertices[i,0])))/2
        vertices[i,:2]*=factor
    result['pipette_plunger_cap']=(vertices.tolist(),faces)
    return result


def add_details(pipe,p,element):
    white=[.88,.91,.93,1];blue=[.18,.34,.55,1]
    def component(parent,name,color,mass):
        element(parent,'geom',type='mesh',mesh='object_'+name,group=2,mass=0,rgba=color,contype=0,conaffinity=0)
        key={'pipette_lower':'shaft_profile','pipette_nozzle':'nozzle_profile'}.get(name)
        if key:
            profile=np.asarray(p[key]);volumes=[(b[0]-a[0])*(a[1]**2+a[1]*b[1]+b[1]**2) for a,b in zip(profile[:-1],profile[1:])]
            for i,volume in enumerate(volumes):
                element(parent,'geom',name=name if i==0 else f'{name}_{i}',type='mesh',mesh=f'object_{name}_contact_{i}',group=3,mass=mass*volume/sum(volumes),contype=0,conaffinity=0)
        else:
            element(parent,'geom',name=name,type='mesh',mesh='object_'+name,group=3,mass=mass,contype=0,conaffinity=0)
    component(pipe,'pipette_handle',white,.060);component(pipe,'pipette_lower',blue,.018)
    component(pipe,'pipette_nozzle',[.035,.04,.045,1],.001)
    tape=p.get('red_tape')
    if tape and tape.get('enabled',True):
        element(pipe,'geom',name='pipette_red_tape_visual',type='mesh',mesh='object_pipette_red_tape',group=2,mass=0,rgba=tape.get('rgba',[.78,.045,.035,1]),contype=0,conaffinity=0)
    element(pipe,'geom',type='mesh',mesh='object_pipette_finger_rest',group=2,mass=0,rgba=white,contype=0,conaffinity=0)
    for i in range(20):element(pipe,'geom',name=f'finger_rest_{i}',type='mesh',mesh=f'object_pipette_rest_contact_{i}',group=3,mass=.0015/20,contype=0,conaffinity=0)
    element(pipe,'geom',type='cylinder',pos=[0,0,.078],size=[.0045,.013],rgba=white,mass=.002,contype=0,conaffinity=0)
    # Thin concentric collar around the shaft; photo shows a metal ring at the head, not the nozzle.
    element(pipe,'geom',type='cylinder',pos=[0,0,.074],size=[.006,.0012],rgba=[.65,.68,.70,1],mass=.0005,contype=0,conaffinity=0)
    top=p['nozzle_z']+p['measured_dimensions_m']['length']
    component(pipe,'pipette_plunger_cap',blue,.001)
    # Narrow volume window and illustrative three-digit display on the flat front.
    y=-max(r for z,r in p['handle_profile'])*p['handle_y_scale']
    element(pipe,'geom',type='box',pos=[0,y-.00025,-.020],size=[.0028,.00025,.012],rgba=[.025,.03,.035,1],mass=0,contype=0,conaffinity=0)
    for z in [-.027,-.020,-.013]:
        for dz in [-.002,.002]:element(pipe,'geom',type='box',pos=[0,y-.00055,z+dz],size=[.0012,.0001,.00025],rgba=[.95,.95,.95,1],mass=0,contype=0,conaffinity=0)
        for x in [-.0012,.0012]:element(pipe,'geom',type='box',pos=[x,y-.00055,z],size=[.00025,.0001,.002],rgba=[.95,.95,.95,1],mass=0,contype=0,conaffinity=0)
    button=element(pipe,'body',name='ejector',pos=[0,.017,.065])
    element(button,'joint',name='ejector_slide',type='slide',axis=[0,0,-1],range=[0,.012],damping=.1)
    component(button,'pipette_ejector',blue,.0035)
    component(button,'pipette_ejector_pad',blue,.0015)
