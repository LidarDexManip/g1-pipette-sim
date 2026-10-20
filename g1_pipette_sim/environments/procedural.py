"""Shared deterministic lab geometry; no renderer or dataset dependencies."""
import numpy as np

THEMES = ['Open steel shelving', 'Wire storage racks', 'Mobile lab carts',
          'Storage cabinets', 'Mixed wire and cabinet storage', 'Compact benchtop storage',
          'Tall utility shelving', 'Wide low storage']
# Fixed material palette: scene-to-scene color shifts come from illumination.
STEEL = [.58,.62,.64]
SHELF = [.72,.75,.76]
BOX = [.72,.65,.53]
LID = [.82,.8,.73]
LIGHTING = [
    {'name':'Neutral overhead','tint':[1.,.97,.92],'intensity':.82,'position':[.4,-.4,2.8]},
    {'name':'Cool diffuse','tint':[.87,.94,1.],'intensity':.85,'position':[.9,.4,2.6]},
    {'name':'Warm overhead','tint':[1.,.88,.73],'intensity':.79,'position':[-.4,-.6,2.5]},
    {'name':'Daylight from left','tint':[.92,.97,1.],'intensity':.93,'position':[-1.2,.3,2.]},
    {'name':'Soft neutral','tint':[1.,.96,.9],'intensity':.7,'position':[.2,.9,2.8]},
    {'name':'Warm task lighting','tint':[1.,.9,.78],'intensity':.88,'position':[.8,-.8,2.]},
    {'name':'Cool overhead','tint':[.88,.95,1.],'intensity':.75,'position':[0.,0.,3.]},
    {'name':'Bright neutral','tint':[1.,.98,.94],'intensity':.95,'position':[1.2,.3,2.7]},
]


def lab_layout(index, seed, cfg):
    """Renderer primitives for room furniture and task-adjacent benchtop equipment."""
    name=THEMES[index]
    wall=[.84,.85,.84];floor=[.55,.57,.57];accent=[.44,.49,.5]
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    objects = []
    def box(pos, size, color):
        objects.append(dict(kind='box', pos=list(pos), size=list(size), rgba=list(color)+[1]))
    def cylinder(pos, radius, halfheight, color):
        objects.append(dict(kind='cylinder', pos=list(pos), size=[radius,halfheight,0], rgba=list(color)+[1]))
    box([.4,0,-.02],[2.6,2.5,.025],floor)
    # Back wall and a side wall leave the foreground camera unobstructed.
    box([1.95,0,1.15],[.045,2.5,1.15],wall)
    box([.3,-1.55,1.15],[1.65,.045,1.15],wall)
    # Panel seams and an accent strip give scale to the room.
    box([1.897,0,.92],[.006,2.45,.055],accent)
    for y in [-1.1,-.2,.7,1.6]:
        box([1.895,y,1.45],[.007,.009,.8],[.65,.69,.7])
    # Move and resize the rear cabinetry between environments.
    y0 = float(rng.uniform(-.45,.1))
    for j in range(3):
        y = y0+(j-1)*.56
        box([1.55,y,.38],[.24,.26,.38],accent)
        box([1.53,y,.78],[.29,.28,.025],[.82,.85,.84])
        for z in [.2,.44,.64]:
            box([1.301,y,z],[.01,.235,.075],[min(1,c+.12) for c in accent])
            box([1.285,y,z+.015],[.012,.08,.012],[.75,.78,.8])
        # Vary equipment between bottles, a benchtop instrument and tube racks.
        equipment=(j+index)%3
        if equipment==0:
            for k in range(4):
                cylinder([1.5,y-.17+k*.11,.9],.035,.09,[.5,.7,.78])
                cylinder([1.5,y-.17+k*.11,1.0],.022,.015,accent)
        elif equipment==1:
            box([1.51,y,.94],[.16,.2,.13],[.91,.92,.9])
            box([1.342,y,.98],[.008,.13,.055],[.05,.19,.22])
            cylinder([1.51,y,1.085],.11,.015,[.36,.4,.45])
        else:
            box([1.49,y,.84],[.12,.19,.025],accent)
            for k in range(5):
                cylinder([1.48,y-.14+k*.07,.9],.017,.045,[.94,.76,.22])
    # Side shelving alternates with a large instrument enclosure.
    if index%2==0:
        for x in [.65,1.25]:
            box([x,-1.18,.88],[.025,.025,.88],[.48,.53,.56])
        for z in [.25,.7,1.15,1.6]:
            box([.95,-1.18,z],[.34,.2,.025],[.78,.81,.82])
            for k in range(3):
                box([.72+.21*k,-1.18,z+.11],[.07,.12,.085],BOX)
    else:
        box([.9,-1.19,.72],[.34,.25,.72],[.86,.88,.9])
        box([.9,-.93,1.0],[.28,.012,.3],[.19,.32,.37])
        box([.9,-.91,.61],[.28,.025,.025],accent)
    # Wall cabinet / fume hood and a differently located utility cart.
    box([1.69,.8,1.55],[.2,.38,.3],[.87,.9,.91])
    box([1.48,.8,1.53],[.012,.32,.22],[.3,.47,.51])
    cart_y = .9+float(rng.uniform(0,.18))
    for z in [.2,.65]:
        box([.9,cart_y,z],[.23,.18,.018],[.68,.73,.76])
    for x in [.69,1.11]:
        for y in [cart_y-.16,cart_y+.16]:
            cylinder([x,y,.34],.015,.32,[.5,.56,.6])
    box([.9,cart_y,.76],[.14,.12,.09],accent)
    # Camera is at +x/-y; put furniture behind the task, at -x/+y.
    for obj in objects:
        obj['pos'][0] = .4-obj['pos'][0]
        obj['pos'][1] = -obj['pos'][1]
    room_count=len(objects)
    top=cfg['table']['top_z']
    tx,ty=cfg['table']['center_xy']
    # Three separated slots along the far side of the 60 x 65 cm task table.
    # Keep the original task rack, open lid and insertion corridor clear.
    slots=[(tx+.15,ty-.19),(tx+.15,ty+.18),(tx+.16,ty-.005)]
    equipment=[]
    for slot,(x,y) in enumerate(slots):
        x+=float(rng.uniform(-.009,.009));y+=float(rng.uniform(-.01,.01))
        kind='centrifuge' if slot==index%2 else 'tube_rack'
        color=[.34,.45,.49]
        start=len(objects)
        if kind=='centrifuge':
            box([x,y,top+.035],[.065,.06,.035],[.9,.92,.94])
            cylinder([x,y,top+.073],.054,.009,color)
            cylinder([x,y,top+.085],.043,.003,[.18,.26,.3])
            # Visible lid ring and front display/controls.
            cylinder([x,y,top+.09],.012,.003,[.68,.73,.77])
            box([x-.0655,y,top+.047],[.001,.028,.011],[.04,.19,.24])
            box([x-.067,y+.042,top+.045],[.002,.006,.006],[.3,.8,.53])
        else:
            box([x,y,top+.015],[.055,.052,.015],color)
            box([x,y,top+.035],[.055,.052,.004],color)
            for row in range(3):
                for col in range(4):
                    xx=x+(col-1.5)*.024;yy=y+(row-1)*.032
                    cylinder([xx,yy,top+.041],.007,.018,[.76,.87,.9])
                    cylinder([xx,yy,top+.061],.008,.004,[.68,.71,.69])
        for obj in objects[start:]:obj['benchtop']=True
        equipment.append({'kind':kind,'center_xy':[x,y],'top_z':top,'first_geom':start,'end_geom':len(objects)})
    storage_start=len(objects)
    # Low storage shelves immediately beyond the table edges fall inside the
    # downward-looking policy views; the distant room shelves remain in place.
    storage=[]
    style_pairs=[('open','open'),('wire','wire'),('cart','cart'),('cabinet','cabinet'),
                 ('wire','cabinet'),('cart','open'),('open','wire'),('cabinet','cart')]
    for shelf,style in enumerate(style_pairs[index]):
        tall=[.85,1.03,.66,.94,1.08,.7,1.16,.62][index]
        levels=2 if style=='cart' or index in [5,7] else (4 if index==6 else 3)
        depth=[.10,.13,.11,.15,.12,.09,.11,.15][index]
        span=[.25,.31,.22,.26,.28,.19,.23,.34][index]
        hx,hy=(depth,span) if shelf==0 else (span,depth)
        yaw=np.deg2rad([-7,5,0,-4,7,0,-6,4][index])
        R=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        extent=np.abs(R[:2,:2])@np.array([hx,hy])
        if shelf==0:x=tx+.3+extent[0]+.045+.025*(index%3);y=ty+[-.05,.07,-.12,.04,-.08,.13,.0,-.04][index]
        else:x=tx+[-.09,.1,-.02,.07,-.06,.04,.13,-.03][index];y=ty-.325-extent[1]-.055-.025*(index%3)
        first=len(objects)
        for xx in [-hx+.012,hx-.012]:
            for yy in [-hy+.012,hy-.012]:
                box([xx,yy,tall/2],[.009,.009,tall/2],STEEL)
                if style=='cart':cylinder([xx,yy,.035],.025,.026,[.17,.18,.19])
        if style=='cabinet':
            # Closed side/back panels, open front cubbies with boxed supplies.
            if shelf==0:
                box([hx,0,tall/2],[.01,hy,tall/2],SHELF)
                for yy in [-hy,hy]:box([0,yy,tall/2],[hx,.01,tall/2],SHELF)
            else:
                box([0,-hy,tall/2],[hx,.01,tall/2],SHELF)
                for xx in [-hx,hx]:box([xx,0,tall/2],[.01,hy,tall/2],SHELF)
        for level,z in enumerate(np.linspace(.15,tall-.06,levels)):
            if style=='wire':
                for yy in np.linspace(-hy,hy,max(3,int(hy/.025))):box([0,yy,z],[hx,.003,.003],STEEL)
                for xx in np.linspace(-hx,hx,max(3,int(hx/.025))):box([xx,0,z],[.003,hy,.003],STEEL)
            else:box([0,0,z],[hx,hy,.01],SHELF)
            count=2 if index in [2,5] else 3
            for k in range(count):
                offset=(k-(count-1)/2)*(2*span/count)
                bx,by=(0,offset) if shelf==0 else (offset,0)
                half=[depth*.72,span/count*.78,.033+.007*((k+level)%3)] if shelf==0 else [span/count*.78,depth*.72,.033+.007*((k+level)%3)]
                box([bx,by,z+.012+half[2]],half,BOX)
                box([bx,by,z+.018+2*half[2]],[half[0]+.002,half[1]+.002,.005],LID)
                box([bx-half[0]-.001,by,z+.04],[.001,.023,.011],[.94,.93,.87])
        for obj in objects[first:]:
            obj['pos']=(R@np.asarray(obj['pos'])+[x,y,0]).tolist()
            obj['mat']=R.ravel().tolist()
        storage.append({'kind':style,'center_xy':[x,y],'size_xyz':[2*hx,2*hy,tall],'yaw_deg':float(np.rad2deg(yaw)),'levels':levels})
    # Small labeled consumables boxes on the near table corners.
    for x,y in [(tx-.19,ty-.20),(tx-.18,ty+.22)]:
        z=top
        for level in range(1+index%2):
            box([x,y,z+.019],[.045,.035,.019],list(BOX))
            box([x,y,z+.041],[.047,.037,.003],accent)
            box([x,y-.0355,z+.02],[.025,.001,.01],[.96,.96,.92])
            z+=.044
        storage.append({'kind':'consumables_boxes','center_xy':[x,y]})
    for obj in objects[storage_start:]:obj['policy_storage']=True
    return {'name':name,'objects':objects,'benchtop_equipment':equipment,'room_geom_count':room_count,'policy_storage':storage,'storage_geom_start':storage_start,'lighting':LIGHTING[index],'material_colors_fixed':True}
