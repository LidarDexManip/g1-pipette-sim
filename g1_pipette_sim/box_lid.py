"""Photo-guided rear hinge; dimensions and hinge friction remain estimates."""
import math
import numpy as np
from scipy.spatial.transform import Rotation


def enabled(config):
    return config.get('waste', {}).get('hinge', {}).get('enabled', False)


def geometry(config):
    h = config['waste']['hinge']
    axis = np.asarray(h.get('position', [0., -config['rack']['size_xy'][1] / 2 - .010, config['rack']['height']]), dtype=float)
    closed_center = np.array([0., -axis[1], config['waste']['height']])
    return axis, closed_center


def pose(config, angle_deg=None):
    """Initial lid bowl frame in world coordinates, before dynamic settling."""
    axis, center = geometry(config)
    angle = config['waste']['hinge']['initial_angle_deg'] if angle_deg is None else angle_deg
    yaw = Rotation.from_euler('z', config['rack']['yaw_deg'], degrees=True).as_matrix()
    opening = Rotation.from_euler('x', angle, degrees=True).as_matrix()
    origin = np.r_[config['rack']['center_xy'], config['table']['top_z']]
    return origin + yaw @ (axis + opening @ center), yaw @ opening @ Rotation.from_euler('x', 180, degrees=True).as_matrix()


def attach(config, world, rack, waste, element):
    axis, center = geometry(config)
    h = config['waste']['hinge'];angle = math.radians(h['initial_angle_deg'])
    carrier = element(rack, 'body', name='lid_hinge_carrier', pos=axis, euler=[angle, 0, 0])
    element(carrier, 'joint', name='lid_hinge', type='hinge', axis=[1, 0, 0],
            range=np.deg2rad(h.get('range_deg', [0, 230])).tolist(), ref=angle,
            damping=h.get('damping', .003), frictionloss=h.get('frictionloss', .001), armature=.00001)
    world.remove(waste);carrier.append(waste)
    waste.set('pos', ' '.join(map(str, center)));waste.set('euler', f'{math.pi} 0 0')
    # Two molded hinge stations. Pin/knuckle mating surfaces are visual only;
    # the revolute joint supplies the actual connection without a wrist-like weld.
    for x in [-config['rack']['size_xy'][0] * .33, config['rack']['size_xy'][0] * .33]:
        element(rack, 'geom', type='capsule', fromto=[x-.009, axis[1], axis[2], x+.009, axis[1], axis[2]],
                size=[.002], rgba=[.75,.85,.88,.65], mass=0, contype=0, conaffinity=0)
        element(carrier, 'geom', type='cylinder', pos=[x,0,0], euler=[0,math.pi/2,0], size=[.0025,.004],
                rgba=[.8,.88,.9,.6], mass=0, contype=0, conaffinity=0)
    # Explicit shell pairs preserve rack/lid collision despite MuJoCo's default
    # filtering of joints between parent and child bodies. No well meshes need
    # pairing: the oversized shell clears the plate over the allowed arc.
    for prefix, body in [('rack_shell', rack), ('lid_shell', waste)]:
        for i, geom in enumerate(body.findall('geom')):
            if geom.get('type') == 'box' and geom.get('mass') != '0':
                if not geom.get('name'):geom.set('name', f'{prefix}_{i}')


def contact_pairs(root, contact, element):
    rack = root.find(".//body[@name='rack']");lid = root.find(".//body[@name='waste']")
    for a in rack.findall('geom'):
        if a.get('type') != 'box' or a.get('mass') == '0':continue
        for b in lid.findall('geom'):
            if b.get('type') != 'box' or b.get('mass') == '0':continue
            element(contact, 'pair', geom1=a.get('name'), geom2=b.get('name'), condim=3,
                    friction=[.7,.7,.005,.0001,.0001], solref=[.008,1])
