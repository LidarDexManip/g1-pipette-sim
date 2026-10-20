"""Audit LIDAR geometry and measured replays; never modify scene calibration."""
import argparse
import copy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from g1_pipette_sim.dataset import Episode
from g1_pipette_sim.environments import ROOT, environment_manifest
from g1_pipette_sim.environments.robot import build_robot_scene, MeasuredReplay


def bounds(model, data, geom):
    rotation = data.geom_xmat[geom].reshape(3, 3)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        points = model.mesh_vert[start:start+count] @ rotation.T + data.geom_xpos[geom]
        return np.array([points.min(axis=0), points.max(axis=0)])
    center = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
    half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
    return np.array([center - half, center + half])


def update(records, key, frame, time, depth):
    row = records.setdefault(key, dict(frames=0, first_frame=frame, first_time_s=float(time),
                                      max_penetration_mm=0., worst_frame=frame))
    row['frames'] += 1
    if depth * 1000 > row['max_penetration_mm']:
        row.update(max_penetration_mm=float(depth * 1000), worst_frame=frame)


def geometry_audit(model, data):
    records = []
    objects = environment_manifest('lidar')['objects']
    groups = model.geom_group.copy()
    try:
        for obj in objects:
            body = model.body(obj['body']).id
            ids = np.where(model.geom_bodyid == body)[0]
            visual = ids[groups[ids] == 1]
            collision = ids[model.geom_contype[ids] != 0]
            def union(indices):
                boxes = np.array([bounds(model, data, int(g)) for g in indices])
                return np.array([boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)])
            vb, cb = union(visual), union(collision)
            # Isolate one object's visual/collision surfaces for identical downward rays.
            model.geom_group[:] = 5
            model.geom_group[visual] = 1
            model.geom_group[collision] = 3
            grid = []; recessed = []
            for x in np.linspace(vb[0, 0], vb[1, 0], 13)[1:-1]:
                for y in np.linspace(vb[0, 1], vb[1, 1], 13)[1:-1]:
                    # Avoid exact shared triangle edges at symmetric grid points.
                    origin = np.array([x+1.234e-6, y+2.345e-6, max(vb[1, 2], cb[1, 2]) + .05])
                    distances = []
                    for group in (1, 3):
                        mask = np.zeros(6, dtype=np.uint8); mask[group] = 1
                        distances.append(float(mujoco.mj_ray(model, data, origin, np.array([0., 0., -1.]),
                                                            mask, 1, -1, None)))
                    v, c = distances
                    if (v < 0 <= c) or (v >= 0 and c >= 0 and v - c > .001):
                        grid.append(dict(x=float(x), y=float(y), visual_distance_m=v,
                                         collision_distance_m=c,
                                         excess_height_mm=(v-c)*1000 if v >= 0 else None))
                    if v >= 0 and (c < 0 or c-v > .001):
                        recessed.append(dict(x=float(x),y=float(y),
                                             missing_collision=c<0,
                                             recess_mm=(c-v)*1000 if c>=0 else None))
            records.append(dict(body=obj['body'], visual_geoms=len(visual), collision_geoms=len(collision),
                                bounds_difference_mm=((cb-vb)*1000).tolist(),
                                downward_rays=121, inflated_or_filled_rays=len(grid), examples=grid[:8],
                                missing_or_recessed_rays=len(recessed), recessed_examples=recessed[:8],
                                max_recess_mm=max((r['recess_mm'] or 0 for r in recessed),default=0),
                                max_excess_height_mm=max((r['excess_height_mm'] or 0 for r in grid), default=0)))
            print('Geometry:', obj['body'], 'inflated rays', len(grid), flush=True)
    finally:
        model.geom_group[:] = groups
    # Static scene bodies are welded to world: direct distance queries bypass
    # the normal welded-body contact filter.
    colliders = np.where((model.geom_contype == 1) &
                         (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE))[0]
    boxes = {int(g): bounds(model, data, int(g)) for g in colliders}
    box_array=np.array([boxes[int(g)] for g in colliders])
    pairs = {}
    for k, ga in enumerate(colliders):
        a=boxes[int(ga)]
        candidates=(np.all(np.minimum(a[1],box_array[k+1:,1])-
                           np.maximum(a[0],box_array[k+1:,0])>.001,axis=1) &
                    (model.geom_bodyid[colliders[k+1:]]!=model.geom_bodyid[ga]))
        for gb in colliders[k+1:][candidates]:
            ba, bb = model.geom_bodyid[ga], model.geom_bodyid[gb]
            distance = mujoco.mj_geomDistance(model, data, int(ga), int(gb), .1, None)
            if distance < -.001:
                key = ' / '.join(sorted([model.body(ba).name, model.body(bb).name]))
                if -distance * 1000 > pairs.get(key, {}).get('max_penetration_mm', 0):
                    pairs[key] = dict(max_penetration_mm=float(-distance*1000),
                                      geoms=[model.geom(int(ga)).name, model.geom(int(gb)).name])
    return records, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--episodes', type=int, nargs='+', default=[0, 126, 188, 127, 1])
    p.add_argument('--output', type=Path, default=Path('outputs/lidar-teleop/collision-audit'))
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    xml = build_robot_scene()
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    geometry, static_pairs = geometry_audit(model, data)
    # Separate diagnostic model: enable self contacts, retaining the robot asset's
    # intentional exclusions. Production masks and geometry are untouched.
    root = ET.fromstring(xml)
    exclusions = ET.parse(ROOT/'assets/robot/g1_inspire.xml').getroot().find('contact')
    if exclusions is not None and root.find('contact') is None: root.append(copy.deepcopy(exclusions))
    self_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    self_model.geom_conaffinity[self_model.geom_contype == 2] = 3
    self_data = mujoco.MjData(self_model)
    mesh_alignment = []
    for geom in np.where(model.geom_contype == 2)[0]:
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH: continue
        candidates = np.where((model.geom_bodyid == model.geom_bodyid[geom]) & (model.geom_group == 1) &
                              (model.geom_dataid == model.geom_dataid[geom]))[0]
        matched = any(np.allclose(model.geom_pos[g],model.geom_pos[geom],atol=1e-9) and
                      np.allclose(model.geom_quat[g],model.geom_quat[geom],atol=1e-9) for g in candidates)
        mesh_alignment.append(dict(body=model.body(model.geom_bodyid[geom]).name,
                                   mesh=model.mesh(model.geom_dataid[geom]).name,
                                   exact_visual_mesh_and_transform_match=matched))
    report = dict(threshold_mm=1., geometry=geometry, static_object_overlaps=static_pairs,
                  robot_collision_mesh_alignment=mesh_alignment,
                  production_robot_self_collision_enabled=bool(np.any(
                      model.geom_conaffinity[model.geom_contype==2] & 2)),
                  production_robot_contact_exclusions=model.nexclude,
                  diagnostic_robot_contact_exclusions=self_model.nexclude,
                  static_props_have_world_weld=all(model.body_weldid[model.body(o['body']).id] == 0
                                                  for o in environment_manifest('lidar')['objects']),
                  limitations=['Prescribed qpos: detected contacts cannot stop motion.',
                               'Static object distances bypass welded-body filtering.',
                               'Bounds agreement does not establish concave-surface fidelity.',
                               'Ray tests sample top surfaces only, not every cavity.',
                               'Self-contact diagnostics retain the intended robot asset exclusions.',
                               'Depth is per convex piece: thinner decomposition pieces can reduce reported depth without removing overlap.',
                               'Penetration depths are collision-proxy estimates, not measured physical deformation.'],
                  episodes=[])
    for index in args.episodes:
        episode = Episode(args.dataset, index)
        replay = MeasuredReplay(model, data, episode)
        records = {}; self_records = {}; frame_rows = []; joint_violations = {}
        for frame, time in enumerate(episode.times):
            replay.pose(time)
            for joint in np.where(model.jnt_limited)[0]:
                q = data.qpos[model.jnt_qposadr[joint]]
                excess = max(model.jnt_range[joint,0]-q, q-model.jnt_range[joint,1], 0)
                if excess > 1e-5:
                    name = model.joint(int(joint)).name
                    joint_violations[name] = max(joint_violations.get(name,0.), float(excess))
            current = {}; any_scene_penetration=False
            for c in data.contact:
                types = model.geom_contype[[c.geom1, c.geom2]]
                if 2 not in types or np.all(types==2): continue
                if c.dist<-.00001:any_scene_penetration=True
                if c.dist >= -.001: continue
                names = [model.body(model.geom_bodyid[g]).name or 'world' for g in (c.geom1,c.geom2)]
                key = ' / '.join(sorted(names))
                current[key] = max(current.get(key, 0), float(-c.dist))
            for key, depth in current.items(): update(records, key, frame, time, depth)
            self_data.qpos[:] = data.qpos; mujoco.mj_forward(self_model, self_data)
            current_self = {}
            for c in self_data.contact:
                if c.dist >= -.001 or any(self_model.geom_contype[g] != 2 for g in (c.geom1,c.geom2)): continue
                key = ' / '.join(sorted(self_model.body(self_model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)))
                current_self[key] = max(current_self.get(key, 0), float(-c.dist))
            for key, depth in current_self.items(): update(self_records, key, frame, time, depth)
            if current or current_self or any_scene_penetration:
                frame_rows.append(dict(frame=frame, time_s=float(time), robot_scene_penetration_m=current,
                                       robot_scene_penetration_over_10um=any_scene_penetration,
                                       diagnostic_self_penetration_m=current_self))
        row = dict(episode=index, frames=len(episode.times),
                   scene_state=replay.scene_state,
                   joint_range_excess_rad=joint_violations,
                   robot_scene_penetration_frames=sum(bool(r['robot_scene_penetration_m']) for r in frame_rows),
                   robot_scene_penetration_frames_over_10um=sum(r['robot_scene_penetration_over_10um'] for r in frame_rows),
                   diagnostic_self_penetration_frames=sum(bool(r['diagnostic_self_penetration_m']) for r in frame_rows),
                   robot_scene_pairs=records, diagnostic_self_pairs=self_records)
        report['episodes'].append(row)
        (args.output/f'episode-{index:03d}-frames.json').write_text(json.dumps(frame_rows, indent=2)+'\n')
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if not k.endswith('_pairs')}), flush=True)


if __name__ == '__main__':
    main()
