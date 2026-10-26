import argparse, json, os, time, threading
from pathlib import Path
from queue import SimpleQueue


def configure_viewer_camera(camera, model, data, config, name):
    """Overview is a mouse-controlled free camera; sensor views stay fixed."""
    import mujoco
    import numpy as np
    camera_id = model.camera(name).id
    if name != 'overview':
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        return
    position = data.cam_xpos[camera_id].copy()
    forward = -data.cam_xmat[camera_id].reshape(3, 3)[:, 2]
    distance = max(float(np.linalg.norm(position - np.asarray(config['cameras']['overview']['target']))), .1)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    camera.trackbodyid = -1
    camera.distance = distance
    camera.lookat[:] = position + distance * forward
    camera.azimuth = float(np.degrees(np.arctan2(forward[1], forward[0])))
    camera.elevation = float(np.degrees(np.arcsin(np.clip(forward[2], -1, 1))))


def main():
    parser=argparse.ArgumentParser(description='G1/Inspire workspace with gravity, collisions and servo target replay.')
    from .environments import ENVIRONMENTS
    parser.add_argument('--environment',choices=list(ENVIRONMENTS),default=None,help='Workspace/environment selection. lidar uses fixed-base dynamic servo replay with movable pipette/tube; --physics kinematic restores measured-position diagnostics. lab-* adds visual surroundings to the pipette task.')
    parser.add_argument('--list-environments',action='store_true',help='List environments and exit without loading a dataset.')
    parser.add_argument('--hand',choices=['inspire','wuji'],default='inspire',help='Robot hands. wuji uses the G1 + Wuji Hand 2 model from g1-wuji-retargeting (prepare with tools/prepare_wuji_assets.py); recorded Inspire registers drive it through an approximate mapping. Currently --environment lidar only.')
    parser.add_argument('--dataset',type=Path,default=None,help='Dataset root or HF cache (default: cached HIL194, or HIL293 with its matching replay config when HIL194 is absent; explicit inputs are never substituted)')
    parser.add_argument('--episode',type=int,default=None,help='Episode index; default: fitted config episode, otherwise a verified HIL194 pickup example for automatic dynamic replay, or 0')
    parser.add_argument('--mode',choices=['replay','demo'],default='replay',help=(
        'Motion generation (default: replay). replay: follow the selected dataset episode using --source. '
        'demo: generate a 10-second scripted approach/attach/lift/eject trajectory using inverse kinematics, '
        'seeded from the episode initial pose; does not replay its recorded trajectory and ignores --source. '
        'Neither mode runs a learned policy or guarantees task success.'))
    parser.add_argument('--physics',choices=['dynamic','kinematic'],default=None,help='Dynamic: torque-limited servo tracking and rigid-body objects. Kinematic: original exact-pose preview.')
    parser.add_argument('--free-base',action='store_true',help='Remove pelvis support in dynamic mode. No balance controller is included.')
    parser.add_argument('--source',choices=['state','action','low_level'],default=None,help=(
        'Robot joint trajectory for replay (default: low_level; ignored in demo). '
        'state: measured joint positions, i.e. where the robot actually was. '
        'action: recorded high-level joint/mimic targets, applied directly to servos; this bypasses the TWIST2 controller and is not equivalent to motor replay. '
        'low_level: recorded low-level motor joint targets sent to the robot. '
        'With dynamic physics, the selected values become simulated servo targets; use low_level for command replay. '
        'With kinematic physics, they prescribe joint positions directly; use state for measured-pose playback. '
        'Dynamic + state can add tracking error and gravity sag. This option does not select hand feedback '
        'or run GROOT inference.'))
    parser.add_argument('--clock',choices=['recorded','nominal'],default='recorded')
    parser.add_argument('--config',type=Path,default=None,help='Explicit workspace config; otherwise use the default dataset and matching per-episode pickup fit when available')
    parser.add_argument('--scene-layout',choices=['auto','visual','fixed'],default='auto',help='auto: use the saved episode pickup fit for default dynamic low_level replay, otherwise initialize visually; visual: use the original image-based estimate; fixed: use supplied object poses unchanged. Replay only.')
    parser.add_argument('--scene-cache',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/episode-scenes')
    parser.add_argument('--refresh-scene',action='store_true',help='Re-estimate this episode layout instead of using the matching cache.')
    parser.add_argument('--fit-layout',action='store_true',help='Initialize rack/table from low nozzle positions. A kinematic guess, not calibrated reconstruction.')
    parser.add_argument('--show-collisions',action='store_true',help='Show collision meshes in the main viewer and exports for geometry inspection.')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--no-camera-window',action='store_true',help='Disable the separate camera window (replay: synchronized real/sim views; demo: sim views only; never opened in headless mode).')
    parser.add_argument('--duration',type=float,help='Maximum simulation seconds, capped at episode/demo duration.')
    parser.add_argument('--speed',type=float,default=1)
    parser.add_argument('--camera-pose',choices=['corrected','config'],default='corrected',help='corrected: apply dataset/session camera-view corrections, including with --config (default). config: preserve the camera poses in the selected config.')
    parser.add_argument('--camera',choices=['overview','rgb','wrist_left','wrist_right'],default='overview')
    parser.add_argument('--video',action='store_true',help='Export MP4 at 30 fps. Interactive recordings follow the viewer camera, including mouse zoom/pan/orbit; --headless uses --camera. --compare keeps fixed camera pairs.')
    parser.add_argument('--snapshot',action='store_true')
    parser.add_argument('--compare',action='store_true',help='Render three real/sim view pairs; implies --video. Use with replay.')
    parser.add_argument('--output',type=Path,default=Path('outputs/latest'))
    args=parser.parse_args()
    if args.list_environments:
        for name,entry in ENVIRONMENTS.items():print(f"{name:24s} {entry['description']}")
        return
    imported=args.environment is not None and ENVIRONMENTS[args.environment]['kind']=='imported'
    args.physics=args.physics or 'dynamic'
    args.source=args.source or ('state' if imported and args.physics=='kinematic' else 'low_level')
    if args.speed<=0 or (args.duration is not None and args.duration<=0):parser.error('speed and duration must be positive')
    if args.free_base and args.physics!='dynamic':parser.error('--free-base requires --physics dynamic')
    if args.compare and args.mode!='replay':parser.error('--compare requires --mode replay')
    if args.hand!='inspire' and not imported:parser.error('--hand wuji currently requires --environment lidar; the pipette workspace grasp, ejector and contact settings are tuned for Inspire.')
    if args.headless:os.environ.setdefault('MUJOCO_GL','egl')
    if not args.headless and not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):parser.error('No display found. Use --headless (optionally --video).')
    if imported:
        if args.mode!='replay' or args.free_base:
            parser.error('LIDAR supports fixed-base replay only; demo and free-base control are not implemented.')
        if args.physics=='kinematic' and args.source!='state':
            parser.error('LIDAR kinematic diagnostics require --source state; use dynamic physics for motor target replay.')
        if args.camera!='overview':parser.error('LIDAR currently has an overview camera only; policy camera calibration is not yet available.')
        from .environments.viewer import run_imported
        run_imported(args)
        return
    import mujoco
    import numpy as np
    from .dataset import Episode,DEFAULT_DATASET,resolve_dataset_root
    from .scene import load_config,fit_layout
    from .simulation import Simulation
    from .replay_defaults import configure_default_replay
    try:default_note=configure_default_replay(args)
    except (FileNotFoundError,ValueError) as error:parser.error(str(error))
    if default_note:print(default_note,flush=True)
    local_config=Path(__file__).resolve().parents[1]/'.local/hil194.json'
    cfg=load_config(args.config or (local_config if local_config.exists() else None))
    args.dataset=args.dataset or cfg.get('layout_search',{}).get('dataset') or cfg.get('replay_audit_setup',{}).get('dataset') or DEFAULT_DATASET
    try:args.dataset=resolve_dataset_root(args.dataset)
    except FileNotFoundError as error:parser.error(str(error))
    if args.episode is None:
        args.episode=cfg.get('layout_search',{}).get('episode',0)
        if (args.config is None and args.scene_layout=='auto' and not args.refresh_scene and not args.fit_layout
                and args.mode=='replay' and args.physics=='dynamic' and args.source=='low_level'
                and args.clock=='recorded' and not args.free_base):
            from .pickup_layouts import recommended_episode
            args.episode=recommended_episode(resolve_dataset_root(args.dataset))
        print(f'Default episode: {args.episode}',flush=True)
    episode=Episode(args.dataset,args.episode,args.clock)
    print(f'Dataset: {episode.root}; hand actions: {episode.hand_action_provenance}',flush=True)
    if args.mode=='replay':
        from .pickup_layouts import recorded_reviews
        review=recorded_reviews(episode.root).get(str(episode.index),{})
        if review.get('note'):print(f"Recorded-video note: {review['note']}",flush=True)
    if episode.timing['action_timestamps_frozen']:
        print('Recorded action timestamps are frozen; commands use their paired image-row times. Actual command latency is unknown.',flush=True)
    if (args.config is None and args.scene_layout=='auto' and not args.refresh_scene and not args.fit_layout
            and args.mode=='replay' and args.physics=='dynamic' and args.source=='low_level'
            and args.clock=='recorded' and not args.free_base):
        from .pickup_layouts import select_pickup_layout
        try:cfg,pickup=select_pickup_layout(cfg,episode)
        except ValueError as error:parser.error(str(error))
        if pickup:
            print(f"Episode pickup config: {pickup['path']}; status={pickup['status']}",flush=True)
            validation=pickup.get('validation',{})
            if 'lift_target_m' in validation:
                height_label="world-frame lift" if validation.get("lift_frame")=="world" else "rack clearance (not world-frame lift)"
                print(f"Saved full-replay {height_label}: {validation['lift_m']*1000:.2f} mm simulated; {validation['lift_target_m']*1000:.2f} mm recorded-joint estimate (tolerance ±{validation['lift_tolerance_m']*1000:.2f} mm).",flush=True)
            if pickup['status']!='verified_pickup':
                print('WARNING: placement search has not verified pickup with the recorded-motion lift target for this episode. This is its best available diagnostic layout.',flush=True)
    fitted=cfg.get('layout_search')
    if fitted:
        result=fitted.get('result',{})
        print(f"Fitted layout: episode {fitted.get('episode')}, source {fitted.get('source')}; collision-screened={result.get('valid',False)}, attachment-and-lift success={result.get('success',False)}.",flush=True)
        if (fitted.get('dataset') and resolve_dataset_root(fitted['dataset'])!=episode.root) or args.episode!=fitted.get('episode') or args.source!=fitted.get('source') or args.clock!=fitted.get('clock','recorded'):
            print('WARNING: dataset/episode/source/clock differs from the fitted replay. Its reported outcome does not apply to this run.',flush=True)

    if args.physics=='dynamic' and args.mode=='replay' and args.source=='state':
        print('NOTE: dynamic + state treats measured positions as servo targets and can sag under load. Use --source low_level for recorded motor commands, or --physics kinematic --source state for measured-pose playback.',flush=True)
    if args.mode=='replay' and not args.fit_layout:
        from .episode_scene import resolve_scene
        try:cfg,scene_path=resolve_scene(cfg,episode,'auto' if args.scene_layout=='visual' else args.scene_layout,args.scene_cache,args.refresh_scene)
        except ValueError as error:parser.error(f'{error}. Correct landmarks with python -m g1_pipette_sim.episode_scene --edit {args.episode}, or explicitly use --scene-layout fixed.')
        if scene_path:
            print(f'Episode scene: {scene_path}',flush=True)
            print(f"Rack={cfg['rack']['center_xy']}, active tip={cfg['rack']['active_cell']}; lid={cfg['scene_initialization']['objects']['lid']['status']}. Review the saved .overlay.jpg.",flush=True)
    if args.fit_layout:cfg=fit_layout(cfg,episode)
    from .profiles import for_episode
    cfg=for_episode(cfg,episode)
    if args.camera_pose=='corrected':
        from .camera_views import apply_camera_views
        cfg=apply_camera_views(cfg,episode)
        if cfg.get('camera_view_correction'):print('Camera-view correction: '+', '.join(cfg['camera_view_correction']['views'])+' (camera pose only)',flush=True)
    if args.physics=='dynamic':
        from .dynamics import physics_config
        cfg['physics']=physics_config(cfg)
    if args.environment is not None:
        cfg['environment']={'name':args.environment}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'workspace.json').write_text(json.dumps(cfg,indent=2)+'\n')
    print(f'Loading G1/Inspire: {args.mode}, episode {args.episode}, {args.clock} clock',flush=True)
    if args.physics=='dynamic':
        from .dynamics import DynamicSimulation
        sim=DynamicSimulation(cfg,episode,args.mode,args.source,args.free_base)
    else:sim=Simulation(cfg,episode,args.mode,args.source)
    (args.output/'scene.xml').write_text(sim.xml)
    limit=min(args.duration or sim.duration,sim.duration)
    print(f'Running {limit:.2f} seconds. Geometry/cameras are estimates; physics={args.physics}.',flush=True)
    camera_window=None;renderer=None;view_renderers={};writer=None;captures={};viewer=None;viewer_threads=[];events=SimpleQueue();paused=False;camera=args.camera
    rendering=args.video or args.snapshot or args.compare
    width,height=(960,760) if args.compare else (960,720)
    try:
        if rendering:
            import cv2
            renderer=mujoco.Renderer(sim.model,height=360 if args.compare else height,width=320 if args.compare else width)
            if args.video or args.compare:
                writer=cv2.VideoWriter(str(args.output/'simulation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(width,height))
                if not writer.isOpened():raise RuntimeError('Cannot open MP4 writer')
            if args.compare:
                for name in ['rgb','wrist_left','wrist_right']:
                    shape=episode.info['features'][f'observation.images.{name}']['shape']
                    view_renderers[name]=mujoco.Renderer(sim.model,width=320,height=round(320*shape[0]/shape[1]))
                    from .video import VideoFrames
                    cap=VideoFrames(episode.video_path(name))
                    captures[name]=cap
        if not args.headless:
            import mujoco.viewer
            threads_before=set(threading.enumerate())
            viewer=mujoco.viewer.launch_passive(sim.model,sim.data,key_callback=events.put,show_left_ui=False,show_right_ui=False)
            viewer_threads=[thread for thread in threading.enumerate() if thread not in threads_before]
            with viewer.lock():
                configure_viewer_camera(viewer.cam,sim.model,sim.data,cfg,camera)
                viewer.opt.geomgroup[3]=args.show_collisions
            if not args.no_camera_window:
                from .camera_window import CameraWindow
                resolutions={name:tuple(episode.info['features'][f'observation.images.{name}']['shape'][1::-1]) for name in ['rgb','wrist_left','wrist_right']}
                camera_window=CameraWindow(sim.model,cfg,resolutions,events,episode=episode if args.mode=='replay' else None)
            print('Space: pause/resume | R: reset | E: eject attached tip | G: release pipette grip (dynamic) | C: camera presets | F: free overview | Mouse: left orbit, right pan, wheel zoom',flush=True)
        def render_frame():
            import cv2
            option=mujoco.MjvOption();option.geomgroup[3]=args.show_collisions
            if args.compare:
                image=np.zeros((height,width,3),dtype=np.uint8)
                for col,name in enumerate(['rgb','wrist_left','wrist_right']):
                    cap=captures[name];idx=episode.frame_at(sim.time)
                    real=cap.read(idx)
                    rect=cfg['cameras'][name].get('rectification')
                    if rect:
                        K=np.asarray(rect['camera_matrix']);dist=np.asarray(rect['distortion_coefficients']);Knew=np.asarray(rect['rectified_camera_matrix'])
                        if list(real.shape[1::-1])!=rect['resolution']:raise ValueError(f'Calibration resolution does not match {name} video')
                        if rect['lens_model']=='fisheye':real=cv2.fisheye.undistortImage(real,K,dist,Knew=Knew)
                        else:real=cv2.undistort(real,K,dist,None,Knew)
                    real=cv2.cvtColor(real,cv2.COLOR_BGR2RGB);rh,rw=real.shape[:2];scale=min(320/rw,340/rh);real=cv2.resize(real,(round(rw*scale),round(rh*scale)))
                    image[25:25+real.shape[0],col*320:col*320+real.shape[1]]=real
                    cv2.putText(image,f'real: {name}',(col*320+5,18),0,.45,(235,235,235),1)
                    view_renderer=view_renderers[name];view_renderer.update_scene(sim.data,camera=name,scene_option=option);rendered=view_renderer.render();image[395:395+rendered.shape[0],col*320:(col+1)*320]=rendered
                    cv2.putText(image,f'sim: {name} (estimated)',(col*320+5,389),0,.42,(235,235,235),1)
            else:
                renderer.update_scene(sim.data,camera=viewer.cam if viewer is not None else camera,scene_option=option);image=renderer.render().copy()
                cv2.putText(image,f'{args.mode} | ep {args.episode} | {sim.time:.2f}s | tip: {sim.state}',(20,30),0,.65,(235,235,235),2)
                cv2.putText(image,f'Estimated workspace / {args.physics} physics',(20,height-18),0,.5,(235,235,235),1)
            return image
        next_frame=0.;frame_count=0;last_wall=time.monotonic();wall_credit=0.;event_count=0
        while sim.time<limit-1e-9 and (viewer is None or viewer.is_running()):
            if viewer:
                while not events.empty():
                    key=events.get()
                    if key==32:paused=not paused
                    elif key in [82,114]:
                        if writer:print('Reset disabled during video export; rerun to start a new recording.',flush=True)
                        else:
                            with viewer.lock():sim.reset()
                            next_frame=0;event_count=0
                    elif key in [69,101]:
                        with viewer.lock():sim.eject()
                    elif key in [71,103] and args.physics=='dynamic':
                        with viewer.lock():sim.release_grip()
                    elif key in [70,102]:
                        camera='overview'
                        with viewer.lock():configure_viewer_camera(viewer.cam,sim.model,sim.data,cfg,camera)
                    elif key in [67,99]:
                        names=['overview','rgb','wrist_left','wrist_right'];camera=names[(names.index(camera)+1)%len(names)]
                        with viewer.lock():configure_viewer_camera(viewer.cam,sim.model,sim.data,cfg,camera)
                now=time.monotonic();elapsed=now-last_wall;last_wall=now
                if camera_window:
                    with viewer.lock():camera_window.update(sim.data,sim.time,paused)
                if paused:viewer.sync();time.sleep(.01);continue
                wall_credit+=min(elapsed,.1)*args.speed
                count=min(int(wall_credit/sim.model.opt.timestep),250)
                wall_credit-=count*sim.model.opt.timestep
            else:count=1
            if viewer:
                with viewer.lock():
                    for _ in range(count):
                        if sim.time>=limit:break
                        sim.step()
                viewer.sync()
            else:sim.step()
            if len(sim.events)>event_count:
                for event in sim.events[event_count:]:print(json.dumps(event),flush=True)
                event_count=len(sim.events)
            if renderer and (writer or frame_count==0) and sim.time>=next_frame-1e-9:
                image=render_frame()
                if args.snapshot and frame_count==0:cv2.imwrite(str(args.output/'snapshot.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                if writer:writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                frame_count+=1;next_frame=frame_count/30
            if viewer:time.sleep(.001)
        if renderer:cv2.imwrite(str(args.output/'final.png'),cv2.cvtColor(render_frame(),cv2.COLOR_RGB2BGR))
        report=sim.report();report['scene_initialization']=cfg.get('scene_initialization');report['camera_window_frames']=camera_window.frames if camera_window else 0;report['video_frames']=frame_count if writer else 0
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        if args.physics=='dynamic' and not any(event['state']=='attached' for event in sim.events):
            closest=report['attachment_diagnostics']['closest_approach']
            if closest:
                print(f"No tip attachment during this run. Closest approach: lateral {closest['radial_error_mm']:.1f} mm (required < {cfg['attachment']['radial_tolerance']*1000:.1f}), insertion depth {closest['insertion_depth_mm']:.1f} mm, angle {closest['angle_deg']:.1f} deg. See attachment_diagnostics in report.json.",flush=True)
        print(f'Outputs: {args.output.resolve()}',flush=True)
    finally:
        if camera_window:camera_window.close()
        if writer:writer.release()
        for cap in captures.values():cap.release()
        if renderer:renderer.close()
        for view_renderer in view_renderers.values():view_renderer.close()
        if viewer:
            viewer.close()
            # MuJoCo 3.8 launches a daemon render thread; close() only requests
            # exit. Let it destroy its GL context before interpreter teardown.
            for thread in viewer_threads:
                thread.join(timeout=5)
                if thread.is_alive():
                    print('Viewer thread did not finish shutdown within 5 seconds.',flush=True)

if __name__=='__main__':main()
