"""Viewer and measured-joint diagnostics for calibrated imported environments."""
import json
import time
from . import ENVIRONMENTS, imported_xml, environment_manifest


def run_imported(args):
    import mujoco
    import numpy as np
    if args.compare and args.dataset is None:
        raise ValueError('--compare requires --dataset for the real video panels.')
    args.video=args.video or args.compare
    from .robot import build_robot_scene,load_registration,apply_joint_pose,MeasuredReplay
    from .physical import build_physical_scene,PhysicalReplay
    from ..hands import HANDS
    hand=getattr(args,'hand','inspire')
    registration=load_registration(args.config)
    dynamic=args.physics=='dynamic'
    xml=(build_physical_scene if dynamic else build_robot_scene)(args.environment,registration,hand)
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    episode=None;replay=None
    if args.dataset is not None:
        from ..dataset import Episode
        episode=Episode(args.dataset,args.episode or 0,args.clock)
        if not dynamic:replay=MeasuredReplay(model,data,episode,hand);replay.pose(0)
    else:
        apply_joint_pose(model,data,registration.get('joint_pose',{}));mujoco.mj_forward(model,data)
    if dynamic:replay=PhysicalReplay(model,data,episode,args.source,registration.get('joint_pose',{}),hand)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
    view=registration['viewer'];camera.lookat[:]=view['lookat'];camera.distance=view['distance'];camera.azimuth=view['azimuth'];camera.elevation=view['elevation']
    option=mujoco.MjvOption();option.geomgroup[3]=args.show_collisions
    args.output.mkdir(parents=True,exist_ok=True)
    # Include runtime episode presence overrides in the exported preview too.
    mujoco.mj_saveLastXML(str(args.output/'scene.xml'),model)
    manifest=environment_manifest(args.environment)
    (args.output/'environment.json').write_text(json.dumps(manifest,indent=2)+'\n')
    description=('dynamic servos, gravity, free pipette/tube; fixed pelvis and furniture; grasp success unvalidated'
                 if dynamic else 'measured-position diagnostics, static props, no physical response')
    print(f"Environment: {args.environment}; {HANDS[hand]['label']}; {description}.",flush=True)
    print('Mouse: left orbit, right pan, wheel zoom. Collision groups are '+('visible.' if args.show_collisions else 'hidden.'),flush=True)
    duration=min(args.duration or episode.duration,episode.duration) if episode else (args.duration or (5.0 if args.headless else float('inf')))
    if episode:print(f'Replay: episode {episode.index}; physics={args.physics}, source={args.source}; hand sources: {replay.hand_sources}',flush=True)
    if replay:print(f'Episode scene: {replay.scene_state}',flush=True)
    (args.output/'robot-registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    renderer=None;writer=None;viewer=None;captures={}
    if args.compare:
        from ..video import VideoFrames
        for view in ('rgb','wrist_left'):captures[view]=VideoFrames(episode.video_path(view))
    try:
        if args.snapshot or args.video:
            import cv2
            renderer=mujoco.Renderer(model,height=1080,width=1280 if args.compare else 1920,
                                    max_geom=max(10000,model.ngeom+100))
            if args.video:
                writer=cv2.VideoWriter(str(args.output/'simulation.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(1920,1080))
                if not writer.isOpened():raise RuntimeError('Cannot open video writer')
        if not args.headless:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(model,data)
            with viewer.lock():
                for attr in ('lookat','distance','azimuth','elevation'):
                    if attr=='lookat':viewer.cam.lookat[:]=camera.lookat
                    else:setattr(viewer.cam,attr,getattr(camera,attr))
                viewer.opt.geomgroup[3]=args.show_collisions
        frame=0;start=time.monotonic();next_video=0.0;snapshot_done=False;next_progress=2.
        # Physics integrates floating-point timesteps; its final time can be a
        # few picoseconds below duration. Match advance()'s stopping tolerance.
        while data.time+1e-9<duration and (viewer is None or viewer.is_running()):
            if renderer is not None and data.time+1e-9>=next_video:
                if viewer:
                    with viewer.lock():renderer.update_scene(data,camera=viewer.cam,scene_option=option)
                else:renderer.update_scene(data,camera=camera,scene_option=option)
                image=renderer.render()
                if args.compare:
                    canvas=np.zeros((1080,1920,3),dtype=np.uint8);canvas[:,:1280]=image
                    for view,y,height in [('rgb',40,360),('wrist_left',440,640)]:
                        real=captures[view].read(episode.frame_at(data.time))
                        canvas[y:y+height,1280:]=cv2.cvtColor(cv2.resize(real,(640,height)),cv2.COLOR_BGR2RGB)
                        cv2.putText(canvas,'Real '+view,(1290,y-12),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                    label='Sim - physical servo replay' if dynamic else 'Sim - measured joints; static props'
                    cv2.putText(canvas,label,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                    image=canvas
                if args.snapshot and not snapshot_done:
                    cv2.imwrite(str(args.output/'snapshot.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR));snapshot_done=True
                if writer:writer.write(cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
                frame+=1;next_video=frame/30
                if args.headless and not args.video:break
            next_time=min(data.time+(1/30 if args.headless and renderer is not None else model.opt.timestep),duration)
            if dynamic:replay.advance(next_time)
            elif replay:replay.pose(next_time)
            else:data.time=next_time;mujoco.mj_forward(model,data)
            if dynamic and args.headless and data.time>=next_progress:
                print(f'Physics: {data.time:.1f}/{duration:.1f} s; '
                      f'hand/pipette contact steps={replay.pipette_robot_contact_steps}',flush=True)
                next_progress+=2.
            if viewer:
                viewer.sync();time.sleep(max(0,min(.01,start+data.time/args.speed-time.monotonic())))
        limitations=(['Fixed estimated pelvis pose; no root odometry','Furniture fixed; pipette/tube initially use approximate imported placement','Hand mapping, gains, friction and inertia need calibration','Grasp success unvalidated']
                     if dynamic else ['Kinematic playback; no physical response','Static props do not follow hands'])
        if hand!='inspire':limitations+=['Recorded Inspire registers drive Wuji joints through an approximate synergy','Robot placement was fitted with the Inspire palm, not the longer Wuji hand']
        report={'environment':args.environment,'robot':HANDS[hand]['label'],'hand':hand,'playback':'physical servo replay' if dynamic else ('measured joint positions' if replay else 'static pose'),
                'physics':args.physics,'source':args.source,
                'episode':episode.index if episode else None,'dataset':str(episode.root) if episode else None,
                'duration_s':float(data.time),'finite':bool(np.isfinite(data.qpos).all()),'hand_sources':replay.hand_sources if replay else {},
                'wall_time_s':time.monotonic()-start,
                'scene_state':replay.scene_state if replay else None,
                'dynamics':replay.report() if dynamic else None,
                'task_success':None,'limitations':limitations,
                'video_frames':frame if writer else 0}
        (args.output/'replay-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'Scene preview saved in {args.output}',flush=True)
    finally:
        if writer:writer.release()
        if renderer:renderer.close()
        if viewer:viewer.close()
        for capture in captures.values():capture.release()
