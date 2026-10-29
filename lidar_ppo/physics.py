"""MuJoCo CPU reference and zero-copy Torch/MuJoCo-Warp batched physics."""
import numpy as np
import torch


class CPUPhysics:
    def __init__(self, model, nworld, device="cpu", **unused):
        import mujoco
        if str(device) != "cpu":
            raise ValueError("CPU backend is for validation; use --backend warp on CUDA")
        self.mujoco, self.model = mujoco, model
        self.data = [mujoco.MjData(model) for _ in range(nworld)]
        self.nworld, self.renderer = nworld, None
        self.fields = {}
        self.pipette = model.body("env_01_pipette").id
        self.hand = {i for i in range(model.nbody) if model.body(i).name.startswith("R_") or model.body(i).name == "right_wrist_yaw_link"}
        self.support = {i for i in range(model.nbody) if i != self.pipette and (i == 0 or model.body(i).name.startswith("env_"))}
        self.metrics = torch.zeros(nworld, 4)
        self.peak = torch.zeros(nworld, 4)
        self.nominal = {k:getattr(model,k).copy() for k in ("geom_friction","actuator_gainprm","actuator_biasprm")}
        self.factors = np.ones((nworld,2))
        self.sync()

    def randomize(self,mask,friction,servo):
        self.factors[mask.cpu().numpy(),0]=friction[mask].cpu().numpy()
        self.factors[mask.cpu().numpy(),1]=servo[mask].cpu().numpy()

    def apply_model(self,i):
        self.model.geom_friction[:]=self.nominal["geom_friction"]
        self.model.geom_friction[:,0]*=self.factors[i,0]
        self.model.actuator_gainprm[:]=self.nominal["actuator_gainprm"]
        self.model.actuator_biasprm[:]=self.nominal["actuator_biasprm"]
        self.model.actuator_gainprm[:,0]*=self.factors[i,1]
        self.model.actuator_biasprm[:,1:3]*=self.factors[i,1]

    def sync(self):
        for key in ("qpos", "qvel", "ctrl", "xipos", "xquat", "xpos", "cvel"):
            self.fields[key] = torch.tensor(np.stack([getattr(d,key) for d in self.data]), dtype=torch.float32)

    def get(self, key):
        return self.fields[key]

    def reset(self, mask, values):
        for i in torch.where(mask)[0].tolist():
            self.apply_model(i)
            d = self.data[i]; self.mujoco.mj_resetData(self.model, d)
            for key, value in values.items():
                getattr(d,key)[:] = value[i].cpu().numpy()
            self.mujoco.mj_forward(self.model, d)
        self.sync()

    def contacts(self, i):
        d, m = self.data[i], self.model
        row = np.zeros(4)
        for j, contact in enumerate(d.contact):
            row[0] = max(row[0], -contact.dist)
            a,b = m.geom_bodyid[contact.geom]
            if self.pipette not in (a,b):
                continue
            force = np.zeros(6); self.mujoco.mj_contactForce(m,d,j,force)
            other = b if a == self.pipette else a
            row[1 if other in self.hand else 2 if other in self.support else 3] += max(0,force[0])
        return torch.tensor(row, dtype=torch.float32)

    def forward(self):
        for i,d in enumerate(self.data):
            self.apply_model(i)
            self.mujoco.mj_forward(self.model,d)
            self.metrics[i] = self.contacts(i)
        self.sync()

    def step(self, ctrl, steps):
        self.peak.zero_()
        for i,d in enumerate(self.data):
            self.apply_model(i)
            d.ctrl[:] = ctrl[i].cpu().numpy()
            for _ in range(steps):
                self.mujoco.mj_step(self.model,d)
                if any(w.number for w in d.warning):
                    raise RuntimeError("MuJoCo numerical/capacity warning")
                self.metrics[i] = self.contacts(i)
                self.peak[i] = torch.maximum(self.peak[i], self.metrics[i])
        self.sync()

    def render(self, camera, width, height):
        if self.renderer is None:
            self.renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        frames = []
        for d in self.data:
            self.renderer.update_scene(d, camera=camera)
            frames.append(self.renderer.render().copy())
        return torch.from_numpy(np.stack(frames)).permute(0,3,1,2)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


class WarpPhysics:
    def __init__(self, model, nworld, device="cuda:0", nconmax=512, njmax=2048, cuda_graph=True):
        import warp as wp
        import mujoco_warp as mjw
        from .warp_metrics import reduce_contacts, update_peak, track_overflow
        if not str(device).startswith("cuda"):
            raise ValueError("Warp production backend requires CUDA")
        self.wp, self.mjw, self.model = wp, mjw, model
        self.reduce, self.update_peak = reduce_contacts, update_peak
        self.track_overflow=track_overflow
        self.device, self.nworld, self.graphs = str(device), nworld, {}
        self.use_graph = cuda_graph
        wp.init()
        # Use the same stream for Torch views, physics, and controller inference.
        self.stream = wp.stream_from_torch(torch.cuda.current_stream(device))
        with wp.ScopedDevice(device), wp.ScopedStream(self.stream):
            self.m = mjw.put_model(model)
            self.d = mjw.make_data(model, nworld=nworld, nconmax=nconmax, njmax=njmax)
            self.force = wp.zeros(self.d.naconmax, dtype=wp.spatial_vector)
            self.ids = wp.array(np.arange(self.d.naconmax,dtype=np.int32))
            self.stats = wp.zeros((nworld,4),dtype=float)
            self.peaks = wp.zeros((nworld,4),dtype=float)
            self.overflow=wp.zeros(1,dtype=int)
            self.metrics, self.peak = wp.to_torch(self.stats), wp.to_torch(self.peaks)
            category = np.zeros(model.nbody,dtype=np.int32)
            for i in range(model.nbody):
                name = model.body(i).name or ""
                category[i] = 1 if name.startswith("R_") or name == "right_wrist_yaw_link" else 2 if i == 0 or name.startswith("env_") else 3
            self.category = wp.array(category)
            self.body_ids = wp.array(model.geom_bodyid.astype(np.int32))
            self.nominal={}
            for key in ("geom_friction","actuator_gainprm","actuator_biasprm"):
                source=getattr(self.m,key)
                tensor=wp.to_torch(source)
                expanded=tensor.expand(nworld,*tensor.shape[1:]).clone()
                setattr(self.m,key,wp.from_torch(expanded,dtype=source.dtype))
                self.nominal[key]=expanded.clone()
        self.pipette = model.body("env_01_pipette").id
        self.rc = None

    def randomize(self,mask,friction,servo):
        for key in self.nominal:
            array=self.wp.to_torch(getattr(self.m,key))
            values=self.nominal[key][mask].clone()
            if key=="geom_friction":values[:,:,0]*=friction[mask,None]
            elif key=="actuator_gainprm":values[:,:,0]*=servo[mask,None]
            else:values[:,:,1:3]*=servo[mask,None,None]
            array[mask]=values

    def get(self, key):
        return self.wp.to_torch(getattr(self.d,key))

    def measure(self):
        wp,mjw,d = self.wp,self.mjw,self.d
        self.stats.zero_()
        mjw.contact_force(self.m,d,self.ids,False,self.force)
        wp.launch(self.reduce,dim=d.naconmax,inputs=[d.nacon,d.contact.worldid,d.contact.geom,
            d.contact.dist,self.body_ids,self.category,self.pipette,self.force],outputs=[self.stats])
        wp.launch(self.update_peak,dim=(self.nworld,4),inputs=[self.stats],outputs=[self.peaks])
        wp.launch(self.track_overflow,dim=self.nworld,inputs=[d.nacon,d.nefc,d.naconmax,d.njmax],outputs=[self.overflow])

    def reset(self, mask, values):
        with self.wp.ScopedDevice(self.device), self.wp.ScopedStream(self.stream):
            self.mjw.reset_data(self.m,self.d,self.wp.from_torch(mask.contiguous()))
            for key,value in values.items():
                dest = self.get(key)
                dest[mask] = value[mask]
            self.mjw.forward(self.m,self.d)

    def forward(self):
        with self.wp.ScopedDevice(self.device), self.wp.ScopedStream(self.stream):
            self.mjw.forward(self.m,self.d)
            self.measure()
        self.check_capacity()

    def step(self, ctrl, steps):
        wp = self.wp
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            self.get("ctrl").copy_(ctrl)
            self.peaks.zero_()
            if self.use_graph and steps not in self.graphs:
                # Warm kernels on a separate state; do not advance the live environments.
                warm = self.mjw.make_data(self.model,nworld=self.nworld,
                                          naconmax=self.d.naconmax,njmax=self.d.njmax)
                self.mjw.step(self.m,warm)
                self.measure()
                with wp.ScopedCapture() as capture:
                    for _ in range(steps):
                        self.mjw.step(self.m,self.d)
                        self.measure()
                self.graphs[steps] = capture.graph
                self.peaks.zero_()
            if self.use_graph:
                wp.capture_launch(self.graphs[steps])
            else:
                for _ in range(steps):
                    self.mjw.step(self.m,self.d)
                    self.measure()
        self.check_capacity()

    def check_capacity(self):
        if bool(self.wp.to_torch(self.overflow)[0]) or int(self.wp.to_torch(self.d.nacon)[0]) > self.d.naconmax or bool((self.wp.to_torch(self.d.nefc) > self.d.njmax).any()):
            raise RuntimeError("Warp contact/constraint overflow: increase nconmax/njmax; do not continue")

    def render(self, camera, width, height):
        wp,mjw = self.wp,self.mjw
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            if self.rc is None:
                index = self.model.camera(camera).id
                self.rc = mjw.create_render_context(self.model,nworld=self.nworld,cam_res=(width,height),
                    render_rgb=True,render_depth=False,render_seg=False,
                    cam_active=[i == index for i in range(self.model.ncam)],
                    enabled_geom_groups=[0,1,2],use_shadows=False)
                self.rgb = wp.zeros((self.nworld,height,width),dtype=wp.vec3)
            mjw.kinematics(self.m,self.d); mjw.camlight(self.m,self.d)
            mjw.refit_bvh(self.m,self.d,self.rc); mjw.render(self.m,self.d,self.rc)
            mjw.get_rgb(self.rc,0,self.rgb)
            return (wp.to_torch(self.rgb)*255).clamp(0,255).to(torch.uint8).permute(0,3,1,2)

    def close(self):
        pass
