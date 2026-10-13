"""Synchronized real/sim camera monitor, serviced on the simulation thread."""
import time
from contextlib import ExitStack
from .camera_streams import CameraStreams


class CameraWindow:
    def __init__(self, model, config, resolutions, events, fps=15, width=360, episode=None):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk=tk;self.Image=Image;self.ImageTk=ImageTk
        self.stack=ExitStack();self.root=None;self.closed=False;self.frames=0
        self.next_frame=0.;self.interval=1/fps;self.photos={};self.labels={}
        self.names=('rgb','wrist_right','wrist_left')
        self.episode=episode;self.config=config;self.real_videos={};self.real_labels={}
        self.real_frame_index=None;self.sizes={}
        try:
            self.root=tk.Tk()
            self.root.title('G1 — real demonstration / simulation cameras' if episode is not None else 'G1 simulator — camera views')
            self.root.configure(bg='#20242a')
            self.root.protocol('WM_DELETE_WINDOW',self.close)
            self.root.bind('<Escape>',lambda event:self.close())
            def key(event):
                if event.char and event.char.lower() in ' recgf':events.put(ord(event.char))
            self.root.bind('<KeyPress>',key)
            self.cameras={}
            if episode is not None:
                from .video import VideoFrames
                for name in self.names:
                    capture=VideoFrames(episode.video_path(name))
                    self.stack.callback(capture.release);self.real_videos[name]=capture
            sim_row=2 if episode is not None else 0
            for col,name in enumerate(self.names):
                w,h=resolutions[name];height=round(width*h/w);self.sizes[name]=(width,height)
                self.cameras[name]=self.stack.enter_context(CameraStreams(model,[name],width,height))
                role=config['cameras'][name].get('role',name).replace('_',' ')
                if episode is not None:
                    tk.Label(self.root,text=f'Real · {name} — {role}',bg='#20242a',fg='white',padx=8,pady=8).grid(row=0,column=col,sticky='ew')
                    label=tk.Label(self.root,bg='black');label.grid(row=1,column=col,padx=3,pady=3)
                    self.real_labels[name]=label
                tk.Label(self.root,text=f'Sim · {name} — {role}',bg='#20242a',fg='white',padx=8,pady=8).grid(row=sim_row,column=col,sticky='ew')
                label=tk.Label(self.root,bg='black');label.grid(row=sim_row+1,column=col,padx=3,pady=3)
                self.labels[name]=label
            self.status=tk.Label(self.root,bg='#20242a',fg='white',padx=8,pady=8)
            self.status.grid(row=sim_row+2,column=0,columnspan=3,sticky='ew')
        except Exception:
            self.close();raise

    def _update_real(self, simulation_time):
        if self.episode is None:return
        # Use the dataset's paired image timestamps, including on pause/reset.
        # Never derive real frame numbers from wall time or nominal video FPS.
        index=self.episode.frame_at(simulation_time)
        if index==self.real_frame_index:return
        import cv2
        for name,capture in self.real_videos.items():
            pixels=capture.read(index)
            rect=self.config['cameras'][name].get('rectification')
            if rect:
                import numpy as np
                if list(pixels.shape[1::-1])!=rect['resolution']:
                    raise ValueError(f'Calibration resolution does not match {name} video')
                K=np.asarray(rect['camera_matrix']);dist=np.asarray(rect['distortion_coefficients']);Knew=np.asarray(rect['rectified_camera_matrix'])
                if rect['lens_model']=='fisheye':pixels=cv2.fisheye.undistortImage(pixels,K,dist,Knew=Knew)
                else:pixels=cv2.undistort(pixels,K,dist,None,Knew)
            image=self.Image.fromarray(cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)).resize(self.sizes[name],self.Image.Resampling.LANCZOS)
            photo=self.ImageTk.PhotoImage(image,master=self.root)
            self.photos['real_'+name]=photo;self.real_labels[name].configure(image=photo)
        self.real_frame_index=index

    def update(self,data,simulation_time,paused=False):
        if self.closed:return
        # Pump events even while paused or when a render is not yet due.
        self.root.update_idletasks();self.root.update()
        if self.closed:return
        now=time.monotonic()
        if now<self.next_frame:return
        self.next_frame=now+self.interval
        # All six views share this replay timestamp; no simulation stepping here.
        self._update_real(simulation_time)
        for name,camera in self.cameras.items():
            pixels=camera.render(data)[name]
            photo=self.ImageTk.PhotoImage(self.Image.fromarray(pixels),master=self.root)
            self.photos[name]=photo;self.labels[name].configure(image=photo)
        source=f'Episode {self.episode.index} | real frame {self.real_frame_index} | ' if self.episode is not None else ''
        self.status.configure(text=f'{source}{simulation_time:.2f} s | {"Paused" if paused else "Running"} | Space: pause/resume | Esc: close camera window')
        self.frames+=1

    def close(self):
        if self.closed:return
        self.closed=True
        try:self.stack.close()
        finally:
            if self.root is not None:self.root.destroy()
            self.photos.clear()
