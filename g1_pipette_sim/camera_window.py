"""Live three-camera monitor, serviced on the simulation thread."""
import time
from contextlib import ExitStack
from .camera_streams import CameraStreams


class CameraWindow:
    def __init__(self, model, config, resolutions, events, fps=15, width=360):
        import tkinter as tk
        from PIL import Image, ImageTk
        self.tk=tk;self.Image=Image;self.ImageTk=ImageTk
        self.stack=ExitStack();self.root=None;self.closed=False;self.frames=0
        self.next_frame=0.;self.interval=1/fps;self.photos={};self.labels={}
        self.names=('rgb','wrist_right','wrist_left')
        try:
            self.root=tk.Tk();self.root.title('G1 simulator — camera views')
            self.root.configure(bg='#20242a')
            self.root.protocol('WM_DELETE_WINDOW',self.close)
            self.root.bind('<Escape>',lambda event:self.close())
            def key(event):
                if event.char and event.char.lower() in ' recgf':events.put(ord(event.char))
            self.root.bind('<KeyPress>',key)
            self.cameras={}
            for col,name in enumerate(self.names):
                w,h=resolutions[name];height=round(width*h/w)
                self.cameras[name]=self.stack.enter_context(CameraStreams(model,[name],width,height))
                role=config['cameras'][name].get('role',name).replace('_',' ')
                tk.Label(self.root,text=f'{name} — {role}',bg='#20242a',fg='white',padx=8,pady=8).grid(row=0,column=col,sticky='ew')
                label=tk.Label(self.root,bg='black');label.grid(row=1,column=col,padx=3,pady=3)
                self.labels[name]=label
            self.status=tk.Label(self.root,bg='#20242a',fg='white',padx=8,pady=8)
            self.status.grid(row=2,column=0,columnspan=3,sticky='ew')
        except Exception:
            self.close();raise

    def update(self,data,simulation_time,paused=False):
        if self.closed:return
        # Pump events even while paused or when a render is not yet due.
        self.root.update_idletasks();self.root.update()
        if self.closed:return
        now=time.monotonic()
        if now<self.next_frame:return
        self.next_frame=now+self.interval
        # All views observe the same state; no stepping occurs between renders.
        for name,camera in self.cameras.items():
            pixels=camera.render(data)[name]
            photo=self.ImageTk.PhotoImage(self.Image.fromarray(pixels),master=self.root)
            self.photos[name]=photo;self.labels[name].configure(image=photo)
        self.status.configure(text=f'{simulation_time:.2f} s | {"Paused" if paused else "Running"} | Space: pause/resume | Esc: close camera window')
        self.frames+=1

    def close(self):
        if self.closed:return
        self.closed=True
        self.stack.close()
        if self.root is not None:self.root.destroy()
        self.photos.clear()
