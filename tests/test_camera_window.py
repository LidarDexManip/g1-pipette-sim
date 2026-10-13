"""Camera-window synchronization and resource cleanup without a desktop/GPU."""
from types import SimpleNamespace
from queue import SimpleQueue
import sys
import numpy as np
import pytest
from g1_pipette_sim import camera_window, video


class Widget:
    def __init__(self,*args,**kwargs):self.options=kwargs;self.destroyed=False
    def configure(self,**kwargs):self.options.update(kwargs)
    def grid(self,**kwargs):self.grid_options=kwargs
    def title(self,*args):pass
    def protocol(self,*args):pass
    def bind(self,*args):pass
    def update_idletasks(self):pass
    def update(self):pass
    def destroy(self):self.destroyed=True


@pytest.fixture
def components(monkeypatch):
    from PIL import ImageTk
    captures=[];renders=[]
    class Capture:
        def __init__(self,path):self.reads=[];self.released=False;captures.append(self)
        def read(self,index):
            self.reads.append(index)
            # BGR: the monitor must present red, not blue, after conversion.
            return np.full((2,4,3),[index,0,255],dtype=np.uint8)
        def release(self):self.released=True
    class Stream:
        def __init__(self,model,names,width,height):self.name=names[0];self.size=(height,width,3);self.closed=False;renders.append(self)
        def render(self,data):return {self.name:np.zeros(self.size,dtype=np.uint8)}
        def __enter__(self):return self
        def __exit__(self,*args):self.closed=True
    monkeypatch.setitem(sys.modules,'tkinter',SimpleNamespace(Tk=Widget,Label=Widget))
    monkeypatch.setattr(ImageTk,'PhotoImage',lambda image,master:image.copy())
    monkeypatch.setattr(video,'VideoFrames',Capture)
    monkeypatch.setattr(camera_window,'CameraStreams',Stream)
    monkeypatch.setattr(camera_window.time,'monotonic',lambda:100.)
    names=['rgb','wrist_right','wrist_left']
    return SimpleNamespace(captures=captures,renders=renders,config={'cameras':{n:{} for n in names}},resolutions={n:(4,2) for n in names})


def make_window(parts,episode=None):
    return camera_window.CameraWindow(None,parts.config,parts.resolutions,SimpleQueue(),width=8,episode=episode)


def test_timestamp_sync_pause_and_reset(components):
    times=[]
    def frame_at(t):times.append(t);return 7 if t>=.12 else 0
    ep=SimpleNamespace(index=3,frame_at=frame_at,video_path=lambda name:name)
    w=make_window(components,ep)
    w.update(None,.15)
    assert [c.reads for c in components.captures]==[[7],[7],[7]]
    image=w.real_labels['rgb'].options['image']
    assert image.size==(8,4) and image.getpixel((0,0))==(255,0,7)
    assert len(w.photos)==6
    w.next_frame=0;w.update(None,.15,paused=True)
    assert [c.reads for c in components.captures]==[[7],[7],[7]]
    assert 'Paused' in w.status.options['text']
    w.next_frame=0;w.update(None,0.)
    assert [c.reads for c in components.captures]==[[7,0],[7,0],[7,0]]
    assert times==[.15,.15,0.]
    w.close();w.close()
    assert all(c.released for c in components.captures)
    assert all(r.closed for r in components.renders) and w.root.destroyed


def test_demo_has_only_simulator_views(components):
    w=make_window(components);w.update(None,1.)
    assert not components.captures and not w.real_labels
    assert len(w.photos)==3
    w.close()


def test_decoder_failure_closes_already_open_video(components,monkeypatch):
    original=video.VideoFrames
    def broken(path):
        if path=='wrist_right':raise RuntimeError('bad video')
        return original(path)
    monkeypatch.setattr(video,'VideoFrames',broken)
    ep=SimpleNamespace(index=3,video_path=lambda name:name)
    with pytest.raises(RuntimeError,match='bad video'):make_window(components,ep)
    assert len(components.captures)==1 and components.captures[0].released
