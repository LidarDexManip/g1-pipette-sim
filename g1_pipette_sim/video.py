"""Sequential cached decoding for timestamp-paired real/sim comparisons."""
import cv2


class VideoFrames:
    def __init__(self,path):
        self.path=str(path)
        self.capture=cv2.VideoCapture(self.path,cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1])
        if not self.capture.isOpened():raise RuntimeError(f'Cannot open {path}')
        self.index=-1
        self.frame=None

    def read(self,index):
        if index<0:raise ValueError('Frame index must be nonnegative')
        if index<self.index:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES,index)
            self.index=index-1
        while self.index<index:
            ok,frame=self.capture.read()
            if not ok:raise RuntimeError(f'Cannot decode {self.path} frame {self.index+1}')
            self.index+=1;self.frame=frame
        return self.frame

    def release(self):self.capture.release()
