import numpy as np
import cv2 as cv
import torch

class VideoReader :
    def __init__(self, filename) :
        cap = cv.VideoCapture(filename)
        self.size = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
        self.length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv.CAP_PROP_FPS)
        self._cap = cap

    def read(self, nr_frame = -1) :
        cap = self._cap
        video = []
        while nr_frame and cap.isOpened() :
            ret, frame = cap.read()
            if not ret : break
            frame = torch.tensor(frame)
            frame = frame.permute(2, 0, 1).float() / 256.0 + 1.0 / 512.0
            video.append(frame)
            nr_frame -= 1
        return video

class VideoWriter :
    def __init__(self, filename, fps, fourcc='XVID') :
        self.filename = filename
        self.fps = fps
        self.fourcc = cv.VideoWriter_fourcc(*fourcc)
        self.size = None
        self.__writer = None

    def __init_writer(self, size = None) :
        if size : self.size = tuple(size)
        self.__writer = cv.VideoWriter(self.filename, self.fourcc, self.fps, self.size)

    def write(self, video) :
        for frame in video :
            if not self.__writer : self.__init_writer((frame.shape[2], frame.shape[1]))
            frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-6)
            frame = (frame * 256.0).permute(1, 2, 0).to(torch.uint8)
            self.__writer.write(np.array(frame))

