#!/usr/bin/env python3
import torch
import numpy as np
import cv2 as cv
import torchvision.transforms as xfrm
from PIL import Image
import sys
import math
import argparse
from tqdm import tqdm
from adain import adain

class VideoReader :
    def __init__(self, filename) :
        cap = cv.VideoCapture(filename)
        self.size = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        )
        self.length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        self.fps = math.ceil(cap.get(cv.CAP_PROP_FPS))
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
    def __init__(self, filename, fps, size, fourcc='XVID') :
        writer = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*fourcc), fps, size)
        self._writer = writer

    def write(self, video) :
        writer = self._writer
        for frame in video :
            frame = (frame * 256.0).permute(1, 2, 0).to(torch.uint8)
            writer.write(np.array(frame))

def parse_args() :
    parser = argparse.ArgumentParser(description='AdaIN Video Style Trasfer')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Transform batch size')
    parser.add_argument('--content', '-c', type=str, default=None, required=True,
                        help='Content video path, e.g. content.mp4')
    parser.add_argument('--style', '-s', type=str, default=None, required=True,
                        help='Style image path, e.g. style.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None, required=True,
                        help='Output file name for generated video, e.g. output.mp4')
    parser.add_argument('--alpha', '-a', type=float, default=1.0,
                        help='alpha control the fusion degree in AdaIN')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (nagative value indicates CPU)')
    parser.add_argument('--model_state_path', type=str, default='model_state.pth',
                        help='save directory for result and loss')
    return parser.parse_args()

def main() :
    args = parse_args()
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
    transformer = adain(args.model_state_path, xfrm.ToTensor()(Image.open(args.style)), device)
    print(f'# model state loaded from "{args.model_state_path}"')
    reader = VideoReader(args.content)
    writer = VideoWriter(args.output_name, reader.fps, reader.size)
    batch_size = args.batch_size
    for _ in tqdm(range(0, reader.length, batch_size)) :
        result = transformer(reader.read(batch_size))
        writer.write(result)

if __name__ == '__main__' :
    main()