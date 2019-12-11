import cv2 as cv
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from videoio import *
from tqdm import tqdm

def parse_args() :
    parser = argparse.ArgumentParser(description='Warp optical flow')
    parser.add_argument('--content', '-c', type=str, default=None, required=True,
                        help='Content video path, e.g. content.avi')
    parser.add_argument('--flow', '-f', type=str, default=None, required=True,
                        help='optical flow path, e.g. content.npy')
    parser.add_argument('--output_name', '-o', type=str, default=None, required=True,
                        help='Output file name for generated video, e.g. output.avi')
    return parser.parse_args()

def main() :
    args = parse_args()
    optical_flow = torch.from_numpy(np.load(args.flow))
    content_video = VideoReader(args.content).read()
    writer = VideoWriter(args.output_name, 30.0)
    for id in tqdm(range(len(content_video) - 1)) :
        frame = content_video[id]
        flow = optical_flow[id]
        writer.write(F.grid_sample(frame.unsqueeze(0), flow.unsqueeze(0), padding_mode='border'))

if __name__ == '__main__' :
    main()
