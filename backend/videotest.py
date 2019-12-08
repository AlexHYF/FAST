#!/usr/bin/env python3
import torch
import cv2 as cv
import torchvision.transforms as xfrm
from PIL import Image
import sys
import math
import argparse
from tqdm import tqdm
from adain import adain

def parse_args() :
    parser = argparse.ArgumentParser(description='AdaIN Video Style Trasfer')
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
    video, _, info = read_video(args.content, pts_unit='sec')
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'
    print(info)
    video = video.permute(0, 3, 1, 2).float() / 256.0 + 1.0 / 512.0
    transformer = adain('model_state.pth', xfrm.ToTensor()(Image.open(args.style)), device)
    result = video
    nframe = video.shape[0]
    for i in tqdm(range(nframe)) :
        result[i] = transformer(video[i])
    result = (result * 256.0).permute(0, 2, 3, 1).to(torch.uint8)
    write_video(args.output_name, result, math.ceil(info['video_fps']))

if __name__ == '__main__' :
    main()