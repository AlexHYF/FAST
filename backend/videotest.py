#!/usr/bin/env python3
import torch
from torchvision.io import read_video, write_video
import torchvision.transforms as xfrm
from PIL import Image
import sys
from tqdm import tqdm
from adain import adain

video, _, info = read_video(sys.argv[1], pts_unit='sec')
print(info)
video = video.permute(0, 3, 1, 2).float() / 256.0 + 1.0 / 512.0
transformer = adain('model_state.pth', xfrm.ToTensor()(Image.open('trial.jpg')), 'cuda:0')
result = video
nframe = video.shape[0]
for i in tqdm(range(nframe)) :
    result[i] = transformer(video[i])
result = (result * 256.0).permute(0, 2, 3, 1).to(torch.uint8)
write_video('result.mp4', result, 30)