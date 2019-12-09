import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from videoio import VideoReader

class Dataset :
    def __init__(self, content_dir, style_dir) :
        self.content_videos = glob.glob(content_dir + '/*.avi')
        self.style_images = glob.glob(style_dir + '/*')
        self.shuffle()

    def __len__(self):
        return len(self.images_pairs)

    def __getitem__(self, index):
        content_video, style_image = self.images_pairs[index]
        print(f'loading ({content_video}, {style_image})')
        optical_flow = torch.from_numpy(np.load(os.path.splitext(content_video)[0] + '.npy'))
        content_video = torch.stack(VideoReader(content_video).read())
        style_image = transforms.ToTensor()(Image.open(style_image)).unsqueeze(0)
        return content_video, style_image, optical_flow.to(torch.float32)
    
    def shuffle(self) :
        np.random.shuffle(self.content_videos)
        np.random.shuffle(self.style_images)
        self.images_pairs = list(zip(self.content_videos, self.style_images))
