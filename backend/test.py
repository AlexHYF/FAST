#!/usr/bin/env python3
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import argparse
from tqdm import tqdm
from adain import AdaIN
from videoio import VideoReader, VideoWriter

def parse_args() :
    parser = argparse.ArgumentParser(description='AdaIN Video Style Trasfer')
    parser.add_argument('--batch_size', '-b', type=int, default=4,
                        help='Transform batch size')
    parser.add_argument('--content', '-c', type=str, default=None, required=True,
                        help='Content video path, e.g. content.mp4')
    parser.add_argument('--fourcc', '-f', type=str, default='XVID',
                        help='Four character code for codec')
    parser.add_argument('--style', '-s', type=str, default=None, required=True,
                        help='Style image path, e.g. style.jpg')
    parser.add_argument('--output_name', '-o', type=str, default=None, required=True,
                        help='Output file name for generated video, e.g. output.avi')
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
    transformer = AdaIN(args.model_state_path, ToTensor()(Image.open(args.style)), args.alpha, device)
    print(f'# model state loaded from "{args.model_state_path}"')
    reader = VideoReader(args.content)
    writer = VideoWriter(args.output_name, reader.fps)
    batch_size = args.batch_size
    for _ in tqdm(range(0, reader.length, batch_size)) :
        result = transformer(reader.read(batch_size))
        writer.write(result)

if __name__ == '__main__' :
    main()