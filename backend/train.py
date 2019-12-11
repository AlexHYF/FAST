#!/usr/bin/env python3
import warnings
warnings.simplefilter("ignore", UserWarning)
import os
import argparse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import Dataset
from model import Model

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer by Pytorch')
    parser.add_argument('--batch_size', '-b', type=int, default=8,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID(nagative value indicate CPU)')
    parser.add_argument('--learning_rate', '-lr', type=int, default=5e-5,
                        help='learning rate for Adam')
    parser.add_argument('--train_content_dir', type=str, default='dataset/train',
                        help='content images directory for train')
    parser.add_argument('--train_style_dir', type=str, default='style',
                        help='style images directory for train')
    parser.add_argument('--test_content_dir', type=str, default='dataset/test',
                        help='content images directory for test')
    parser.add_argument('--test_style_dir', type=str, default='style',
                        help='style images directory for test')
    parser.add_argument('--save_dir', type=str, default='model',
                        help='save directory for result and loss')
    parser.add_argument('--reuse', default='model_state.pth',
                        help='model state path to load for reuse')

    args = parser.parse_args()

    # create directory to save
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    loss_dir = f'{args.save_dir}/loss'
    model_state_dir = f'{args.save_dir}/model_state'

    if not os.path.exists(loss_dir):
        os.mkdir(loss_dir)
        os.mkdir(model_state_dir)

    # set device on GPU if available, else CPU
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
        print(f'# CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        device = 'cpu'

    print(f'# epoch: {args.epoch}')
    print('')

    # prepare dataset and dataLoader
    train_dataset = Dataset(args.train_content_dir, args.train_style_dir)
    iters = len(train_dataset)
    print(f'Length of train video/style pairs: {iters}')

    # set model and optimizer
    model = Model().to(device)
    if args.reuse is not None:
        model.load_state_dict(torch.load(args.reuse))
        print(f'# model path file "{args.reuse}" loaded')
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    # start training
    batch_size = args.batch_size
    loss_list = []
    for e in range(1, args.epoch + 1):
        print(f'Start {e} epoch')
        for i, (content_video, style, optical_flow) in enumerate(train_dataset, 1) :
            style = style.to(device)
            for id in range(content_video.shape[0]) :
                content_video[id] = normalize(content_video[id])
            for chunk in tqdm(range(0, content_video.shape[0] - batch_size, batch_size)) :
                content = content_video[chunk:chunk+batch_size].contiguous().to(device)
                flow = optical_flow[chunk:chunk+batch_size-1].contiguous().to(device)
                loss = model(content, style, flow)
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'[{e}/{args.epoch} epoch], '
                  f'[{i} /{len(train_dataset)} iteration]: {loss.item()}')
        train_dataset.shuffle()
        torch.save(model.state_dict(), f'{model_state_dir}/{e}_epoch.pth')
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.savefig(f'{loss_dir}/train_loss.png')
    with open(f'{loss_dir}/loss_log.txt', 'w') as f:
        for l in loss_list:
            f.write(f'{l}\n')
    print(f'Loss saved in {loss_dir}')


if __name__ == '__main__':
    main()
