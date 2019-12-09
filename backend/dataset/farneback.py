import cv2 as cv
import numpy as np
import argparse
from tqdm import tqdm

def parse_args() :
    parser = argparse.ArgumentParser(description='Compute Farneback dense optical flow')
    parser.add_argument('--content', '-c', type=str, default=None, required=True,
                        help='Content video path, e.g. content.mp4')
    parser.add_argument('--output_name', '-o', type=str, default=None, required=True,
                        help='Output file name for generated video, e.g. output.npz')
    parser.add_argument('--silent', '-s', help='Enable silent mode', action='store_true')
    return parser.parse_args()

def to_gray(image) :
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def main() :
    args = parse_args()
    cap = cv.VideoCapture(args.content)
    flow = []
    _, last = cap.read()
    last = to_gray(last)
    nr_frame = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    iter_obj = range(1, nr_frame)
    if not args.silent : iter_obj = tqdm(iter_obj)
    for _ in iter_obj :
        ret, cur = cap.read()
        if not ret : break
        cur = to_gray(cur)
        flow.append(cv.calcOpticalFlowFarneback(last, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0))
        last = cur
    flow = np.stack(flow)
    np.save(args.output_name, flow)

if __name__ == '__main__' :
    main()