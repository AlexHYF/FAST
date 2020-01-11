
## Requirements

- Python 3.7
- PyTorch 1.0+
- TorchVision
- Pillow
- Skimage
- tqdm

Anaconda environment recommended here!

(optional)

- GPU environment for training



## Usage

------

## video test

1. Download the pretrained model [here](https://drive.google.com/file/d/1aTS_O3FfLzq5peh20vbWfU4kNAnng6UT/view?usp=sharing)

2. Prepare the content video and the style image.

3. Generate the output video.

``` python
python test -c video.avi -s style.jpg -o result.avi
```

## References

- [X. Huang and S. Belongie. "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.", in ICCV, 2017.](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)
- [Original implementation in Torch](https://github.com/xunhuang1995/AdaIN-style)
- [Pytorch_implementation_using_pretrained_torch_model](https://github.com/irasin/pytorch-AdaIN) 
- [Chainer implementation](https://github.com/SerialLain3170/Style-Transfer/tree/master/AdaIN)

