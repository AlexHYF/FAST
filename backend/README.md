# Pytorch_Adain_from_scratch
Unofficial Pytorch implementation of [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization [Huang+, ICCV2017]](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.pdf)

Original torch implementation from the author can be found [here](https://github.com/xunhuang1995/AdaIN-style).

Other implementations such as [Pytorch_implementation_using_pretrained_torch_model](https://github.com/irasin/pytorch-AdaIN) or [Chainer_implementation](https://github.com/SerialLain3170/Style-Transfer/tree/master/AdaIN) are also available.I have learned a lot from them and try the pure Pytorch implementation from scratch in this repository.This repository provides a pre-trained model for you to generate your own image given content image and style image. Also, you can download the training dataset or prepare your own dataset to train the model from scratch.

I give a brief qiita blog and you can check it from [here](https://qiita.com/edad811/items/02ca5292276572f9dad8).

If you have any question, please feel free to contact me. (Language in English/Japanese/Chinese will be ok!)

------

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

