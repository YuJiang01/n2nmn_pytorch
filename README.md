# Learning to Reason: End-to-End Module Networks for Visual Question Answering

This repository re-implement https://github.com/ronghanghu/n2nmn in pytorch:

* R. Hu, J. Andreas, M. Rohrbach, T. Darrell, K. Saenko, *Learning to Reason: End-to-End Module Networks for Visual Question Answering*. in ICCV, 2017. ([PDF](https://arxiv.org/pdf/1704.05526.pdf))

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for testing.
(Note, this codebase is still under development. To run it, still need to use part of the original code for data preprocessing)

### Installing

1. Install Python 3 (Anaconda recommended: https://www.continuum.io/downloads).
2. Install Pytorch (http://pytorch.org/)
3. cudnn/v7.0-cuda.9.0 (optional)
```
git clone git@github.com:YuJiang01/n2nmn_pytorch.git
```


### Get preprocessed data
* Follow https://github.com/ronghanghu/n2nmn#download-and-preprocess-the-data preprocess step for CLEVR

After preprocess the data, 



### Training

Example:
```
python train_model/train_clevr_gt_layout.py --exp_name gt_test --model_type gt_layout  --data_dir /private/home/tinayujiang/n2nmn/exp_clevr/data --image_feat_dir /private/home/tinayujiang/n2nmn/exp_clevr/data/vgg_pool5/train --out_dir /private/home/tinayujiang/temp_out
```


