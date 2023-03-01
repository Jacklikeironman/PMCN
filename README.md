# PMCN: Parallax Motion Collaboration Network for Efficient Stereo Video Dehazing 
This repository is the official PyTorch implementation of PMCN:  Parallax Motion Collaboration Network for Efficient Stereo Video Dehazing. PMCN ahcieves state-of-the-art performance in the stereo video dehazing (SVD) task.

> Recently, learning-based stereo dehazing has received more and more attention, but rare studies have focused on the stereo video dehazing (SVD) task. In this paper, we propose a parallax-motion collaboration network (PMCN) to perform haze removal for stereo videos by utilizing joint parallax and motion information. First, we propose a
parallax-motion collaboration block (PMCB) consisting of a parallax interaction module (PIM) and two motion alignment modules (MAM). PIM aims to find short- and long-range binocular parallax correspondences to handle different disparity variations. MAM explores the correlation between temporally adjacent frames to implement precise
motion compensation for an independent view. By cascading multiple PMCBs, parallax and motion information can be fully extracted and integrated for stereo video dehazing. Second, we propose a dynamic residual block (DRB), which enables the network to be content-adaptive to boost restoration performance. In addition, we construct a benchmark dataset, named Stereo Foggy Video Cityscapes dataset, suitable for the SVD task. Quantitative and qualitative results demonstrate that the proposed PMCN outperforms stateof-the-art methods in terms of performance and processing speed.

![img](figs/PMCN.png)

### Dependencies 
- Pytorch >= 1.4.0
- basicsr >= 1.3.4.6 (https://github.com/XPixelGroup/BasicSR)
- ddf == 1.0 (https://github.com/theFoxofSky/ddfnet)

### Data Preparation
We constructed a new dataset named Stereo Foggy Video Cityscapes, which is extended from the [Cityscapes sequences dataset](https://www.cityscapes-dataset.com/). We apply synthetic fog to these clean stereo video pairs as in [Foggy Cityscapes](https://github.com/sakaridis/fog_simulation-SFSU_synthetic/). Here, we provide the testing dataset for performance evaluation. 

[Test Set](https://pan.baidu.com/s/1qFheJIZvQBbB-NBjnqwRcw)     password: n7of

### Test
Firstly, we download the pretrained models and save them to the folder `checkpoints`.
- [Google Driver](https://drive.google.com/drive/folders/1Q9KCSO8Tn593PC2kPNBS0Wc-DzwSJYRK?usp=sharing)
- [百度云盘](https://pan.baidu.com/s/11RkA8476AOeOoPkcy8_d_Q) password: 82qr

You can test different fog intensity by changing the `--fi` (fog intensity) in command line.
```
## do not need output
python test.py -g 0 evaluate --fi 0.005 
python test.py -g 0 evaluate --fi 0.01 
python test.py -g 0 evaluate --fi 0.02

## need output
python test.py -g 0 evaluate --fi 0.005 -o
python test.py -g 0 evaluate --fi 0.01 -o
python test.py -g 0 evaluate --fi 0.02 -o
```
The results should be the same as that in `logs_fi/test.log`.
### Train
Coming soon
### References
