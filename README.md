# PMCN: Parallax Motion Collaboration Network for Efficient Stereo Video Dehazing 
This repository is the official PyTorch implementation of PMCN:  Parallax Motion Collaboration Network for Efficient Stereo Video Dehazing. SwinIR ahcieves state-of-the-art performance in the stereo video dehazing (SVD) task.

> Recently, learning-based stereo dehazing has received more and more attention, but rare studies have focused on the stereo video dehazing (SVD) task. In this paper, we propose a parallax-motion collaboration network (PMCN) to perform haze removal for stereo videos by utilizing joint parallax and motion information. First, we propose a
parallax-motion collaboration block (PMCB) consisting of a parallax interaction module (PIM) and two motion alignment modules (MAM). PIM aims to find short- and long-range binocular parallax correspondences to handle different disparity variations. MAM explores the correlation between temporally adjacent frames to implement precise
motion compensation for an independent view. By cascading multiple PMCBs, parallax and motion information can be fully extracted and integrated for stereo video dehazing. Second, we propose a dynamic residual block (DRB), which enables the network to be content-adaptive to boost restoration performance. In addition, we construct a benchmark dataset, named Stereo Foggy Video Cityscapes dataset, suitable for the SVD task. Quantitative and qualitative results demonstrate that the proposed PMCN outperforms stateof-the-art methods in terms of performance and processing speed.

![img](figs/PMCN.png)

### Dependencies 
- Pytorch >= 1.4.0
- basicsr >= 1.3.4.6 (https://github.com/XPixelGroup/BasicSR)
- ddf == 1.0 (https://github.com/theFoxofSky/ddfnet)

### Data Preparation
we constructed a new dataset named Stereo Foggy Video Cityscapes, which is extended from the [Cityscapes sequences dataset](https://www.cityscapes-dataset.com/). We apply synthetic fog to these clean stereo video pairs as in [Foggy Cityscapes](https://github.com/sakaridis/fog_simulation-SFSU_synthetic/). Here, we provide the testing dataset for the performance evaluation. 

[Test Set](https://github.com/Jacklikeironman/PMCN/edit/main/README.md)     password: aaaa

### Testing

