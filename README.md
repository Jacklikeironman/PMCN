# PMCN: Parallax-Motion Collaboration Network for Efficient Stereo Video Dehazing 
This repository is the official PyTorch implementation of PMCN:  Parallax-Motion Collaboration Network for Efficient Stereo Video Dehazing. PMCN ahcieves state-of-the-art performance in the stereo video dehazing (SVD) task.

## Data Preparation
We constructed a new dataset named **Stereo Foggy Video Cityscapes**, which is extended from the [Cityscapes sequences dataset](https://www.cityscapes-dataset.com/). We apply synthetic fog to these clean stereo video pairs as in [Foggy Cityscapes](https://github.com/sakaridis/fog_simulation-SFSU_synthetic/). There are 2973 stereo video clips in the training dataset and 453 stereo video clips in the test dataset. Each video clip contains 7 consecutive frames with the size of 1024 × 2048. Here, we provide the test dataset for performance evaluation. 

[Training & Test Set](https://pan.baidu.com/s/1kCgHJL30DlN3FaS3o9ycFQ)     password: sx6e (278 GB)

## Test
***Coming soon***
## Train
***Coming soon***
## References
[1] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 5

[2] Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei. Deformable convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 764–773, 2017. 1, 2, 5

[3] Christos Sakaridis, Dengxin Dai, and Luc Van Gool. Semantic foggy scene understanding with synthetic data. International Journal of Computer Vision, 126:973–992, 2018. 5

[4] Jingkai Zhou, Varun Jampani, Zhixiong Pi, Qiong Liu, and Ming-Hsuan Yang. Decoupled dynamic filter networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6647–6656, 2021. 2, 3, 4

[5] Achanta R , Shaji A , Smith K , et al. SLIC Superpixels Compared to State-of-the-Art Superpixel Methods[J]. IEEE Transactions on Pattern Analysis & Machine Intelligence, 2012, 34(11):2274-2282.

## License and Acknowledgement
This project is released under the Apache 2.0 license. The deformable convolution layer and the dynamic convolution layer are implemented by [BasicSR](https://github.com/XPixelGroup/BasicSR) and [DDFNet](https://github.com/theFoxofSky/ddfnet). Thanks for their awesome works.
