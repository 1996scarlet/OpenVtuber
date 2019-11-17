# OpenVtuber-虚拟爱抖露共享计划

## Kizuna-Ai MMD demo : face capture via single RGB camera

![kizuna1](docs/images/one.gif)
![kizuna2](docs/images/two.gif)


## Installation
### Requirements

* Python 3.5+
* Linux, Windows or macOS
* mxnet (>=1.4)
* node.js and npm or yarn

While not required, for optimal performance(especially for the detector) it is highly recommended to run the code using a CUDA enabled GPU.

### Run

* `node ./NodeServer/server.js`
* `make -C ./PythonClient/rcnn/`
* `python3.7 ./PythonClient/vtuber_usb_camera.py --gpu -1`

## 人脸检测 （Face Detection）
* [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641)
* [RetinaFace (mxnet version)](https://github.com/deepinsight/insightface/tree/master/RetinaFace)

RetinaFace is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector which is initially described in [arXiv technical report](https://arxiv.org/abs/1905.00641)

![demoimg1](https://github.com/deepinsight/insightface/blob/master/resources/11513D05.jpg)

![demoimg2](https://github.com/deepinsight/insightface/blob/master/resources/widerfacevaltest.png)

## 头部姿态估计（Head Pose Estimation）
* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)

## 特征点检测（Facial Landmarks Tracking）
The 2D pre-trained model is from the [deep-face-alignment](https://github.com/deepinx/deep-face-alignment) repository.
* Algorithm from [TPAMI 2019](https://arxiv.org/pdf/1808.04803.pdf)
* Training set is based on i-bug 300-W datasets. It's annotation is shown below:<br><br>
![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)

## 注视估计（Gaze Estimation)

- [Laser Eye : Gaze Estimation via Deep Neural Networks](https://github.com/1996scarlet/Laser-Eye)

## MMD Loader

- [Three.js Webgl Loader](https://threejs.org/examples/?q=MMD#webgl_loader_mmd)

## Live2D

- [插件版本](https://github.com/EYHN/hexo-helper-live2d)
- [打包版本](https://github.com/galnetwen/Live2D)

## Thanks

- [threejs.org](https://threejs.org/)
- [kizunaai.com](http://kizunaai.com/)

## Citation

```
@article{Bulat2018Hierarchical,
  title={Hierarchical binary CNNs for landmark localization with limited resources},
  author={Bulat, Adrian and Tzimiropoulos, Yorgos},
  journal={IEEE Transactions on Pattern Analysis & Machine Intelligence},
  year={2018},
}
  
@inproceedings{deng2019retinaface,
title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
booktitle={arxiv},
year={2019}
}
```
