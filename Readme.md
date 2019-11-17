# OpenVtuber-虚拟爱抖露共享计划

## 运行方法（Easy Start）

* `node ./NodeServer/server.js`
* `make -C ./PythonClient/rcnn/`
* `python3.7 ./PythonClient/vtuber_usb_camera.py --gpu -1`


## 人脸检测 （Face Detection）
* [MTCNN (Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks)](https://kpzhang93.github.io/MTCNN_face_detection_alignment/)
* [MTCNN (mxnet version)](https://github.com/deepinsight/insightface)
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

## 表情识别（Emotion Recognition）

- [face_classification](https://github.com/oarriaga/face_classification)
- IMDB gender classification test accuracy: 96%.
- fer2013 emotion classification test accuracy: 66%.

## Live2D

- [插件版本](https://github.com/EYHN/hexo-helper-live2d)
- [打包版本](https://github.com/galnetwen/Live2D)

## 推流框架

- [MJPEG Framework](https://github.com/1996scarlet/MJPEG_Framework)

## FAQ

* Why use RetinaFace ?

    | Methods | LFW | CFP-FP | AgeDB-30
    | --------|-----|--------|---------
    | MTCNN+ArcFace | 99.83 | 98.37 | 98.15
    | RetinaFace+ArcFace | 99.86 | 99.49 | 98.60


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
