# OpenVtuber-虚拟爱抖露共享计划

Kizuna-Ai MMD demo : face capture via single RGB camera

<p align="center"><img src="docs/images/one.gif" /></p>
<p align="center"><img src="docs/images/two.gif" /></p>

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

[RetinaFace: Single-stage Dense Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) of **CVPR 2020**, is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector. It is highly recommended to read the official repo [RetinaFace (mxnet version)](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

However, since the detection target of the face capture system is in the middle-close range, there is no need for complex pyramid scaling. We designed and published [Faster RetinaFace](https://github.com/1996scarlet/faster-mobile-retinaface) to trade off between speed and accuracy, which can reach 500~1000 fps on normal laptops.

| Plan | Inference | Postprocess | Throughput Capacity (FPS)
| --------|-----|--------|---------
| 9750HQ+1660TI | 0.9ms | 1.5ms | 500~1000
| Jetson-Nano | 4.6ms | 11.4ms | 80~200

## 特征点检测（Facial Landmarks Tracking）

The 2D pre-trained model is from the [deep-face-alignment](https://github.com/deepinx/deep-face-alignment) repository.

* Algorithm from [TPAMI 2019](https://arxiv.org/pdf/1808.04803.pdf)
* Training set is based on i-bug 300-W datasets. It's annotation is shown below:<br><br>
![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)

## 头部姿态估计（Head Pose Estimation）

* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)

## 注视估计（Gaze Estimation)

* [Laser Eye : Gaze Estimation via Deep Neural Networks](https://github.com/1996scarlet/Laser-Eye)

## MMD Loader

We apply [Three.js Webgl Loader](https://threejs.org/examples/?q=MMD#webgl_loader_mmd) to  render MMD model on web pages.

## Special Thanks

* [threejs.org](https://threejs.org/)
* [kizunaai.com](http://kizunaai.com/)

## Citation

``` bibtex
@misc{sun2020backbone,
      title={A Backbone Replaceable Fine-tuning Network for Stable Face Alignment},
      author={Xu Sun and Yingjie Guo and Shihong Xia},
      year={2020},
      eprint={2010.09501},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{Bulat2018Hierarchical,
      title={Hierarchical binary CNNs for landmark localization with limited resources},
      author={Bulat, Adrian and Tzimiropoulos, Yorgos},
      journal={IEEE Transactions on Pattern Analysis & Machine Intelligence},
      year={2018},
}

@InProceedings{Deng_2020_CVPR,
      author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
      title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2020}
}
```
