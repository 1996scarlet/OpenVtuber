# OpenVtuber-虚拟アイドル共享计划

<p align="center"><img src="https://s3.ax1x.com/2020/12/12/rVO3FO.gif" /></p>
<p align="center"><img src="https://s3.ax1x.com/2020/12/12/rZeXD0.gif" /></p>

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/1996scarlet/OpenVtuber.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/1996scarlet/OpenVtuber/context:python)
[![License](https://badgen.net/github/license/1996scarlet/OpenVtuber)](LICENSE)
[![CVPR](https://badgen.net/badge/ECCV/2018/red)](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html)

OpenVtuber: An application of real-time face and gaze analyzation via deep nerual networks.

* Lightweight network architecture for low computing capability devices.
* [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html) 3D gaze estimation based on semantic informations.
* Drive MMD models through a single RGB camera.

## Setup

### Requirements

* Python 3.6+
* `pip3 install -r requirements.txt`
* node.js and npm or [yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)

### Socket-IO Server

* `cd NodeServer`
* `yarn start`

### Python Client

* `cd PythonClient`
* `python3 vtuber_usb_camera.py <your-video-path>`

## Face Detection

[RetinaFace: Single-stage Dense Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) of **CVPR 2020**, is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector. It is highly recommended to read the official repo [RetinaFace (mxnet version)](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

However, since the detection target of the face capture system is in the middle-close range, there is no need for complex pyramid scaling. We designed and published [Faster RetinaFace](https://github.com/1996scarlet/faster-mobile-retinaface) to trade off between speed and accuracy, which can reach 500~1000 fps on normal laptops.

| Plan | Inference | Postprocess | Throughput Capacity (FPS)
| --------|-----|--------|---------
| 9750HQ+1660TI | 0.9ms | 1.5ms | 500~1000
| Jetson-Nano | 4.6ms | 11.4ms | 80~200

## Face Alignment

In this project, we applying facial landmarks for calculating head pose and slicing the eye regions for gaze estimation.

The 2D pre-trained model is provided by [insightface](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg) repository, based on the coordinate regression face alignment algorithm.
The model is trained on 2D 106 landmarks dataset, which annotated as below:

![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG)

## Head Pose Estimation

<p align="center"><img src="docs/images/one.gif" /></p>

* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)

## Iris Localization

3D Gaze Estimation is based on

and the head posed

* [Laser Eye : Gaze Estimation via Deep Neural Networks](https://github.com/1996scarlet/Laser-Eye)

## Kizuna-Ai Demo

face capture via single RGB camera

## Special Thanks

* [threejs.org](https://threejs.org/): Applying Three.js WebGL Loader to render MMD models on web pages.
* [kizunaai.com](http://kizunaai.com/): モデルは無料でご利用いただけます.

## Citation

``` bibtex
@InProceedings{Park_2018_ECCV,
      author = {Park, Seonwook and Spurr, Adrian and Hilliges, Otmar},
      title = {Deep Pictorial Gaze Estimation},
      booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
      month = {September},
      year = {2018}
}
```
