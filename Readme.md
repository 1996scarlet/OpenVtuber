# OpenVtuber-虚拟アイドル共享计划

<p align="center"><img src="https://s3.ax1x.com/2020/12/12/rVO3FO.gif" /></p>
<p align="center"><img src="https://s3.ax1x.com/2020/12/12/rZeXD0.gif" /></p>

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/1996scarlet/OpenVtuber.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/1996scarlet/OpenVtuber/context:python)
[![License](https://badgen.net/github/license/1996scarlet/OpenVtuber)](LICENSE)
[![ECCV](https://badgen.net/badge/ECCV/2018/red)](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html)

OpenVtuber: An application of real-time face and gaze analyzation via deep nerual networks.

* Lightweight network architecture for low computing capability devices.
* 3D gaze estimation based on the whole face semantic informations.
* The total framework is an upgradation of the [[ECCV 2018]](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html) version.
* Drive MMD models through a single RGB camera.

## Setup

### Requirements

* Python 3.6+
* `pip3 install -r requirements.txt`
* node.js and npm or [yarn](https://classic.yarnpkg.com/en/docs/install/#debian-stable)
* `cd NodeServer && yarn`  # install node modules

### Socket-IO Server

* `cd NodeServer`
* `yarn start`
* Open `http://127.0.0.1:6789/kizuna.html`

### Python Client

* `python3 vtuber_link_start.py <your-video-path>`

## Face Detection

[RetinaFace: Single-stage Dense Face Localisation in the Wild](https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html) of **CVPR 2020**, is a practical single-stage [SOTA](http://shuoyang1213.me/WIDERFACE/WiderFace_Results.html) face detector. It is highly recommended to read the official repo [RetinaFace (mxnet version)](https://github.com/deepinsight/insightface/tree/master/RetinaFace).

However, since the detection target of the face capture system is in the middle-close range, there is no need for complex pyramid scaling. We designed and published [Faster RetinaFace](https://github.com/1996scarlet/faster-mobile-retinaface) to trade off between speed and accuracy, which can reach 500~1000 fps on normal laptops.

| Plan | Inference | Postprocess | Throughput Capacity (FPS)
| --------|-----|--------|---------
| 9750HQ+1660TI | 0.9ms | 1.5ms | 500~1000
| Jetson-Nano | 4.6ms | 11.4ms | 80~200

## Face Alignment

In this project, we apply the facial landmarks for calculating head pose and slicing the eye regions for gaze estimation. Moreover, the mouth and eys status can be inferenced via these key points.

![Emotion](https://s3.ax1x.com/2020/12/13/rm8az6.jpg)

The 2D pre-trained 106 landmarks model is provided by [insightface](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg) repository, based on the coordinate regression face alignment algorithm. We refine this model into TFLite version with lower weights (4.7 MB), which can be found at [here](PythonClient/weights/coor_2d106.tflite). For checking the effectiveness of landmark detection, run the following command at `PythonClient` sub directory:

``` bash
python3 service/TFLiteFaceAlignment.py <your-video-path>
```

## Head Pose Estimation

The Perspective-n-Point (PnP) is the problem of determining the 3D position and orientation (pose) of a camera from observations of known point features.
The PnP is typically formulated and solved linearly by employing [lifting](https://ieeexplore.ieee.org/document/1195992), or [algebraically](https://openaccess.thecvf.com/content_cvpr_2017/html/Ke_An_Efficient_Algebraic_CVPR_2017_paper.html) or [directly](https://ieeexplore.ieee.org/document/6126266).

Briefily, for head pose estimation, a set of pre-defined 3D facial landmarks and the corresponding 2D image projections need to be given. In this project, we employed the eyebrow, eye, nose, mouth and jaw landmarks in the [AIFI Anthropometric Model](https://aifi.isr.uc.pt/Downloads.html) as origin 3D feature points. The pre-defined vectors and mapping proctol can be found at [here](PythonClient/weights/head_pose_object_points.npy).

<p align="center"><img src="docs/images/one.gif" /></p>

We adopt `cv2.SolvePnP` API for calculating the rotation vector and transform vector. Run the following command at `PythonClient` sub directory for real-time head pose estimation:

``` bash
python3 service/SolvePnPHeadPoseEstimation.py <your-video-path>
```

## Iris Localization

Estimating human gaze from a single RGB face image is a challenging task.
Theoretically speaking, the gaze direction can be defined by pupil and eyeball center, however, the latter is unobservable in 2D images. Previous work of [Swook, et al.](https://openaccess.thecvf.com/content_ECCV_2018/html/Seonwook_Park_Deep_Pictorial_Gaze_ECCV_2018_paper.html) presents a method to extract the semantic information of iris and eyeball into the intermediate representation, which so called gazemaps, and then decode the gazemaps into euler angle through regression network.

Inspired by this, we propose a 3D semantic information based gaze estimation method. Instead of employing gazemaps as the intermediate representation, we estimate the center of the eyeball directly from the average geometric information of human gaze.

![rKWPK0.jpg](https://s3.ax1x.com/2020/12/15/rKWPK0.jpg)

Our eye region landmark detection and iris localization models are more robust than the [original implementation](https://github.com/swook/GazeML), which leads to the higher accuracy in more complex situations. The demo of iris localization can be run as follows:

``` bash
python3 service/TFLiteIrisLocalization.py <your-video-path>
```

More details about 3D gaze estimation can be found at the [Laser Eye](https://github.com/1996scarlet/Laser-Eye) repository.

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

@inproceedings{Liu_2018_ECCV,
      author = {Liu, Songtao and Huang, Di and Wang, Yunhong},
      title = {Receptive Field Block Net for Accurate and Fast Object Detection},
      booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
      month = {September},
      year = {2018}
}
```
