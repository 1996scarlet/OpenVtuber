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

In this project, we apply the facial landmarks for calculating head pose and slicing the eye regions for gaze estimation. Moreover, the mouth and eys status can be inferenced via these key points.

![rm8az6.jpg](https://s3.ax1x.com/2020/12/13/rm8az6.jpg)

The 2D pre-trained 106 landmarks model is provided by [insightface](https://github.com/deepinsight/insightface/tree/master/alignment/coordinateReg) repository, based on the coordinate regression face alignment algorithm. We refine this model into TFLite version with lower weights (4.7 MB), which can be found at [here](PythonClient/pretrained/coor_2d106_face_alignment.tflite). For checking the effectiveness of landmark detection, run the following command at `PythonClient` sub directory:

``` bash
python3 TFLiteFaceAlignment.py <your-video-path>
```

## Head Pose Estimation

The model is trained on 2D 106 landmarks dataset, which annotated as below:

<!-- ![ibug](https://cloud.githubusercontent.com/assets/16308037/24229391/1910e9cc-0fb4-11e7-987b-0fecce2c829e.JPG) -->

<!-- <p align="center"><img src="docs/images/one.gif" /></p> -->

* [head-pose-estimation](https://github.com/lincolnhard/head-pose-estimation)

```
        self.object_pts = np.float32([
            [1.330353, 7.122144, 6.903745],  # 29
            [2.533424, 7.878085, 7.451034],
            [4.861131, 7.878672, 6.601275],
            [6.137002, 7.271266, 5.200823],
            [6.825897, 6.760612, 4.402142],
            [-1.330353, 7.122144, 6.903745],  # 34
            [-2.533424, 7.878085, 7.451034],
            [-4.861131, 7.878672, 6.601275],
            [-6.137002, 7.271266, 5.200823],
            [-6.825897, 6.760612, 4.402142],
            [5.311432, 5.485328, 3.987654],  # 13
            [4.461908, 6.189018, 5.594410],
            [3.550622, 6.185143, 5.712299],
            [2.542231, 5.862829, 4.687939],
            [1.789930, 5.393625, 4.413414],
            [2.693583, 5.018237, 5.072837],
            [3.530191, 4.981603, 4.937805],
            [4.490323, 5.186498, 4.694397],
            [-5.311432, 5.485328, 3.987654],  # 21
            [-4.461908, 6.189018, 5.594410],
            [-3.550622, 6.185143, 5.712299],
            [-2.542231, 5.862829, 4.687939],
            [-1.789930, 5.393625, 4.413414],
            [-2.693583, 5.018237, 5.072837],
            [-3.530191, 4.981603, 4.937805],
            [-4.490323, 5.186498, 4.694397],
            [0.981972, 4.554081, 6.301271],  # 57
            [-0.981972, 4.554081, 6.301271],  # 47
            [-1.930245, 0.424351, 5.914376],  # 50
            [-0.746313, 0.348381, 6.263227],
            [0.000000, 0.000000, 6.763430],  # 52
            [0.746313, 0.348381, 6.263227],
            [1.930245, 0.424351, 5.914376],  # 54
            [0.000000, 1.916389, 7.700000],  # nose tip
            [-2.774015, -2.080775, 5.048531],  # 39
            [0.000000, -1.646444, 6.704956],  # 41
            [2.774015, -2.080775, 5.048531],  # 43
            [0.000000, -3.116408, 6.097667],  # 45
            [0.000000, -7.415691, 4.070434],
        ])

        self.mapping = [
            50, 51, 49, 48, 43,  # eyebrow
            102, 103, 104, 105, 101,  # eyebrow
            35, 41, 40, 42, 39, 37, 33, 36,  # eye
            93, 96, 94, 95, 89, 90, 87, 91,  # eye
            75, 81,  # nose
            84, 85, 80, 79, 78,  # nose
            86,  # nose tip
            61, 71, 52, 53,  # mouth
            0  # jaw
        ]
```

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
