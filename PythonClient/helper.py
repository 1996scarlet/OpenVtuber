# coding: utf-8
import math
import cv2
import numpy as np
import argparse
import os
import sys


def draw_coordinate(img, poi, stroke=2):
    '''
    Usage: 
        right_copy = draw_coordinate(right_copy, coordinate[0])
    '''

    for i in range(0, 6, 2):
        img = cv2.circle(img, (poi[i], poi[i+1]), stroke, (255, 255, 0), -1)

    return img





def start_up_init():
    parser = argparse.ArgumentParser(description='Vtuber Online Test')

    # =================== General ARGS ====================
    parser.add_argument('--max_face_number',
                        type=int,
                        help='同时检测的最大人脸数量',
                        default=1)
    parser.add_argument('--face_margin',
                        type=int,
                        help='margin of face',
                        default=8)
    parser.add_argument('--max_frame_rate',
                        type=int,
                        help='Max frame rate',
                        default=30)
    parser.add_argument('--queue_buffer_size',
                        type=int,
                        help='MP Queue size',
                        default=12)
    parser.add_argument('--address_list',
                        type=float,
                        nargs='+',
                        help='IP address of web camera',
                        default=['10.41.0.198', '10.41.0.199'])
    parser.add_argument('--retina_model',
                        default='./weights/M26',
                        help='人脸检测网络预训练模型路径')
    parser.add_argument('--cab2d_model',
                        default='./weights/cab2d',
                        help='face alignment model path')
    parser.add_argument('--gaze_model',
                        default='./weights/iris',
                        help='gaze segmentation model path')
    parser.add_argument('--gpu', default=0, type=int, help='GPU设备ID，-1代表使用CPU')
    parser.add_argument('--threshold',
                        default=.6,
                        type=float,
                        help='RetinaNet的人脸检测阈值')
    parser.add_argument('--scales',
                        type=float,
                        nargs='+',
                        help='RetinaNet的图像缩放系数',
                        default=[.4])

    return parser.parse_args()
