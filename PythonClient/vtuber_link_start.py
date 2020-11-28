# coding: utf-8
from TFLiteFaceDetector import UltraLightFaceDetecion
from TFLiteFaceAlignment import CoordinateAlignmentModel
from TFLiteIrisLocalization import IrisLocalizationModel
from SolvePnPHeadPoseEstimation import HeadPoseEstimator
from threading import Thread
import cv2
import sys
import numpy as np
from queue import Queue
import time
import socketio

cap = cv2.VideoCapture(sys.argv[1])
fd = UltraLightFaceDetecion("pretrained/version-RFB-320_without_postprocessing.tflite",
                            conf_threshold=0.95)
fa = CoordinateAlignmentModel("pretrained/coor_2d106_face_alignment.tflite")
hp = HeadPoseEstimator("pretrained/head_pose_object_points.npy",
                       cap.get(3), cap.get(4))
gs = IrisLocalizationModel("pretrained/iris_localization.tflite")

QUEUE_BUFFER_SIZE = 18
RUNNING_FLAG = True

box_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
landmark_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
iris_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)
upstream_queue = Queue(maxsize=QUEUE_BUFFER_SIZE)

# ======================================================


def encode_image(image, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    return cv2.imencode('.jpg', image, encode_param)[1].tostring()


sio = socketio.Client()


@sio.on('connect', namespace='/kizuna')
def on_connect():
    sio.emit('result_data', 0, namespace='/kizuna')


sio.connect("http://127.0.0.1:6789")

# ======================================================


def face_detection():
    while RUNNING_FLAG:
        ret, frame = cap.read()

        if not ret:
            break

        face_boxes, _ = fd.inference(frame)
        box_queue.put((frame, face_boxes))


def face_alignment():
    while RUNNING_FLAG:
        frame, boxes = box_queue.get()
        landmarks = fa.get_landmarks(frame, boxes)
        landmark_queue.put((frame, landmarks))


def iris_localization(YAW_THD=45, thickness=1):
    while RUNNING_FLAG:
        frame, preds = landmark_queue.get()
        preds = list(preds)

        for landmarks in preds:
            # calculate head pose
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            print(pitch)

            if roll < 0:
                roll = -180 - roll
            else:
                roll = 180 - roll

            eye_centers = landmarks[[34, 88]]
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

            # start_time = time.perf_counter()

            if yaw > -YAW_THD:
                iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                gs.draw_pupil(iris_left, frame, thickness=thickness)

            # print(time.perf_counter()-start_time)

            if yaw < YAW_THD:
                iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                gs.draw_pupil(iris_right, frame, thickness=thickness)

            result_string = {'euler': (pitch, yaw, roll)}
            sio.emit('result_data', result_string, namespace='/kizuna')

        upstream_queue.put((frame, preds))


def draw(color=(125, 255, 0), thickness=2):
    while RUNNING_FLAG:
        frame, landmarks = upstream_queue.get()

        # for pred in landmarks:
        #     for p in np.round(pred).astype(np.int):
        #         cv2.circle(frame, tuple(p), 1, color, thickness, cv2.LINE_AA)

        cv2.imshow('result', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


draw_thread = Thread(target=draw)
draw_thread.start()

iris_thread = Thread(target=iris_localization)
iris_thread.start()

alignment_thread = Thread(target=face_alignment)
alignment_thread.start()

# detection_thread = Thread(target=face_detection)
# detection_thread.start()

face_detection()
cap.release()
