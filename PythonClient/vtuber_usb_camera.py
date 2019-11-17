# coding: utf-8
import cv2
import os
import numpy as np
import time
from termcolor import colored
import asyncio
from multiprocessing import Process, Queue
import socketio
from helper import *
from face_detector import DetectorModel
from mxnet_cab_face_alignment import MxnetAlignmentorModel
from gaze_segmentation import MxnetSegmentationModel
import mss


async def upload_loop(url="http://127.0.0.1:6789"):
    # =====================Uploader Setsup========================
    def result_to_json(res):
        landmarks, iris_pois, blinks = res
        return {'shape': landmarks.tolist(),
                'iris': iris_pois.tolist(),
                'blinks': blinks.tolist()}

    sio = socketio.AsyncClient()
    @sio.on('response', namespace='/sakuya')
    async def on_response(data):
        upload_frame = upstream_queue.get()
        # await sio.emit('frame_data', encode_image(upload_frame), namespace='/sakuya')
        await sio.emit('frame_data', 0, namespace='/sakuya')
        try:
            result_string = result_to_json(result_queue.get_nowait())
            await sio.emit('result_data', result_string, namespace='/sakuya')
        except Exception as e:
            print(e)
            pass

    @sio.on('connect', namespace='/sakuya')
    async def on_connect():
        await sio.emit('frame_data', 0, namespace='/sakuya')

    await sio.connect(url)
    await sio.wait()


# async def camera_loop(preload):
#     # =================== FD MODEL ====================
#     reciprocal_of_max_frame_rate = 1/preload.max_frame_rate
#     loop = asyncio.get_running_loop()

#     camera = cv2.VideoCapture(2)
#     # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
#     # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

#     while True:
#         start_time = loop.time()

#         ret, frame = camera.read()

#         if not ret:
#             break

#         frame_queue.put(frame)

#         restime = reciprocal_of_max_frame_rate - loop.time() + start_time
#         if restime > 0:
#             await asyncio.sleep(restime)


async def detection_loop(preload):
    loop = asyncio.get_running_loop()
    deal_latency = 1/preload.max_frame_rate

    # =================== CAMERA ====================
    # camera = cv2.VideoCapture(2)
    camera = cv2.VideoCapture('./Temp/obama.mp4')
    # camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    # camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    # =================== MODELS ====================
    fd = DetectorModel(preload)
    fa = MxnetAlignmentorModel(preload)
    gs = MxnetSegmentationModel(preload)

    # =================== ETERNAL LOOP ====================
    while True:
        start_time = loop.time()
        ret, head_frame = camera.read()

        if not ret:
            break

        det = fd.get_margin_det(head_frame)

        if len(det) > 0:
            preds = fa.get_landmarks(head_frame, detected_faces=det[:1])
            landmarks = preds[0]

            left_poi = (landmarks[36:42].sum(axis=0) / 6)
            right_poi = (landmarks[42:48].sum(axis=0) / 6)

            # eye ROI slice
            left_slice_h, left_slice_w, right_slice_h, right_slice_w = \
                fa.get_eye_roi_slice(left_poi, right_poi)

            # eye region of interest
            left_eye_im = head_frame[left_slice_h, left_slice_w, :]
            right_eye_im = head_frame[right_slice_h, right_slice_w, :]

            LH, LW = left_eye_im.shape[0:2]
            RH, RW = right_eye_im.shape[0:2]

            if LH < 5 or LW < 5 or RH < 5 or RW < 5:
                continue

            _, iris_pois = gs.predict(left_eye_im, right_eye_im)

            draw = fa.draw_poly(head_frame, landmarks, stroke=1)
            eyes = [draw[left_slice_h, left_slice_w, :],
                    draw[right_slice_h, right_slice_w, :]]

            real_weight_left = (landmarks[39] - landmarks[36])[0]
            real_weight_right = (landmarks[45] - landmarks[42])[0]

            real_height_left = (landmarks[40:42, 1] - landmarks[37:39, 1]).sum()
            real_height_right = (landmarks[46:48, 1] - landmarks[43:45, 1]).sum()

            blinks = np.array([real_height_left/real_weight_left,
                               real_height_right/real_weight_right])

            real_left = real_weight_left * 48 / LW
            real_right = real_weight_right * 48 / RW

            results_eyes = [gs.draw_arrow(cv2.resize(eye, (360, 180)), point, blink=blink, stroke=2)
                            for eye, point, blink in zip(eyes, iris_pois, blinks)]

            # results_eyes = [cv2.resize(eye, (360, 180)) for eye in eyes]
            result_roi = np.concatenate(results_eyes, axis=1)

            cv2.imshow('result', result_roi)
            cv2.waitKey(1)

            iris_pois -= np.array([48, 24])
            delta = iris_pois / np.array([real_left, real_right])

            upstream_queue.put(head_frame)
            result_queue.put((landmarks, delta, blinks))
        # print(colored(loop.time()-start_time, 'red'), flush=True)

        restime = deal_latency - loop.time() + start_time
        if restime > 0:
            await asyncio.sleep(restime)


# =================== ARGS ====================
# os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
args = start_up_init()

# =================== INIT ====================
# frame_queue = Queue(maxsize=args.queue_buffer_size)
upstream_queue = Queue(maxsize=args.queue_buffer_size)
result_queue = Queue(maxsize=args.queue_buffer_size)

# =================== Process On ====================
Process(target=lambda: asyncio.run(detection_loop(args))).start()
# Process(target=lambda: asyncio.run(camera_loop(args))).start()
asyncio.run(upload_loop())
