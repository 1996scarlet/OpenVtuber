# coding: utf-8
import os
import numpy as np
import cv2
import time
import collections
import mxnet as mx


pred_type = collections.namedtuple('prediction', ['slice', 'close', 'color'])
pred_types = {'face': pred_type(slice(0, 17), False, (173.91, 198.9, 231.795, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), False, (255., 126.99,  14.025, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), False, (255., 126.99,  14.025, 0.4)),
              'nose': pred_type(slice(27, 31), False, (160,  60.945, 112.965, 0.4)),
              'nostril': pred_type(slice(31, 36), False, (160,  60.945, 112.965, 0.4)),
              'eye1': pred_type(slice(36, 42), True, (151.98, 223.125, 137.955, 0.3)),
              'eye2': pred_type(slice(42, 48), True, (151.98, 223.125, 137.955, 0.3)),
              'lips': pred_type(slice(48, 60), True, (151.98, 223.125, 137.955, 0.3)),
              'teeth': pred_type(slice(60, 68), True, (151.98, 223.125, 137.955, 0.4))}


class MxnetAlignmentorModel:
    def __init__(self, args, verbose=False):
        self.device = args.gpu
        self.ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            args.cab2d_model, 0)

        self.model = mx.mod.Module(sym, context=self.ctx, label_names=None)
        self.model.bind(
            data_shapes=[('data', (1, 3, 128, 128))], for_training=False)
        self.model.set_params(arg_params, aux_params)

    @staticmethod
    def draw_poly(src, landmarks, stroke=1, color=(125, 255, 125)):
        draw = src.copy()
        for pred in pred_types.values():
            le = [landmarks[pred.slice].reshape(-1, 1, 2).astype(np.int32)]
            cv2.polylines(draw, le, pred.close, pred.color, thickness=stroke)

        return draw

    def get_landmarks(self, image, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {numpy.array} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
        """
        # timeb = time.time()
        N = detected_faces.shape[0]
        faces = mx.nd.zeros((N, 3, 128, 128), ctx=self.ctx)
        # offset = mx.nd.zeros((N, 4), dtype=np.float, ctx=self.ctx)
        offset = np.zeros((N, 4), dtype=np.float)

        for i in range(N):
            d = detected_faces[i]
            face = cv2.resize(
                image[d[1]:d[3], d[0]:d[2], :], (128, 128))[..., ::-1]
            faces[i] = mx.nd.array(face.transpose(2, 0, 1))
            offset[i] = (d[2] - d[0]) / 64.0, (d[3] - d[1]) / 64.0, d[0], d[1]

        db = mx.io.DataBatch(data=[faces, ])
        # timec = time.time()

        self.model.forward(db, is_train=False)
        out = self.model.get_outputs()[-1]  # .asnumpy()

        # timed = time.time()x
        landmarks = self._calculate_points(out, offset)
        # timee = time.time()

        # print(f'preprocess: {timec - timeb}')
        # print(f'inferance: {timed - timec}')
        # print(f'calculate: {timee - timed}')
        # print(f'alignment: {timee - timeb}')
        # print('==============>>>>>>>>>>>>>>')

        return landmarks

    def _calculate_points(self, heatmaps, offset):
        """Obtain (x,y) coordinates given a set of N heatmaps. If the center
        and the scale is provided the function will return the points also in
        the original coordinate frame.

        Arguments:
            hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, H, W]

        Keyword Arguments:
            center {torch.tensor} -- the center of the bounding box (default: {None})
            scale {float} -- face scale (default: {None})
        """
        heatmaps.wait_to_read()
        B, N, H, W = heatmaps.shape  # B, 68, 64, 64
        HW = H * W
        heatmaps = heatmaps.asnumpy()
        heatline = heatmaps.reshape(B, N, HW)
        indexes = np.argmax(heatline, axis=2)
        x = indexes % W
        y = indexes // W
        preds = np.stack((x, y), axis=2).astype(np.float)

        for i in range(B):
            preds[i] *= offset[i, :2]
            preds[i] += offset[i, -2:]

        return preds

    def get_eye_roi_slice(self, left_eye_xy, right_eye_xy):
        '''
        Input:
            Position of left eye, position of right eye.
        Output:
            Eye ROI slice
        Usage: 
            left_slice_h, left_slice_w, right_slice_h, right_slice_w = get_eye_roi_slice(lms[0], lms[1])
        '''

        dist_eyes = np.linalg.norm(left_eye_xy - right_eye_xy)

        half_eye_bbox_w = dist_eyes * 0.372
        half_eye_bbox_h = dist_eyes * 0.241

        left_slice_h = slice(int(left_eye_xy[1]-half_eye_bbox_h),
                             int(left_eye_xy[1]+half_eye_bbox_h))
        left_slice_w = slice(int(left_eye_xy[0]-half_eye_bbox_w),
                             int(left_eye_xy[0]+half_eye_bbox_w))

        right_slice_h = slice(int(right_eye_xy[1]-half_eye_bbox_h),
                              int(right_eye_xy[1]+half_eye_bbox_h))
        right_slice_w = slice(int(right_eye_xy[0]-half_eye_bbox_w),
                              int(right_eye_xy[0]+half_eye_bbox_w))

        return left_slice_h, left_slice_w, right_slice_h, right_slice_w
