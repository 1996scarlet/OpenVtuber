import sys
import os
import argparse
import numpy as np
import mxnet as mx
import cv2
import time
from rcnn.processing.generate_anchor import generate_anchors_fpn, anchors_plane
from rcnn.processing.nms import gpu_nms_wrapper, cpu_nms_wrapper
from rcnn.processing.bbox_transform import nonlinear_pred

STEP = 2
anchor_shape = {
    '32': {'SCALES': (32, 16), 'BASE_SIZE': 16, 'RATIOS': (1., ), 'ALLOWED_BORDER': 9999},
    '16': {'SCALES': (8, 4), 'BASE_SIZE': 16, 'RATIOS': (1., ), 'ALLOWED_BORDER': 9999},
    '8': {'SCALES': (2, 1), 'BASE_SIZE': 16, 'RATIOS': (1., ), 'ALLOWED_BORDER': 9999},
}


class DetectorModel:
    def __init__(self, args, nms=0.4, verbose=False):
        self.threshold = args.threshold
        self.scales = args.scales
        self.nms_threshold = nms
        self.ctx_id = args.gpu
        self.margin = np.array([-args.face_margin, -args.face_margin,
                                args.face_margin, args.face_margin])

        self._feat_stride_fpn = [int(k) for k in anchor_shape.keys()]
        self.anchor_cfg = anchor_shape

        self.fpn_keys = [f'stride{s}' for s in self._feat_stride_fpn]

        anchors_fpn_list = generate_anchors_fpn(cfg=self.anchor_cfg)

        self._anchors_fpn = dict(
            zip(self.fpn_keys, np.asarray(anchors_fpn_list, dtype=np.float32)))

        self._num_anchors = dict(
            zip(self.fpn_keys,
                [anchors.shape[0] for anchors in self._anchors_fpn.values()]))

        if self.ctx_id < 0:
            self.ctx = mx.cpu()
            self.nms = cpu_nms_wrapper(self.nms_threshold)
        else:
            self.ctx = mx.gpu(self.ctx_id)
            self.nms = gpu_nms_wrapper(self.nms_threshold, self.ctx_id)

        sym, arg_params, aux_params = mx.model.load_checkpoint(
            args.retina_model, 0)
        self.model = mx.mod.Module(sym, context=self.ctx, label_names=None)
        self.model.bind(
            data_shapes=[('data', (1, 3, 640, 640))], for_training=False)
        self.model.set_params(arg_params, aux_params)

    def get_center_det(self, frame):
        return self.detect(frame, self.threshold, self.scales)

    def get_margin_det(self, frame):
        det = self.detect(frame, self.threshold, self.scales)
        det = det[:, :4].astype(np.int)
        det += self.margin
        det[det<0] = 0
        return det

    def _retina_forward(self, src, scale):
        ''' ##### Author 1996scarlet@gmail.com
        Image preprocess and return the forward results.

        Parameters
        ----------
        src : ndarray
            The image batch of shape [H, W, C].

        scale : float
            The src scale para.

        Returns
        -------
        net_out: list, len = 3 * N
            Each block has [scores, bbox_deltas, landmarks]

        Usage
        -----
        >>> net_out = self._retina_forward(img, im_scale)
        '''

        def rescale(x, s):
            return src.copy() if scale == 1.0 else cv2.resize(x, None, None, s, s, cv2.INTER_LINEAR)

        def mxtensor(x):
            return mx.nd.array(np.transpose(x[..., ::-1], (2, 0, 1))[np.newaxis, ...])

        def mxbatch(x):
            return mx.io.DataBatch(data=(x, ), provide_data=[('data', x.shape)])

        # timea = time.time()
        dst = rescale(src, scale)
        data = mxtensor(dst)  # CPU cost 0.3ms
        db = mxbatch(data)
        # print(f'preprocess: {time.time() - timea}')

        # timeb = time.time()
        self.model.forward(db, is_train=False)
        out = self.model.get_outputs()
        # print(f'inferance: {time.time() - timeb}')
        return out

    def _retina_detach(self, net_out, index, stride):

        key = f'stride{stride}'
        A, F = self._num_anchors[key], self._anchors_fpn[key]

        [scores, deltas] = map(lambda x: x.asnumpy(), net_out[index:index+2])
        height, width = deltas.shape[2], deltas.shape[3]

        scores = scores[:, A:, :, :].transpose((0, 2, 3, 1)).reshape((-1, 1))
        deltas = deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
        anchors = anchors_plane(height, width, stride, F).reshape((-1, 4))

        return scores, deltas, anchors

    def detect(self, src, threshold=0.5, scales=[1.0]):
        ''' ##### Author 1996scarlet@gmail.com
        Face detection core function.

        Parameters
        ----------
        src : ndarray
            The image batch of shape [H, W, C].

        threshold : float
            Threshold for filter scores.

        scale : list
            The src scale para list.

        Returns
        -------
        det: ndarray
            The bounding box results of shape [N, 5], 
            and each row has (x1, y1, x2, y2, score)


        Usage
        -----
        >>> fd.detect(frame, self.threshold, self.scales)
        # proposals = clip_boxes(proposals, (H, W))
        '''

        def non_maximum_suppression(x):
            return np.zeros((0, 5)) if x.size == 0 else x[self.nms(x), :]

        def horizontal_stack(scores, deltas, anchors, scale):
            mask = scores.ravel() > threshold
            proposals = nonlinear_pred(anchors[mask], deltas[mask]) / scale
            stacked = np.concatenate((proposals, scores[mask]), axis=1)
            return stacked.astype(np.float32, copy=False)

        dets = []

        for scale in scales:
            # =================== FORWARD cost 0.6ms ====================
            # print('=====================>>>>>>>>>>>>>>>>>>')
            # print(f'input shape: {src.shape}, input scale: {scale}')
            net_out = self._retina_forward(src, scale)

            # times = time.time()
            for i, s in enumerate(self._feat_stride_fpn):
                scores, deltas, anchors = self._retina_detach(
                    net_out, i*STEP, s)
                dets.append(horizontal_stack(scores, deltas, anchors, scale))
            # print(f'GPU->CPU calculate scores bboxes: {time.time() - times}')

        # =================== NMS cost 0.12ms ====================
        # timen = time.time()
        res = non_maximum_suppression(np.concatenate(dets))
        # print(f'NMS: {time.time() - timen}')
        return res

    @staticmethod
    def _filter_boxes(boxes, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        keep = np.where((ws >= min_size) & (hs >= min_size))[0]
        return keep

    @staticmethod
    def _filter_boxes2(boxes, max_size, min_size):
        """ Remove all boxes with any side smaller than min_size """
        ws = boxes[:, 2] - boxes[:, 0] + 1
        hs = boxes[:, 3] - boxes[:, 1] + 1
        if max_size > 0:
            keep = np.where(np.minimum(ws, hs) < max_size)[0]
        elif min_size > 0:
            keep = np.where(np.maximum(ws, hs) > min_size)[0]
        return keep
