import numpy as np
import cv2
# import tensorflow as tf
import mxnet as mx
import time
from abc import abstractmethod


class BaseSegmentation:
    def __init__(self, thd=0.05, device='cuda', verbose=False):
        self.thd = thd
        self.device = device
        self.verbose = verbose
        self.shape = np.array((96, 48))  # H, W
        self.eye_center = self.shape / 2  # x, y

    def calculate_gaze_mask_center(self, masks, method='pdc'):
        """ ##### Author 1996scarlet@gmail.com
        Obtain (x,y) coordinates given a set of N gaze heatmaps. 

        Parameters
        ----------
        masks : ndarray
            The predicted heatmaps, of shape [N, H, W, C]

        Returns
        -------
        points : ndarray
            Points of shape [N, 2] -> N * (x, y)

        Usage
        ----------
        >>> points = gs.calculate_gaze_mask_center(masks)
        [[49 23]
        [33 21]]
        >>> eye = cv2.circle(eye, tuple(points[n]), r, color)
        """
        N, H, W, _ = masks.shape

        def probability_density_center(masks, b=1e-7):
            masks[masks < self.thd] = 0
            # masks_esum = np.einsum('ijkl->il', masks)
            masks_sum = np.sum(masks, axis=(1, 2))
            masks_sum += b

            x_sum = np.arange(W) @ np.sum(masks, axis=1)
            y_sum = np.arange(H) @ np.sum(masks, axis=2)

            points = np.hstack((x_sum, y_sum))
            return points/masks_sum

        if method == 'pdc':
            return probability_density_center(masks).astype(np.int32)
        else:
            indexes = np.argmax(masks.reshape((N, -1)), axis=1)
            return np.stack((indexes % W, indexes // W), axis=1)

    def plot_mask(self, src, masks, thd=0.05, alpha=0.8, mono=True):
        draw = src.copy()

        for mask in masks:
            mask = np.repeat((mask > thd)[:, :, :], repeats=3, axis=2)
            if mono:
                draw = np.where(mask, 255, draw)
            else:
                color = np.random.random(3) * 255
                draw = np.where(mask, draw * (1 - alpha) + color * alpha, draw)

        return draw.astype('uint8')

    def funcname(self, parameter_list):
        raise NotImplementedError

    def draw_arrow(self, src, pupil_center, blink=None, lengthen=3, color=(0, 125, 255), stroke=2):

        if blink < 0.35:
            return src

        draw = src.copy()

        H, W, C = draw.shape
        pt3 = self.eye_center + lengthen * (pupil_center - self.eye_center)

        scale = np.array([W, H]) / self.shape
        pt1 = (pt3 * scale).astype(np.int32)
        pt0 = (self.eye_center * scale).astype(np.int32)

        # cv2.drawMarker(draw, tuple(pt1), (255, 255, 0), markerType=cv2.MARKER_CROSS,
        #                              markerSize=2, thickness=2, line_type=cv2.LINE_AA)
        # cv2.drawMarker(draw, tuple(pt0), (255, 200, 200), markerType=cv2.MARKER_CROSS,
        #                              markerSize=2, thickness=2, line_type=cv2.LINE_AA)
        cv2.arrowedLine(draw, tuple(pt0), tuple(pt1), color, stroke)

        return draw

    @abstractmethod
    def _get_gaze_mask_input(self, *eyes):
        pass

    @abstractmethod
    def predict(self, *eyes):
        pass


class TensorflowSegmentationModel(BaseSegmentation):
    def __init__(self, protobuf, thd=0.05, device='cuda', verbose=False):
        BaseSegmentation.__init__(self, thd, device, verbose)

        self.config = tf.compat.v1.ConfigProto()

        if 'cuda' in self.device:
            self.config.gpu_options.allow_growth = True

        self.sess = tf.compat.v1.Session(config=self.config)
        self._load_graph_get_session(protobuf)

        self._x = self.sess.graph.get_tensor_by_name('data:0')
        self._mask = self.sess.graph.get_tensor_by_name('prob/Sigmoid:0')
        self._blink = self.sess.graph.get_tensor_by_name(
            'classlabel_1/BiasAdd:0')

    def _load_graph_get_session(self, protobuf):
        with tf.io.gfile.GFile(protobuf, 'rb') as f:
            G = tf.compat.v1.GraphDef()
            G.ParseFromString(f.read())
            self.sess.graph.as_default()
            tf.import_graph_def(G, name="")

    def _get_gaze_mask_input(self, *eyes):
        """ ##### Author 1996scarlet@gmail.com
        Reduce mean and then stack the input eyes. 

        Parameters
        ----------
        eyes : ndarray
            The image batch of shape [N, H, W, C].

        Returns
        -------
        input_tensor : ndarray
            Numpy ndarray of shape [N, H, W, C]. 
            [?, 48, 96, 3] for this model.

        Usage
        ----------
        >>> input_tensor = get_gaze_mask_input(eye1, eye2, ..., eyeN)
        """

        # assert eyes.shape[-3:] == (48, 96, 3), 'input eyes shape must be [?, 48, 96, 3]'
        x = np.stack([cv2.resize(e, tuple(self.shape)) for e in eyes])
        return (x - x.mean(axis=(0, 1, 2))) / 128

    def predict(self, *eyes):
        """ ##### Author 1996scarlet@gmail.com
        Predict blink classlabels and gaze masks of input eyes. 

        Parameters
        ----------
        eyes : ndarray
            The image batch of shape [N, H, W, C].

        Returns
        -------
        blinks : ndarray
            Numpy ndarray of shape [N, 2]. 

        masks : ndarray
            Numpy ndarray of shape [N, H, W, 1]. 

        Usage
        ----------
        >>> blinks, masks = gs.predict(left_eye, right_eye)
        """
        x = self._get_gaze_mask_input(*eyes)
        return self.sess.run([self._blink, self._mask], {self._x: x})


class MxnetSegmentationModel(BaseSegmentation):
    def __init__(self, args, thd=0.05, verbose=False):
        gpu = args.gpu
        BaseSegmentation.__init__(self, thd, gpu, verbose)

        self.ctx = mx.cpu() if self.device < 0 else mx.gpu(self.device)
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            args.gaze_model, 0)

        self.model = mx.mod.Module(sym, context=self.ctx, label_names=None)
        self.model.bind(
            data_shapes=[('data', (2, 3, 48, 96))], for_training=False)
        self.model.set_params(arg_params, aux_params)

    def _get_gaze_mask_input(self, *eyes):
        """ ##### Author 1996scarlet@gmail.com
        Reduce mean and then stack the input eyes. 

        Parameters
        ----------
        eyes : ndarray
            The image batch of shape [N, H, W, C].

        Returns
        -------
        input_tensor : ndarray
            Mxnet ndarray of shape [N, C, H, W]. 
            [?, 3, 48, 96] for this model.

        Usage
        ----------
        >>> input_tensor = get_gaze_mask_input(eye1, eye2, ..., eyeN)
        """
        x = np.stack([cv2.resize(e, tuple(self.shape)) for e in eyes])
        x = (x - x.mean(axis=(0, 1, 2))) / 128
        return mx.nd.array(x.transpose((0, 3, 1, 2)))

    def predict(self, *eyes):
        """ ##### Author 1996scarlet@gmail.com
        Predict blink classlabels and gaze masks of input eyes. 

        Parameters
        ----------
        eyes : ndarray
            The image batch of shape [N, H, W, C].

        Returns
        -------
        blinks : ndarray
            Numpy ndarray of shape [N, 2 -> (open, close)].

        masks : ndarray
            Numpy ndarray of shape [N, H, W, 1]. 

        points : ndarray
            Numpy ndarray of shape [N, 2 -> (x, y)]. 

        Usage
        ----------
        >>> blinks, masks, points = mxgs.predict(left_eye, right_eye)
        """

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            return np.exp(x) / np.sum(np.exp(x), axis=0)

        x = self._get_gaze_mask_input(*eyes)
        db = mx.io.DataBatch(data=[x, ])

        self.model.forward(db, is_train=False)
        result = self.model.get_outputs()

        # blinks = softmax(result[0].asnumpy())
        #blinks = result[0].asnumpy()
        masks = result[0].asnumpy().transpose((0, 2, 3, 1))
        points = self.calculate_gaze_mask_center(masks)

        # return blinks, masks, points
        return masks, points
