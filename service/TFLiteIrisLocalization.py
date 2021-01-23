import cv2
import tensorflow as tf
import numpy as np
from functools import partial


class IrisLocalizationModel():

    def __init__(self, filepath):
        # Load the TFLite model and allocate tensors.
        self._interpreter = tf.lite.Interpreter(model_path=filepath)
        self._interpreter.allocate_tensors()

        # model details
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        # inference helper
        self._set_input_tensor = partial(self._interpreter.set_tensor,
                                         input_details[0]["index"])
        self._get_output_tensor = partial(self._interpreter.get_tensor,
                                          output_details[0]["index"])

        self.trans_distance = 32
        self.input_shape = (64, 64)

    def _preprocess(self, img, length, center, name=None):
        """Preprocess the image to meet the model's input requirement.
        Args:
            img: An image in default BGR format.

        Returns:
            image_norm: The normalized image ready to be feeded.
        """

        scale = 23 / length
        cx, cy = self.trans_distance - scale * center

        M = np.array([[scale, 0, cx], [0, scale, cy]])

        resized = cv2.warpAffine(img, M, self.input_shape, borderValue=0.0)

        if name is not None:
            cv2.imshow(name, resized)

        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype(np.float32)
        cv2.normalize(image_norm, image_norm, alpha=-1,
                      beta=1, norm_type=cv2.NORM_MINMAX)

        return image_norm, M

    def get_mesh(self, image, length, center, name=None):
        """Detect the face mesh from the image given.
        Args:
            image: An image in default BGR format.

        Returns:
            mesh: An eyebrow mesh, normalized.
            iris: Iris landmarks.
        """

        # Preprocess the image before sending to the network.
        image, M = self._preprocess(image, length, center, name)

        image = tf.image.convert_image_dtype(image, tf.float32)
        image = image[tf.newaxis, :]

        # The actual detection.
        self._set_input_tensor(image)
        self._interpreter.invoke()

        # Save the results.
        iris = self._get_output_tensor()[0]

        iris = iris.reshape(-1, 3)
        iris[:, 2] = 1

        iM = cv2.invertAffineTransform(M)

        return iris @ iM.T

    @staticmethod
    def draw_pupil(iris, frame, color=(0, 0, 255), thickness=2):
        pupil = iris[0]
        radius = np.linalg.norm(iris[1:] - iris[0], axis=1)

        pupil = pupil.astype(int)
        radius = int(max(radius))

        cv2.circle(frame, tuple(pupil), radius, color, thickness, cv2.LINE_AA)

        return pupil, radius

    @staticmethod
    def draw_eye_markers(landmarks, frame, close=True, color=(0, 255, 255), thickness=2):
        landmarks = landmarks.astype(np.int32)
        cv2.polylines(frame, landmarks, close, color, thickness, cv2.LINE_AA)

    @staticmethod
    def calculate_3d_gaze(poi, scale=256):
        SIN_LEFT_THETA = 2 * np.sin(np.pi / 4)
        SIN_UP_THETA = np.sin(np.pi / 6)

        starts, ends, pupils, centers = poi

        eye_length = np.linalg.norm(starts - ends, axis=1)
        ic_distance = np.linalg.norm(pupils - centers, axis=1)
        zc_distance = np.linalg.norm(pupils - starts, axis=1)

        s0 = (starts[:, 1] - ends[:, 1]) * pupils[:, 0]
        s1 = (starts[:, 0] - ends[:, 0]) * pupils[:, 1]
        s2 = starts[:, 0] * ends[:, 1]
        s3 = starts[:, 1] * ends[:, 0]

        delta_y = (s0 - s1 + s2 - s3) / eye_length / 2
        delta_x = np.sqrt(abs(ic_distance**2 - delta_y**2))

        delta = np.array((delta_x * SIN_LEFT_THETA,
                          delta_y * SIN_UP_THETA))
        delta /= eye_length
        theta, pha = np.arcsin(delta)

        inv_judge = zc_distance**2 - delta_y**2 < eye_length**2 / 4

        delta[0, inv_judge] *= -1
        theta[inv_judge] *= -1
        delta *= scale

        return theta, pha, delta.T


if __name__ == "__main__":
    import sys
    from SolvePnPHeadPoseEstimation import HeadPoseEstimator
    from TFLiteFaceAlignment import CoordinateAlignmentModel
    from TFLiteFaceDetector import UltraLightFaceDetecion

    gpu_ctx = -1
    video = sys.argv[1]
    YAW_THD = 45

    cap = cv2.VideoCapture(video)

    fd = UltraLightFaceDetecion("weights/RFB-320.tflite",
                                conf_threshold=0.9)
    fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")
    hp = HeadPoseEstimator("weights/head_pose_object_points.npy",
                           cap.get(3), cap.get(4))
    gs = IrisLocalizationModel("weights/iris_localization.tflite")

    counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # frame = frame[:480, 380:920, :]  # dress
        # frame = cv2.resize(frame, (960, 1080))

        bboxes, _ = fd.inference(frame)

        for landmarks in fa.get_landmarks(frame, bboxes):
            # calculate head pose
            euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)

            # eye_centers = np.average(eye_markers, axis=1)
            eye_centers = landmarks[[34, 88]]

            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_start = landmarks[[35, 89]]
            eye_end = landmarks[[39, 93]]
            eye_lengths = (eye_end - eye_start)[:, 0]

            pupils = eye_centers.copy()

            if yaw > -YAW_THD:
                iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                pupils[0], _ = gs.draw_pupil(iris_left, frame, thickness=1)

            if yaw < YAW_THD:
                iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                pupils[1], _ = gs.draw_pupil(iris_right, frame, thickness=1)

            poi = landmarks[[35, 89]], landmarks[[39, 93]], pupils, eye_centers

            theta, pha, _ = gs.calculate_3d_gaze(poi)

            # print(theta.mean(), pha.mean())

            gs.draw_eye_markers(eye_markers, frame, thickness=1)

        cv2.imshow('res', frame)
        # cv2.imwrite(f'./asset/orign_dress/img{counter:0>3}.png', frame)

        counter += 1
        if cv2.waitKey(0) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
