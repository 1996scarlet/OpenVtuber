import cv2
import tensorflow as tf
import numpy as np


class IrisLocalizationModel():

    def __init__(self, filepath):
        # Load the TFLite model and allocate tensors.
        self.interpreter = tf.lite.Interpreter(model_path=filepath)
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

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
        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        self.interpreter.invoke()

        # Save the results.
        iris = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        # iris = self.interpreter.tensor(self.output_details[1]["index"])()

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


if __name__ == "__main__":
    import sys
    from SolvePnPHeadPoseEstimation import HeadPoseEstimator
    from TFLiteFaceAlignment import CoordinateAlignmentModel
    from TFLiteFaceDetector import UltraLightFaceDetecion

    gpu_ctx = -1
    video = sys.argv[1]
    YAW_THD = 45

    cap = cv2.VideoCapture(video)

    fd = UltraLightFaceDetecion("pretrained/version-RFB-320_without_postprocessing.tflite",
                                conf_threshold=0.9)
    fa = CoordinateAlignmentModel(
        "pretrained/coor_2d106_face_alignment.tflite")
    hp = HeadPoseEstimator("pretrained/head_pose_object_points.npy",
                           cap.get(3), cap.get(4))
    gs = IrisLocalizationModel("pretrained/iris_localization.tflite")

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
            _, euler_angle = hp.get_head_pose(landmarks)
            pitch, yaw, roll = euler_angle[:, 0]

            eye_markers = np.take(landmarks, fa.eye_bound, axis=0)

            # eye_centers = np.average(eye_markers, axis=1)
            eye_centers = landmarks[[34, 88]]

            # eye_lengths = np.linalg.norm(landmarks[[39, 93]] - landmarks[[35, 89]], axis=1)
            eye_lengths = (landmarks[[39, 93]] - landmarks[[35, 89]])[:, 0]

            if yaw > -YAW_THD:
                iris_left = gs.get_mesh(frame, eye_lengths[0], eye_centers[0])
                gs.draw_pupil(iris_left, frame, thickness=1)

            if yaw < YAW_THD:
                iris_right = gs.get_mesh(frame, eye_lengths[1], eye_centers[1])
                gs.draw_pupil(iris_right, frame, thickness=1)

            gs.draw_eye_markers(eye_markers, frame, thickness=1)

        cv2.imshow('res', frame)
        # cv2.imwrite(f'./asset/orign_dress/img{counter:0>3}.png', frame)

        counter += 1
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
