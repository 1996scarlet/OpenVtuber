import cv2
import numpy as np
import sys


class HeadPoseEstimator:

    def __init__(self, filepath, W, H) -> None:
        # camera matrix
        matrix = np.array([[W, 0, W/2.0],
                           [0, W, H/2.0],
                           [0, 0, 1]])

        # load pre-defined 3d object points and mapping indexes
        obj, index = np.load(filepath, allow_pickle=True)
        obj = obj.T

        def solve_pnp_wrapper(obj, index, matrix):
            def solve_pnp(shape):
                return cv2.solvePnP(obj, shape[index], matrix, None)
            return solve_pnp

        self._solve_pnp = solve_pnp_wrapper(obj, index, matrix)

    def get_head_pose(self, shape):
        if len(shape) != 106:
            raise RuntimeError('Unsupported shape format')

        _, rotation_vec, translation_vec = self._solve_pnp(shape)

        rotation_mat = cv2.Rodrigues(rotation_vec)[0]
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        euler_angle = cv2.decomposeProjectionMatrix(pose_mat)[-1]

        return euler_angle

    @staticmethod
    def draw_axis(img, euler_angle, center, size=80, thickness=3,
                  angle_const=np.pi/180, copy=False):
        if copy:
            img = img.copy()

        euler_angle *= angle_const
        sin_pitch, sin_yaw, sin_roll = np.sin(euler_angle)
        cos_pitch, cos_yaw, cos_roll = np.cos(euler_angle)

        axis = np.array([
            [cos_yaw * cos_roll,
             cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw],
            [-cos_yaw * sin_roll,
             cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll],
            [sin_yaw,
             -cos_yaw * sin_pitch]
        ])

        axis *= size
        axis += center

        axis = axis.astype(np.int)
        tp_center = tuple(center.astype(np.int))

        cv2.line(img, tp_center, tuple(axis[0]), (0, 0, 255), thickness)
        cv2.line(img, tp_center, tuple(axis[1]), (0, 255, 0), thickness)
        cv2.line(img, tp_center, tuple(axis[2]), (255, 0, 0), thickness)

        return img


def main(filename):

    from TFLiteFaceDetector import UltraLightFaceDetecion
    from TFLiteFaceAlignment import CoordinateAlignmentModel

    cap = cv2.VideoCapture(filename)

    fd = UltraLightFaceDetecion("weights/RFB-320.tflite",
                                conf_threshold=0.95)
    fa = CoordinateAlignmentModel("weights/coor_2d106.tflite")
    hp = HeadPoseEstimator("weights/head_pose_object_points.npy",
                           cap.get(3), cap.get(4))

    color = (125, 255, 125)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes, _ = fd.inference(frame)

        for pred in fa.get_landmarks(frame, bboxes):
            for p in np.round(pred).astype(np.int):
                cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)
            face_center = np.mean(pred, axis=0)
            euler_angle = hp.get_head_pose(pred).flatten()
            print(*euler_angle)
            hp.draw_axis(frame, euler_angle, face_center)

        cv2.imshow("result", frame)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == '__main__':
    main(sys.argv[1])
