import cv2
import numpy as np
import sys
import time


class HeadPoseEstimator:

    def __init__(self, filepath, W, H) -> None:
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
        self.cam_matrix = np.array([[W, 0, W/2.0],
                                    [0, W, H/2.0],
                                    [0, 0, 1]])

    def get_head_pose(self, shape):
        if len(shape) == 106:
            image_pts = shape[[
                50, 51, 49, 48, 43,
                102, 103, 104, 105, 101,
                35, 41, 40, 42, 39, 37, 33, 36,
                93, 96, 94, 95, 89, 90, 87, 91,
                75, 81,
                84, 85, 80, 79, 78,
                86,
                61, 71, 52, 53,
                0
            ]]
        else:
            raise RuntimeError('Unsupported shape format')

        ret, rotation_vec, translation_vec = cv2.solvePnP(self.object_pts,
                                                          image_pts,
                                                          cameraMatrix=self.cam_matrix,
                                                          distCoeffs=None)

        rotation_mat = cv2.Rodrigues(rotation_vec)[0]
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        euler_angle = cv2.decomposeProjectionMatrix(pose_mat)[-1]

        return euler_angle

    @staticmethod
    def draw_axis(img, euler_angle, center, size=80, angle_const=np.pi/180, copy=False):
        if copy:
            img = img.copy()

        euler_angle *= angle_const
        sin_pitch, sin_yaw, sin_roll = np.sin(euler_angle)
        cos_pitch, cos_yaw, cos_roll = np.cos(euler_angle)

        # X-Axis pointing to right. drawn in red
        x_axis = np.array([
            cos_yaw * cos_roll,
            cos_pitch * sin_roll + cos_roll * sin_pitch * sin_yaw
        ])
        x_axis *= size
        x_axis += center

        # Y-Axis | drawn in green
        #        v
        y_axis = np.array([
            -cos_yaw * sin_roll,
            cos_pitch * cos_roll - sin_pitch * sin_yaw * sin_roll
        ])
        y_axis *= size
        y_axis += center

        # Z-Axis (out of the screen) drawn in blue
        z_axis = np.array([
            sin_yaw,
            -cos_yaw * sin_pitch
        ])
        z_axis *= size
        z_axis += center

        tp_center = tuple(center.astype(int))

        cv2.line(img, tp_center, tuple(x_axis.astype(int)), (0, 0, 255), 3)
        cv2.line(img, tp_center, tuple(y_axis.astype(int)), (0, 255, 0), 3)
        cv2.line(img, tp_center, tuple(z_axis.astype(int)), (255, 0, 0), 3)

        return img


def main(filename):

    from TFLiteFaceDetector import UltraLightFaceDetecion
    from TFLiteFaceAlignment import CoordinateAlignmentModel

    cap = cv2.VideoCapture(filename)

    fd = UltraLightFaceDetecion("pretrained/version-RFB-320_without_postprocessing.tflite",
                                conf_threshold=0.95)
    fa = CoordinateAlignmentModel(
        "pretrained/coor_2d106_face_alignment.tflite")
    hp = HeadPoseEstimator("pretrained/head_pose_object_points.npy",
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
