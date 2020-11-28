import cv2
import numpy as np
import sys
import time


class HeadPoseEstimator:

    def __init__(self, filepath, W, H) -> None:
        _predefined = np.load(filepath, allow_pickle=True)
        self.object_pts, self.r_vec, self.t_vec = _predefined
        self.cam_matrix = np.array([[W, 0, W/2.0],
                                    [0, W, H/2.0],
                                    [0, 0, 1]])

        self.origin_width = 144.76935
        self.origin_height = 139.839

    def get_head_pose(self, shape):
        if len(shape) == 68:
            image_pts = shape
        elif len(shape) == 106:
            image_pts = shape[[
                9, 10, 11, 14, 16, 3, 7, 8, 0,
                24, 23, 19, 32, 30, 27, 26, 25,
                43, 48, 49, 51, 50, 102, 103, 104, 105, 101,
                72, 73, 74, 86, 78, 79, 80, 85, 84,
                35, 41, 42, 39, 37, 36, 89, 95, 96, 93, 91, 90,
                52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55,
                65, 66, 62, 70, 69, 57, 60, 54
            ]]
        else:
            raise RuntimeError('Unsupported shape format')

        # start_time = time.perf_counter()

        ret, rotation_vec, translation_vec = cv2.solvePnP(
            self.object_pts,
            image_pts,
            cameraMatrix=self.cam_matrix,
            distCoeffs=None,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        rear_depth = -150
        front_depth = 0

        left_width = -75
        top_height = -90
        right_width = 75
        bottom_height = 90

        reprojectsrc = np.float32([[left_width, bottom_height, rear_depth],
                                   [right_width, bottom_height, rear_depth],
                                   [right_width, top_height, rear_depth],
                                   [left_width, top_height, rear_depth],
                                   # -------------------------------------
                                   [left_width, bottom_height, front_depth],
                                   [right_width, bottom_height, front_depth],
                                   [right_width, top_height, front_depth],
                                   [left_width, top_height, front_depth]])

        reprojectdst, _ = cv2.projectPoints(reprojectsrc,
                                            rotation_vec,
                                            translation_vec,
                                            self.cam_matrix,
                                            distCoeffs=None)

        # reprojectdst += 16 * np.random.rand(*reprojectdst.shape)
        # end_time = time.perf_counter()
        # print(end_time - start_time)

        # calc euler angle
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        euler_angle = cv2.decomposeProjectionMatrix(pose_mat)[-1]

        return reprojectdst.astype(np.int32), euler_angle

    @staticmethod
    def draw_head_pose_box(src, pts, color=(0, 255, 255), thickness=2, copy=False):
        if copy:
            src = src.copy()

        pts = pts.reshape(-1, 4, 2)

        cv2.polylines(src, pts.reshape(-1, 4, 2), True, color, thickness)

        for i in range(len(pts[0])):
            cv2.line(src, tuple(pts[0][i]), tuple(pts[1][i]), color, thickness)

        return src


def main(filename):

    from TFLiteFaceDetector import UltraLightFaceDetecion
    from TFLiteFaceAlignment import CoordinateAlignmentModel

    cap = cv2.VideoCapture(filename)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 300)

    fd = UltraLightFaceDetecion("pretrained/version-RFB-320_without_postprocessing.tflite",
                            conf_threshold=0.95)
    fa = CoordinateAlignmentModel("pretrained/coor_2d106_face_alignment.tflite")
    hp = HeadPoseEstimator("pretrained/head_pose_object_points.npy",
                        cap.get(3), cap.get(4))

    color = (125, 255, 125)
    counter = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        bboxes, _ = fd.inference(frame)

        for pred in fa.get_landmarks(frame, bboxes):
            # pred += 18 * np.random.rand(*pred.shape)
            # for p in np.round(pred).astype(np.int):
            #     cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)
            reprojectdst, euler_angle = hp.get_head_pose(pred)
            hp.draw_head_pose_box(frame, reprojectdst, thickness=4)

            # cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, (0, 0, 0), thickness=2)
            # cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, (0, 0, 0), thickness=2)
            # cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.75, (0, 0, 0), thickness=2)

        # frame = frame[150:800, 800:1600, :]  # navi
        # frame = frame[0:480, 380:920, :]  # dress
        # frame = frame[190:514, 0:288, :]  # punch
        cv2.imshow("result", frame)
        # cv2.imwrite(f"../videos/head_stable/punch{counter:0>3}.png", frame)
        # cv2.imwrite(f"../videos/head_jitter/punch{counter:0>3}.png", frame)
        counter += 1

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main(sys.argv[1])
