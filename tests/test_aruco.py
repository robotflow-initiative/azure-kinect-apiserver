import os.path as osp

import cv2
import json
import pandas as pd
import tqdm

from azure_kinect_apiserver.common.MulticalCamera import MulticalCameraInfo
from azure_kinect_apiserver.decoder.aruco import ArucoDetectHelper
from azure_kinect_apiserver.common import remove_green_background


# INIT_DIST = np.array(([[0, 0, 0, 0, 0.1]]))
# INIT_DIST = np.array([
#     0.1927826544288516,
#     -0.34972530095573834,
#     0.011612480526787846,
#     -0.00393533140166019,
#     -2.9216752723525734
# ])


# MARKER_LENGTH = 0.02
# MARKER_POINTS = np.array([[-MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
#                           [MARKER_LENGTH / 2, MARKER_LENGTH / 2, 0],
#                           [MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0],
#                           [-MARKER_LENGTH / 2, -MARKER_LENGTH / 2, 0]])


# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
# parameters = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def main(tagged_path: str):
    cam_info = MulticalCameraInfo(tagged_path)

    metadata = json.load(open(osp.join(tagged_path, 'meta.json')))
    camera_name_list = list(metadata['recordings'].keys())
    camera_meta = {
        cam_name: pd.read_csv(osp.join(tagged_path, cam_name, 'meta.csv')) for cam_name in camera_name_list
    }
    color_img_path_collection = {
        cam_name: list(
            map(lambda x: osp.join(tagged_path, cam_name, 'color', str(x) + '.jpg'), cam_meta['basename'].values)
        ) for cam_name, cam_meta in camera_meta.items()
    }
    depth_img_path_collection = {
        cam_name: list(
            map(lambda x: osp.join(tagged_path, cam_name, 'depth', str(x) + '.png'), cam_meta['basename'].values)
        ) for cam_name, cam_meta in camera_meta.items()
    }
    camera_parameter_collection = {
        cam_name: json.load(open(osp.join(tagged_path, cam_name, 'calibration.kinect.json'))) for cam_name in camera_name_list
    }

    for cam_name in camera_name_list:
        color_img_path_list = color_img_path_collection[cam_name]
        depth_img_path_list = depth_img_path_collection[cam_name]
        # camera_parameter = camera_parameter_collection[cam_name]
        # _kinect_distort = camera_parameter['distortion']['color']
        camera_matrix = cam_info.get_intrinsic(cam_name)
        camera_distort = cam_info.get_distort(cam_name)
        # np.array(
        #     [_kinect_distort['k'][0],
        #      _kinect_distort['k'][1],
        #      _kinect_distort['p'][0],
        #      _kinect_distort['p'][1],
        #      _kinect_distort['k'][2],
        #      _kinect_distort['k'][3],
        #      _kinect_distort['k'][4],
        #      _kinect_distort['k'][5]]
        # )

        ctx = ArucoDetectHelper(marker_length=0.0198,
                                aruco_type=cv2.aruco.DICT_4X4_1000,
                                camera_distort=camera_distort,
                                camera_matrix=camera_matrix)

        with tqdm.tqdm(total=len(color_img_path_list)) as pbar:
            for idx, (color_path, depth_path) in enumerate(zip(color_img_path_list, depth_img_path_list)):
                color_frame = cv2.imread(color_path)
                depth_frame = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
                color_frame, mask = remove_green_background(color_frame)
                depth_frame = cv2.bitwise_and(depth_frame, depth_frame, mask=mask)
                res, processed_color_frame, processed_depth_frame, err = ctx.process_one_frame(color_frame[:, :, :3], depth_frame, debug=True)
                if err is None:
                    cv2.imshow('color', processed_color_frame)
                    cv2.waitKey(0)
                    ctx.vis_2d(res, processed_color_frame)
                    ctx.vis_3d(res, processed_color_frame, processed_depth_frame)

                pbar.update()


main(r'C:\Users\robotflow\Desktop\fast-cloth-pose\data\kinect')
