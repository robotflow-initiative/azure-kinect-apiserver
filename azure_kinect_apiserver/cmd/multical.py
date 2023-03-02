"""
This script is used to run multical in docker

ref: https://gist.github.com/davidliyutong/ceb5d469247ff74bc04d313b37682fb6
"""

import glob
import json
import logging
import os
import os.path as osp
import shutil
import subprocess
from typing import Optional, Dict, Tuple

import cv2
import py_cli_interaction

from azure_kinect_apiserver.common import CameraInfo, RSPointCloudHelper, vis_pcds

CONFIG_DOCKER_IMAGE = "davidliyutong/multical-docker"
CONFIG_BOARD_YAML = """
common:
  _type_: 'charuco'
  size: [10, 10]
  aruco_dict: '5X5_1000'
  square_length: 0.040
  marker_length: 0.032
  min_rows: 3
  min_points: 9
boards:
  charuco_10x10_0:
    aruco_offset: 0
  charuco_10x10_1:
    aruco_offset: 50
  charuco_10x10_2:
    aruco_offset: 100
  charuco_10x10_3:
    aruco_offset: 150
  charuco_10x10_4:
    aruco_offset: 200
aruco_params:
  adaptiveThreshWinSizeMax: 200
  adaptiveThreshWinSizeStep: 50
"""


def run_multical_with_docker(tagged_path: str) -> Tuple[Optional[Dict], Optional[Exception]]:
    _logger = logging.getLogger('run_multical_with_docker')
    _logger.debug(f"run multical with image {CONFIG_DOCKER_IMAGE}")

    if not osp.exists(tagged_path):
        return None, FileNotFoundError(f"file {tagged_path} does not exist")

    cameras_name_list = list(
        filter(lambda x: os.path.isdir(osp.join(tagged_path, x)),
               os.listdir(tagged_path)
               )
    )  # ["rxx", "ryy", "rzz"]

    if len(cameras_name_list) <= 0:
        return None, FileNotFoundError(f"no camera found in {tagged_path}")

    _logger.debug(f"found {len(cameras_name_list)} cameras")
    for camera_name in cameras_name_list:
        frame_filenames = glob.glob(osp.join(tagged_path, camera_name, "color", "*"))
        if len(frame_filenames) <= 0:
            _logger.warning(f"no frame found in {camera_name}/color, assuming they are copied to upper directory")
            continue
        for frame_name in frame_filenames:
            shutil.copy(frame_name, osp.join(tagged_path, camera_name))

    _logger.debug(f"write board.yaml")
    with open(osp.join(tagged_path, "boards.yaml"), "w") as f:
        f.write(CONFIG_BOARD_YAML)

    _logger.debug("Add docker volume")
    with open(osp.join(tagged_path, "multical.sh"), 'w') as f:
        f.write(f"multical calibrate --cameras {' '.join(cameras_name_list)}")

    _logger.debug("launch multical")
    p = subprocess.Popen(f"docker run --rm -v {osp.abspath(tagged_path)}:/input {CONFIG_DOCKER_IMAGE}", shell=True)
    try:
        p.wait(timeout=60)  # should have finished in 30 seconds
    except subprocess.TimeoutExpired as _:
        pass

    if osp.exists(osp.join(tagged_path, "calibration.json")):
        with open(osp.join(tagged_path, "calibration.json"), 'r') as f:
            try:
                res = json.load(f)
                return res, None
            except Exception as err:
                return None, err
    else:
        return None, FileNotFoundError(f"calibration.json not found in {tagged_path}")


def examine_multical_result(tagged_path: str):
    cameras_name_list = list(
        filter(lambda x: os.path.isdir(osp.join(tagged_path, x)),
               os.listdir(tagged_path)
               )
    )  # ["rxx", "ryy", "rzz"]
    calibration_json = osp.join(tagged_path, "calibration.json")
    cam_info = CameraInfo(calibration_json)

    color_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(tagged_path, cam_name, 'color', '*.jpg')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in cameras_name_list
    }
    depth_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(tagged_path, cam_name, 'depth', '*.png')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in cameras_name_list
    }
    num_frames = len(color_img_path_collection[cameras_name_list[0]])
    for img_idx in range(num_frames):
        raw_pc_by_camera = {}
        for camera in cameras_name_list:
            color_img_path = color_img_path_collection[camera][img_idx]
            depth_img_path = depth_img_path_collection[camera][img_idx]
            color_img = cv2.imread(color_img_path)
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

            # undistort
            cam_matrix = cam_info.get_intrinsic(camera)
            cam_dist = cam_info.get_distort(camera)
            color_undist = cv2.undistort(color_img, cam_matrix, cam_dist)
            depth_undist = cv2.undistort(depth_img, cam_matrix, cam_dist)

            pc = RSPointCloudHelper(color_undist,
                                    depth_undist,
                                    camera_intrinsic_desc=(cam_info.get_resolution(camera)[0], cam_info.get_resolution(camera)[1], cam_info.get_intrinsic(camera)),
                                    transform=cam_info.get_extrinsic(camera))
            raw_pc_by_camera[camera] = pc
        vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list], fake_color=True)
        vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list])
        sel = py_cli_interaction.must_parse_cli_bool("stop?", default_value=True)
        if sel:
            return


def entry_point(argv):
    logging.basicConfig(level=logging.INFO)
    if len(argv) < 1:
        print("Usage: python -m azure_kinect_apiserver multical <path>")
        return
    else:
        res, ret = run_multical_with_docker(argv[0])
        if ret is not None:
            print(ret)
        else:
            print(res)
            logging.info("examining camera extrinsics")
            examine_multical_result(argv[0])


if __name__ == '__main__':
    import sys

    entry_point(sys.argv[1:])
