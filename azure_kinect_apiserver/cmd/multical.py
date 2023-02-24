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


if __name__ == '__main__':
    import sys

    entry_point(sys.argv[1:])
