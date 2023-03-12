"""
This script is used to run multical in docker

ref: https://gist.github.com/davidliyutong/ceb5d469247ff74bc04d313b37682fb6
"""

import glob
import io
import json
import logging
import os
import os.path as osp
import shutil
import subprocess
from typing import Optional, Dict, Tuple, List, Iterable

import cv2
import numpy as np
import open3d as o3d
import plyer
import py_cli_interaction
import yaml

from azure_kinect_apiserver.common import (
    MulticalCameraInfo,
    PointCloudHelper,
    vis_pcds,
    save_pcds,
    point_cloud_registration_fine,
    colored_point_cloud_registration_robust,
    get_workspace_limit_by_interaction,
    merge_point_cloud_helpers,
    get_trans_mat_by_nws_combined,
)
from azure_kinect_apiserver.decoder import ArucoDetectHelper

logger = logging.getLogger("azure_kinect_apiserver.cmd.multical")

CONFIG_DOCKER_IMAGE_REF = "davidliyutong/multical-docker"
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
    _logger.debug(f"run multical with image {CONFIG_DOCKER_IMAGE_REF}")

    if not osp.exists(tagged_path):
        return None, FileNotFoundError(f"file {tagged_path} does not exist")

    cameras_name_list = list(
        filter(
            lambda x: os.path.isdir(osp.join(tagged_path, x)),
            filter(
                lambda x: x != 'tmp', os.listdir(tagged_path)
            ),
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
    p = subprocess.Popen(f"docker run --rm -v {osp.abspath(tagged_path)}:/input {CONFIG_DOCKER_IMAGE_REF}", shell=True)
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


def generate_multicam_pc(tagged_path: str, debug: bool = False) -> Iterable[o3d.geometry.PointCloud]:
    cameras_name_list = list(
        filter(
            lambda x: os.path.isdir(osp.join(tagged_path, x)),
            filter(
                lambda x: x != 'tmp', os.listdir(tagged_path)
            ),
        )
    )  # ["rxx", "ryy", "rzz"]
    calibration_json = osp.join(tagged_path, "calibration.json")
    cam_info = MulticalCameraInfo(calibration_json)

    color_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(tagged_path, cam_name, 'color', '*.jpg')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in cameras_name_list
    }
    depth_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(tagged_path, cam_name, 'depth', '*.png')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in cameras_name_list
    }
    num_frames = len(color_img_path_collection[cameras_name_list[0]])
    for img_idx in range(num_frames):
        raw_pc_by_camera = {}
        for cam_name in cameras_name_list:
            color_img_path = color_img_path_collection[cam_name][img_idx]
            depth_img_path = depth_img_path_collection[cam_name][img_idx]
            color_img = cv2.imread(color_img_path)
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

            # undistort
            cam_matrix = cam_info.get_intrinsic(cam_name)
            cam_dist = cam_info.get_distort(cam_name)
            color_undistort = cv2.undistort(color_img, cam_matrix, cam_dist)
            depth_undistort = cv2.undistort(depth_img, cam_matrix, cam_dist)

            pc = PointCloudHelper(color_undistort,
                                  depth_undistort,
                                  camera_intrinsic_desc=(cam_info.get_resolution(cam_name)[0], cam_info.get_resolution(cam_name)[1], cam_info.get_intrinsic(cam_name)),
                                  transform=cam_info.get_extrinsic(cam_name))
            raw_pc_by_camera[cam_name] = pc
        if debug:
            vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list], fake_color=True)
            vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list])

        merged_pc = merge_point_cloud_helpers(raw_pc_by_camera.values())

        yield merged_pc


def refine_multical_result(tagged_path: str, start_idx: int = 0):
    board_cfg = yaml.load(io.StringIO(CONFIG_BOARD_YAML), Loader=yaml.SafeLoader)
    marker_length = board_cfg["common"]["marker_length"]
    aruco_type = getattr(cv2.aruco, "DICT_" + board_cfg["common"]["aruco_dict"])

    cameras_name_list = list(
        filter(
            lambda x: os.path.isdir(osp.join(tagged_path, x)),
            filter(
                lambda x: x != 'tmp', os.listdir(tagged_path)
            ),
        )
    )  # ["rxx", "ryy", "rzz"]
    calibration_json = osp.join(tagged_path, "calibration.json")
    cam_info = MulticalCameraInfo(calibration_json)
    master_cam = cam_info.master_cam

    color_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(tagged_path, cam_name, 'color', '*.jpg')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in cameras_name_list
    }
    depth_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(tagged_path, cam_name, 'depth', '*.png')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in cameras_name_list
    }
    num_frames = len(color_img_path_collection[cameras_name_list[0]])

    aruco_ctx_collection = {
        cam_name: ArucoDetectHelper(
            marker_length=marker_length,
            aruco_type=aruco_type,
            camera_distort=cam_info.get_distort(cam_name),
            camera_matrix=cam_info.get_intrinsic(cam_name)
        ) for cam_name in cameras_name_list
    }

    for img_idx in range(num_frames):
        if img_idx < start_idx:
            continue
        raw_pc_by_camera = {}
        sliced_pc_by_camera = {}

        for cam_name in cameras_name_list:
            color_img_path = color_img_path_collection[cam_name][img_idx]
            depth_img_path = depth_img_path_collection[cam_name][img_idx]
            color_img = cv2.imread(color_img_path)
            depth_img = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)

            # undistort
            cam_matrix = cam_info.get_intrinsic(cam_name)
            cam_dist = cam_info.get_distort(cam_name)
            color_undistort = cv2.undistort(color_img, cam_matrix, cam_dist)
            depth_undistort = cv2.undistort(depth_img, cam_matrix, cam_dist)

            # aruco masking
            # gray = cv2.cvtColor(color_undistort, cv2.COLOR_BGR2GRAY)
            # corners, ids, rejectedImgPoints = aruco_ctx_collection[cam_name].detector.detectMarkers(gray)
            # color_undistort_masked = aruco_ctx_collection[cam_name].apply_polygon_mask_color(color_undistort, corners)
            # depth_undistort_masked = aruco_ctx_collection[cam_name].apply_polygon_mask_depth(depth_undistort, corners)
            # aruco_pc = PointCloudHelper(color_undistort_masked,
            #                             depth_undistort_masked,
            #                             camera_intrinsic_desc=(cam_info.get_resolution(cam_name)[0], cam_info.get_resolution(cam_name)[1], cam_info.get_intrinsic(cam_name)),
            #                             transform=cam_info.get_extrinsic(cam_name))
            # aruco_pc_by_camera[cam_name] = aruco_pc

            raw_pc = PointCloudHelper(color_undistort,
                                      depth_undistort,
                                      camera_intrinsic_desc=(cam_info.get_resolution(cam_name)[0], cam_info.get_resolution(cam_name)[1], cam_info.get_intrinsic(cam_name)),
                                      transform=cam_info.get_extrinsic(cam_name))
            raw_pc_by_camera[cam_name] = raw_pc

            sliced_pc_by_camera[cam_name] = raw_pc

        os.makedirs(osp.join(tagged_path, 'tmp'), exist_ok=True)
        [save_pcds([raw_pc_by_camera[cam_name].pcd], osp.join(tagged_path, 'tmp'), cam_name) for cam_name in cameras_name_list]
        print("saved to {}".format(osp.join(tagged_path, 'tmp')))
        print("edit the point cloud, remove background then press enter")
        try:
            input(">")
        except KeyboardInterrupt:
            shutil.rmtree(osp.join(tagged_path, 'tmp'))
            return
        sliced_pc_by_camera = {cam_name: o3d.io.read_point_cloud(osp.join(tagged_path, 'tmp', cam_name + '.ply')) for cam_name in cameras_name_list}

        refined_transformation_by_camera = {
            cam_name: colored_point_cloud_registration_robust(
                sliced_pc_by_camera[cam_name],
                sliced_pc_by_camera[master_cam], debug=True) for cam_name in cameras_name_list if cam_name != master_cam
        }
        # shutil.move(osp.join(tagged_path, 'tmp'), osp.join(tagged_path, '.tmp'))

        vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list], fake_color=True)
        vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list])
        [x.pcd.transform(refined_transformation_by_camera[cam_name]) for cam_name, x in raw_pc_by_camera.items() if cam_name != master_cam]
        vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list], fake_color=True)
        vis_pcds([raw_pc_by_camera[camera].pcd for camera in cameras_name_list])

        refined_transformation_by_camera[master_cam] = np.eye(4)
        logging.info("refined extrinsic: {}".format(refined_transformation_by_camera))

        sel = py_cli_interaction.must_parse_cli_bool("save?", default_value=False)
        if sel:
            patch_refine_extrinsic(tagged_path, refined_transformation_by_camera)
            return
        else:
            abort = py_cli_interaction.must_parse_cli_bool("abort?", default_value=False)
            if abort:
                patch_refine_extrinsic(tagged_path, {cam_name: np.eye(4) for cam_name in cameras_name_list})
                return


def patch_refine_extrinsic(tagged_path: str, refine_extrinsic: Dict[str, np.ndarray]):
    calibration_info: dict
    with open(osp.join(tagged_path, "calibration.json")) as f:
        calibration_info = json.load(f)
        calibration_info["icp_extrinsic_refinement"] = {
            k: v.tolist() for k, v in refine_extrinsic.items()
        }

    with open(osp.join(tagged_path, "calibration.json"), "w") as f:
        json.dump(calibration_info, f, indent=4)


def patch_workspace_limits(tagged_path: str, limits_nws: List[np.ndarray]):
    calibration_info: dict
    assert len(limits_nws) == 4
    with open(osp.join(tagged_path, "calibration.json")) as f:
        calibration_info = json.load(f)
        points = np.concatenate([limits_nws], axis=1)
        x_lim = (float(points[:, 0].min()), float(points[:, 0].max()))
        y_lim = (float(points[:, 1].min()), float(points[:, 1].max()))
        z_lim = (float(points[:, 2].min()), float(points[:, 2].max()))
        calibration_info["workspace_limits"] = {
            "base": limits_nws[0].tolist(),
            "north": limits_nws[1].tolist(),
            "west": limits_nws[2].tolist(),
            "sky": limits_nws[3].tolist(),
            "x_lim": x_lim,
            "y_lim": y_lim,
            "z_lim": z_lim,
        }

    with open(osp.join(tagged_path, "calibration.json"), "w") as f:
        json.dump(calibration_info, f, indent=4)


def patch_rot_trans(tagged_path: str, rot: np.ndarray, trans: np.ndarray):
    calibration_info: dict
    with open(osp.join(tagged_path, "calibration.json")) as f:
        calibration_info = json.load(f)
        calibration_info["realworld_rot_trans"] = {
            "R": rot.tolist(),
            "T": trans.tolist(),
        }

    with open(osp.join(tagged_path, "calibration.json"), "w") as f:
        json.dump(calibration_info, f, indent=4)


def main(multical_path: str):
    cameras_name_list = list(
        filter(
            lambda x: os.path.isdir(osp.join(multical_path, x)),
            filter(
                lambda x: x != 'tmp', os.listdir(multical_path)
            ),
        )
    )

    if osp.exists(osp.join(multical_path, "calibration.json")):
        run_multical = py_cli_interaction.must_parse_cli_bool("run multical?", default_value=False)
    else:
        run_multical = True
    if run_multical:
        res, err = run_multical_with_docker(multical_path)
        if err is not None:
            logger.error(f"err: {str(err)}")
        else:
            logger.info(f"multical result, {res}")

    refine_result = py_cli_interaction.must_parse_cli_bool("refine?", default_value=False)
    if refine_result:
        logger.info("refining result, press Esc to close preview")
        start_idx = py_cli_interaction.must_parse_cli_int("start idx?", default_value=0)
        logger.info("examining camera extrinsics")
        refine_multical_result(multical_path, start_idx)
    else:
        patch_refine_extrinsic(multical_path, {cam_name: np.eye(4) for cam_name in cameras_name_list})

    get_transmat = py_cli_interaction.must_parse_cli_bool("get transmat?", default_value=False)
    rot: np.ndarray = None
    trans: np.ndarray = None
    if get_transmat:
        logger.info("getting transmat")
        for merged_pc in generate_multicam_pc(multical_path):
            rot, trans, err = get_trans_mat_by_nws_combined(merged_pc)
            if err is None:
                logger.info("rotation and translation")
                logger.info(rot, trans)
                patch_rot_trans(multical_path, rot, trans)
                break
            else:
                should_continue = py_cli_interaction.must_parse_cli_bool("continue?", default_value=False)
                if should_continue:
                    continue
                else:
                    break

    # update realworld rot trans:
    if (rot is not None and trans is not None):
        pass
    else:
        cam_info = MulticalCameraInfo(multical_path)
        res = cam_info.get_realworld_rot_trans()
        if res is not None:
            rot, trans = res["R"], res["T"]

    if rot is None or trans is None:
        logger.info("rot is None or trans is None, skipping")
        return

    mark_workspace = py_cli_interaction.must_parse_cli_bool("mark workspace?", default_value=False)
    if mark_workspace:
        logger.info("marking workspace")
        for merged_pc in generate_multicam_pc(multical_path):
            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3:4] = trans
            merged_pc.transform(mat)

            vis_pcds([merged_pc])

            limits_nws, err = get_workspace_limit_by_interaction(merged_pc)

            if err is None:
                logger.info("workspace limits:")
                logger.info(limits_nws)
                patch_workspace_limits(multical_path, limits_nws)
                break
            else:
                should_continue = py_cli_interaction.must_parse_cli_bool("continue?", default_value=False)
                if should_continue:
                    continue
                else:
                    break


def entry_point(argv):
    logging.basicConfig(level=logging.INFO)

    if len(argv) < 1:
        try:
            f = plyer.filechooser.open_file(title="Select a calibration.json", filter="*.json")
            if len(f) > 0:
                multical_path = osp.dirname(f[0])
                return main(multical_path)
            else:
                logger.info("abort")
                return 1
        except Exception as err:
            logger.error(f"error: {err}")
            print("Usage: python -m azure_kinect_apiserver multical <path>")
            return 1
    else:
        return main(argv[0])


if __name__ == '__main__':
    import sys

    entry_point(sys.argv[1:])
