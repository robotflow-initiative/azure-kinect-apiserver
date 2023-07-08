import open3d as o3d
import numpy as np
import logging
import os.path as osp
import argparse
import tqdm
import sys
import cv2
from scipy.spatial.transform import Rotation as R

from azure_kinect_apiserver.apiserver.app import KinectSystemCfg, Application
from azure_kinect_apiserver.common import PointCloudHelper

from azure_kinect_apiserver.common import vis_pcds, save_pcds, rigid_transform_3d
from azure_kinect_apiserver.common.MulticalCamera import MulticalCameraInfo
import copy
from typing import Optional, List

VOXEL_SIZE = 0.05


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def input_point_cloud(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    trans_init = np.eye(4)
    trans_init[:3, :3] = R.random().as_matrix()
    trans_init[:3, 3] = np.random.rand(3) * 2 - 1
    source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
          % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    return result


def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


SOURCE_CAMERA = 0
TARGET_CAMERA = 1


def test_ransac(args):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("azure_kinect_apiserver.cmd.calibration")
    logger.info("using config file: {}".format(osp.realpath(args.config)))

    cfg = KinectSystemCfg(args.config)
    app = Application(cfg)
    app.enter_single_shot_mode()

    cam_matrix_list: List[Optional[np.ndarray]] = [None] * len(app.option.camera_options)
    cam_dist_list: List[Optional[np.ndarray]] = [None] * len(app.option.camera_options)

    for idx in [SOURCE_CAMERA, TARGET_CAMERA]:
        cam_matrix = np.array(app.calibrations[app.option.camera_options[idx].sn].get_intrinsic_matrix()['color'])
        cam_dist = app.calibrations[app.option.camera_options[idx].sn].get_distort_parameters()['color']
        cam_dist = np.array(
            [cam_dist['k'][0],
             cam_dist['k'][1],
             cam_dist['p'][0],
             cam_dist['p'][1],
             cam_dist['k'][2],
             cam_dist['k'][3],
             cam_dist['k'][4],
             cam_dist['k'][5]]
        )
        cam_matrix_list[idx] = cam_matrix
        cam_dist_list[idx] = cam_dist

    with tqdm.tqdm(total=1000) as pbar:
        for idx in range(1000):
            raw_pc_list: List[Optional[o3d.geometry.PointCloud]] = [None] * len(app.option.camera_options)
            color_frames, depth_frames, index, err = app.single_shot_mem(idx)
            for cam_idx in [SOURCE_CAMERA, TARGET_CAMERA]:
                color_img = color_frames[cam_idx]
                depth_img = depth_frames[cam_idx]

                # undistort
                color_undistort = cv2.undistort(color_img, cam_matrix_list[cam_idx], cam_dist_list[cam_idx])
                depth_undistort = cv2.undistort(depth_img, cam_matrix_list[cam_idx], cam_dist_list[cam_idx])

                raw_pc = PointCloudHelper(color_undistort,
                                          depth_undistort,
                                          camera_intrinsic_desc=(color_img.shape[1], color_img.shape[0], cam_matrix_list[cam_idx]),
                                          transform=np.eye(4))
                raw_pc_list[cam_idx] = raw_pc.pcd

            source, target, source_down, target_down, source_fpfh, target_fpfh = input_point_cloud(raw_pc_list[SOURCE_CAMERA], raw_pc_list[SOURCE_CAMERA], VOXEL_SIZE)
            result_ransac = execute_fast_global_registration(source_down, target_down,
                                                             source_fpfh, target_fpfh,
                                                             VOXEL_SIZE)
            print(result_ransac)
            draw_registration_result(source_down, target_down, result_ransac.transformation)
            # vis_pcds([raw_pc.pcd])
            result_icp = refine_registration(source_down, target_down, result_ransac, VOXEL_SIZE)
            draw_registration_result(source_down, target_down, result_icp.transformation)

            pbar.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--aruco_preview', action='store_true')
    print(sys.argv)

    args = parser.parse_args(sys.argv[1:])
    test_ransac(args)
