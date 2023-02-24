import glob
import logging

import open3d as o3d
import numpy as np
import cv2
import os
import tqdm
import json
from scipy.spatial.transform import Rotation as R

from typing import Optional, Tuple, List, Dict


class RSPointCloudHelper:
    def __init__(self, rgb, depth, camera_intrinsic_path=None, transform=None, L515=False, camera_intrinsic=None) -> None:
        if camera_intrinsic is None:
            self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(camera_intrinsic_path)
        else:
            width, height, intrinsic_matrix = camera_intrinsic
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), intrinsic_matrix[0, 0], intrinsic_matrix[1, 1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.transform = transform

        self.color_raw = o3d.geometry.Image(rgb.astype(np.uint8))
        self.depth_raw = depth
        # self.depth_raw[:, :300] = 0
        # self.depth_raw[:, 860:] = 0
        if L515:
            self.depth_raw = self.depth_raw / 4
        # self.depth_raw = self.depth_raw * 0.975
        # self.depth_raw = self.depth_raw - 20
        self.depth_raw = o3d.geometry.Image(self.depth_raw.astype(np.uint16))

        self.pcd = self.rgbd2pc(self.color_raw, self.depth_raw, self.camera_intrinsic, self.transform)

    @staticmethod
    def rgbd2pc(color_raw, depth_raw, camera_intrinsic, transform=None, depth_scale=1000.):
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw,
                                                                        depth_raw,
                                                                        depth_scale,
                                                                        2.0,
                                                                        convert_rgb_to_intensity=False)

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=camera_intrinsic,
            # extrinsic=camera_extrinsic,
            # extrinsic=np.linalg.inv(camera_extrinsic),
        )

        # remove according to normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # pcd_mask = (np.abs(np.dot(pcd.normals, np.array([0, 0, 1]))) < 0.05)
        # pcd_mask = np.logical_not(pcd_mask)
        pcd_mask = np.ones(shape=(len(pcd.normals),)).astype(bool)

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[pcd_mask])
        new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[pcd_mask])
        pcd = new_pcd

        if transform is not None:
            # inv_extrinsic = np.linalg.inv(transform)
            # pcd.transform(inv_extrinsic)
            pcd.transform(transform)

        # denoise
        # _, ind = pcd.remove_radius_outlier(nb_points=50, radius=0.02)
        # _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.05)
        # pcd = pcd.select_by_index(ind)

        return pcd

    def vis(self, size=0.1):
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        o3d.visualization.draw_geometries([self.pcd] + [coordinate_frame])


# from liuliu
def p2p_ipc(source_h, target_h, initial_trans=np.identity(4)):
    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.01  # 3cm distance threshold

    # Fine - ICP
    source_h.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
    target_h.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_h.pcd, target_h.pcd, threshold, initial_trans,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return reg_p2p


# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector
# From: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
# Copyright (c) 2020, Nghia Ho
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def get_trans_mat_by_north_and_west_combined(pc_helper):
    # Get R, T matrix
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    print('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    vis1.add_geometry(pc_helper.pcd)
    vis1.update_renderer()
    vis1.run()  # user picks points
    vis1.destroy_window()

    pts = vis1.get_picked_points()
    pcd_points = np.asarray(pc_helper.pcd.points)
    pts = pcd_points[pts]
    base, north, west = pts[0], pts[1], pts[2]
    north_direction, west_direction = north - base, west - base
    north_direction /= np.linalg.norm(north_direction)
    west_direction /= np.linalg.norm(west_direction)

    point_set_A = np.concatenate((
        base[:, None],
        (base + north_direction)[:, None],
        (base + west_direction)[:, None],
    ), axis=1)
    point_set_B = np.array(
        [[0, 0, 0],
         [1, 0, 0],
         [0, 1, 0]]
    ).T
    rot, trans = rigid_transform_3D(point_set_A, point_set_B)

    return rot, trans


# Credit: Xuehan
def colored_point_cloud_registration(source, target):
    # o3d.visualization.draw_geometries([source, target])
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    # draw_registration_result_original_color(source, target, result_icp.transformation)
    return result_icp.transformation


def colored_point_cloud_registration_robust(source, target):
    # o3d.visualization.draw_geometries([source, target])
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale] * 10
        radius = voxel_radius[scale]

        print("3-1. Estimate normal.")
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        voxel_size = radius * 5
        print("3-2. Downsample with a voxel size %.2f" % voxel_size)
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        # fps_num = 2000
        # source_down = source.farthest_point_down_sample(fps_num)
        # target_down = target.farthest_point_down_sample(fps_num)

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, voxel_size, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))

        # result_icp = o3d.pipelines.registration.registration_icp(
        #     source_down, target_down, voxel_size, current_transformation,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
        #                                                       relative_rmse=1e-6,
        #                                                       max_iteration=iter))

        source = source.transform(result_icp.transformation)
        current_transformation = current_transformation @ result_icp.transformation
        print("result_icp:", result_icp)
        print("current_step_matrix:", result_icp.transformation)
        print("accumulated_matrix:", current_transformation)
        # print(result_icp.transformation)
    # draw_registration_result_original_color(source, target, result_icp.transformation)
    return current_transformation


# NOTICE: hand-eye calibration
transform_mat = np.eye(4)
# If you do not want hand-eye calibration, you can simply comment out the following lines.
transform_mat[:3, :3] = R.from_quat((0.0710936350871877, 0.12186999407, -0.583974393827845, 0.799416854295149)).as_matrix()  # x, y, z, w
transform_mat[:3, 3] = np.array((0.17857327, 0.5218457133, 0.28518456))  # x,y,z


def vis_pcds(transformed_pcd_list):
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    transformed_pcd_all = o3d.geometry.PointCloud()
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]
    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.concatenate(xyz_list, axis=0))
    transformed_pcd_all.colors = o3d.utility.Vector3dVector(np.concatenate(rgb_list, axis=0))
    o3d.visualization.draw_geometries([coordinate, transformed_pcd_all])


def save_pcds(transformed_pcd_list, save_path):
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]

    xyzs = np.concatenate(xyz_list, axis=0)
    rgbs = np.concatenate(rgb_list, axis=0)

    transformed_pcd_all = o3d.geometry.PointCloud()
    transformed_pcd_all.points = o3d.utility.Vector3dVector(xyzs)
    transformed_pcd_all.colors = o3d.utility.Vector3dVector(rgbs)

    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.asarray(transformed_pcd_all.points)[:, (2, 0, 1)])
    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.asarray(transformed_pcd_all.points) * np.array((1, -1, -1)))
    transformed_pcd_all.transform(transform_mat)

    # vis_pcds([transformed_pcd_all])

    xyz = np.asarray(transformed_pcd_all.points)
    rgb = np.asarray(transformed_pcd_all.colors)

    # NOTICE: The following lines are to purge the points that is outside of the workspace
    # Comment out them if they are unwanted!
    # mask = (xyz[:, 0] > 0.35) & (xyz[:, 0] < 0.95) & (xyz[:, 1] > -0.5) & (xyz[:, 1] < 0.5) & (xyz[:, 2] > 0.03) & (xyz[:, 2] < 0.4)
    # xyz = xyz[mask]
    # rgb = rgb[mask]

    masked_pcd = o3d.geometry.PointCloud()
    masked_pcd.points = o3d.utility.Vector3dVector(xyz)
    masked_pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.io.write_point_cloud(os.path.join('./output', f'{save_path}.ply'), masked_pcd)


# A class wrapper to the camera in/ex-trinsics.
class CameraInfo:
    def __init__(self, json_path):
        self.json_path = json_path
        self.camera_info = json.load(open(json_path))
        self.master_cam = [k for k in self.camera_info['camera_poses'].keys() if '_to_' not in k]
        assert (len(self.master_cam) == 1)
        self.master_cam = self.master_cam[0]

    def get_distort(self, serial):
        return np.array(self.camera_info['cameras'][serial]['dist'])[0]

    def get_intrinsic(self, serial):
        # Here we use the intrinsic from the multi-cal
        return np.array(self.camera_info['cameras'][serial]['K'])
        # Of course, you can switch to the factory-calibration
        # return self.get_intrinsic_factory(serial)

    def get_intrinsic_factory(self, serial):
        intrinsic = json.load(open(f'./intrinsics/{serial}.json'))['intrinsic_matrix']
        intrinsic = np.array(intrinsic)
        intrinsic = intrinsic.reshape((3, 3))
        print(intrinsic)
        return intrinsic

    def get_distort(self, serial):
        return np.array(self.camera_info['cameras'][serial]['dist'])

    def get_extrinsic(self, serial):
        if serial == self.master_cam:
            return np.eye(4, dtype=float)

        R = self.camera_info['camera_poses'][serial + '_to_' + self.master_cam]['R']
        T = self.camera_info['camera_poses'][serial + '_to_' + self.master_cam]['T']
        camera_extrinsic = np.eye(4, dtype=float)
        camera_extrinsic[:3, :3] = np.array(R)
        camera_extrinsic[:3, 3] = np.array(T)

        camera_extrinsic = np.linalg.inv(camera_extrinsic)
        return camera_extrinsic

        # vis_pcds(transformed_pcd_list)


def get_chroma_mask(img: np.ndarray, margin=50) -> Tuple[np.ndarray, Optional[Exception]]:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)

    # get uniques
    # unique_colors, counts = np.unique(s, return_counts=True)
    # max_s = unique_colors[np.argmax(counts)]
    # max_s = 90
    # if max_s == 0:
    #     np.ones_like(img), Exception("No chroma detected")
    #
    # logging.debug(max_s)
    # mask = (s > max_s - margin) & (s < max_s + margin)
    # mask = np.bitwise_not(mask)
    # mask = (mask * 255).astype(np.uint8)

    mask = cv2.inRange(img_hsv, (35, 43, 46), (77, 255, 255))
    # cv2.imshow("mask_hsv", mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    # cv2.imshow("mask_morphology_1", mask)

    mask = cv2.erode(mask, np.ones((8, 8), np.uint8))
    mask = cv2.dilate(mask, np.ones((16, 16), np.uint8))
    mask = cv2.erode(mask, np.ones((8, 8), np.uint8))
    # cv2.imshow("mask_morphology_2", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_index = np.argmax([cv2.contourArea(c) for c in contours])
    biggest_contour = contours[contour_index]

    result_mask = np.zeros_like(mask)
    cv2.fillPoly(result_mask, [biggest_contour], 255)
    # cv2.imshow("mask_contour", result_mask)

    result_mask = cv2.medianBlur(result_mask, 5)
    result_mask = np.bitwise_not(result_mask)
    # cv2.imshow("mask_contour_smooth_final", result_mask)

    # img[result_mask == 255] = (255, 0, 0)
    # cv2.imshow("img", img)

    # cv2.waitKey(0)

    return result_mask, None


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        img[mask == 0] = 0
    else:
        img[mask == 0] = (0, 0, 0)
    return img


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_chroma_filter")
    # Add or Remove the serial of the cameras here!
    CAMERAS = [
        '000673513312',
        '000700713312',
        '000729313312',
        '000760113312',
    ]

    # The path to the multi-cal calibration.json file.
    cam_info = CameraInfo(r'C:\Users\robotflow\Desktop\azure-kinect-apiserver\azure_kinect_data\cali_20230223_164714\calibration.json')

    # The path to the recording folder.
    BASE_PATH = r'C:\Users\robotflow\Desktop\azure-kinect-apiserver\azure_kinect_data\test'
    logger.info(BASE_PATH)

    finetune_transform_dict = {}

    color_img_path_collection = {
        cam_name: glob.glob(os.path.join(BASE_PATH, cam_name, 'color', '*.jpg')) for cam_name in CAMERAS
    }
    depth_img_path_collection = {
        cam_name: glob.glob(os.path.join(BASE_PATH, cam_name, 'depth', '*.png')) for cam_name in CAMERAS
    }

    with tqdm.tqdm(total=18) as pbar:
        for img_idx in range(0, 18):

            raw_pc_by_camera = {}
            transformed_pcd_list = []

            for camera in CAMERAS:
                color_img_path = color_img_path_collection[camera][img_idx]
                color = cv2.imread(color_img_path)[:, :, ::-1]
                depth_img_path = depth_img_path_collection[camera][img_idx]
                depth = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
                color_mask, ret = get_chroma_mask(color)
                color = apply_mask(color, color_mask)
                depth = apply_mask(depth, color_mask)
                # cv2.imshow("color_mask", color_mask)
                # cv2.waitKey()

                # mask_img_path = os.path.join(BASE_PATH, camera, 'mask', f'{img_idx}.png')
                # mask = cv2.imread(mask_img_path)

                # undistort
                # cam_matrix = cam_info.get_intrinsic(camera)
                # cam_dist = cam_info.get_distort(camera)
                # color_undist = cv2.undistort(color, cam_matrix, cam_dist)
                # depth_undist = cv2.undistort(depth, cam_matrix, cam_dist)
                # mask_undist = cv2.undistort(mask, cam_matrix, cam_dist)

                # color = color_undist
                # depth = depth_undist
                # mask = mask_undist

                # mask the original img for registration
                # mask = (np.sum(mask, axis=2) == 255*3)
                # depth[mask] = 0

                # cv2.imshow('origin', color)
                # cv2.imshow('undist', color_undist)
                # cv2.waitKey(0)

                # pc = RSPointCloudHelper(color, depth, f'D:/Temp/articulated/intrinsics/{camera}.json', cam_info.get_extrinsic(camera))
                pc = RSPointCloudHelper(color, depth, None, cam_info.get_extrinsic(camera), False, camera_intrinsic=(2160, 3840, cam_info.get_intrinsic(camera)))
                # pc.vis()

                raw_pc_by_camera[camera] = pc

            # for camera in CAMERAS:
            #     vis_pcds([raw_pc_by_camera[camera].pcd])
            # vis_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS])
            # save_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS], os.path.join(BASE_PATH, 'pcds', f'{img_idx}'))
            save_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS], 'scene_%06d' % img_idx)

            # print('Master Cam:', cam_info.master_cam)
            # for camera in CAMERAS:
            #     if camera == cam_info.master_cam:
            #         transformed_pcd_list.append(raw_pc_by_camera[camera].pcd)
            #         continue

            #     source = raw_pc_by_camera[camera].pcd
            #     target = raw_pc_by_camera[cam_info.master_cam].pcd

            #     if img_idx == 0:
            #         print(f'Aligning {camera} to {cam_info.master_cam}')
            #         finetune_transform = colored_point_cloud_registration_robust(source, target)
            #         finetune_transform_dict[camera] = finetune_transform
            #         transformed_pcd_list.append(source.transform(finetune_transform))
            #         print(finetune_transform)
            #     else:
            #         print(f'Aligned {camera} to {cam_info.master_cam}')
            #         print(finetune_transform_dict[camera])
            #         transformed_pcd_list.append(source.transform(finetune_transform_dict[camera]))

            pbar.update(1)
