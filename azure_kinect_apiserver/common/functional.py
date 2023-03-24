import logging
import os
import subprocess
from typing import Tuple, Optional, List, Iterable, Callable

import cv2
import numpy as np
import open3d as o3d
import py_cli_interaction

from .point import PointCloudHelper

logger = logging.getLogger('azure_kinect_apiserver.common.functional')


def clip_depth_image(depth_image: np.ndarray, min_depth: int = 0, max_depth: int = 10000) -> np.ndarray:
    """Converts a depth image to a color image.

    Args:
        depth_image: A depth image as a numpy array.

    Returns:
        A color image as a numpy array.
        :param depth_image:
        :param min_depth:
        :param max_depth:
    """
    depth_image = depth_image - min_depth
    depth_image = depth_image / (max_depth - min_depth) * 255
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)


def probe_device(exec_path: str):
    res = []
    process = subprocess.Popen(
        f"{exec_path} --list",
        stdout=subprocess.PIPE
    )
    output = process.communicate()[0].decode("utf-8")
    output = output.split('\r\n')[:-1]
    for device_string in output:
        try:
            device = device_string.split('\t')
            result = {}
            for device_property in device:
                device_property_list = device_property.split(':')
                result[device_property_list[0]] = device_property_list[1]
            result['Index'] = int(result['Index'])
            res.append(result)
        except Exception as e:
            logging.getLogger('azure_kinect_apiserver.common.functional').error(e)
            return [], e
    return res, None


__FAKE_COLORS__ = [
    np.array([0.12156862745098039, 0.4666666666666667, 0.7058823529411765]),
    np.array([1.0, 0.4980392156862745, 0.054901960784313725]),
    np.array([0.17254901960784313, 0.6274509803921569, 0.17254901960784313]),
    np.array([0.8392156862745098, 0.15294117647058825, 0.1568627450980392]),
    np.array([0.5803921568627451, 0.403921568627451, 0.7411764705882353]),
    np.array([0.5490196078431373, 0.33725490196078434, 0.29411764705882354]),
    np.array([0.8901960784313725, 0.4666666666666667, 0.7607843137254902]),
    np.array([0.4980392156862745, 0.4980392156862745, 0.4980392156862745]),
    np.array([0.7372549019607844, 0.7411764705882353, 0.13333333333333333]),
    np.array([0.09019607843137255, 0.7450980392156863, 0.8117647058823529])
]


def vis_pcds(transformed_pcd_list, fake_color=False):
    coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    transformed_pcd_all = o3d.geometry.PointCloud()
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    if fake_color:
        rgb_list = [np.tile(__FAKE_COLORS__[i], (xyz_list[i].shape[0], 1)) for i in range(len(xyz_list))]
    else:
        rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]
    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.concatenate(xyz_list, axis=0))
    transformed_pcd_all.colors = o3d.utility.Vector3dVector(np.concatenate(rgb_list, axis=0))
    o3d.visualization.draw_geometries([coordinate, transformed_pcd_all])


def save_pcds(transformed_pcd_list: List[o3d.geometry.PointCloud],
              save_path: str,
              tag: str,
              seperate=False,
              fake_color=False,
              filters: List[Callable[[o3d.geometry.PointCloud], o3d.geometry.PointCloud]] = None):
    """
    If you do not want hand-eye calibration, you can simply comment out the following lines.
    transform_mat[:3, :3] = R.from_quat((0.0710936350871877, 0.12186999407, -0.583974393827845, 0.799416854295149)).as_matrix()  # x, y, z, w
    transform_mat[:3, 3] = np.array((0.17857327, 0.5218457133, 0.28518456))  # x,y,z

    :param tag:
    :param transformed_pcd_list:
    :param save_path:
    :param seperate:
    :param fake_color:
    :return:

    TODO: Optimize
    """
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    if fake_color:
        rgb_list = [np.repeat(__FAKE_COLORS__[i % len(__FAKE_COLORS__)][None, :], len(xyz_list[i]), axis=0) for i in range(len(xyz_list))]
    else:
        rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]

    if not seperate:
        xyz_list = [np.concatenate(xyz_list, axis=0)]
        rgb_list = [np.concatenate(rgb_list, axis=0)]

    for i in range(len(xyz_list)):
        xyzs = xyz_list[i]
        rgbs = rgb_list[i]

        transformed_pcd_all = o3d.geometry.PointCloud()
        transformed_pcd_all.points = o3d.utility.Vector3dVector(xyzs)
        transformed_pcd_all.colors = o3d.utility.Vector3dVector(rgbs)

        xyz = np.asarray(transformed_pcd_all.points)
        rgb = np.asarray(transformed_pcd_all.colors)

        masked_pcd = o3d.geometry.PointCloud()
        masked_pcd.points = o3d.utility.Vector3dVector(xyz)
        masked_pcd.colors = o3d.utility.Vector3dVector(rgb)
        if seperate:
            o3d.io.write_point_cloud(os.path.join(save_path, f'{tag}_{i}.ply'), masked_pcd)
        else:
            o3d.io.write_point_cloud(os.path.join(save_path, f'{tag}.ply'), masked_pcd)


# noinspection PyPep8Naming
def rigid_transform_3d(A, B):
    """
    Input: expects 3xN matrix of points
    Returns R,t
    R = 3x3 rotation matrix
    t = 3x1 column vector
    From: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    Copyright (c) 2020, Nghia Ho

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


    A = R*B + t
    :param A:
    :param B:
    :return:

    TODO: Optimize
    """
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
        logger.debug("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


# Credit: Xuehan
def colored_point_cloud_registration_robust(source, target, debug=False):
    # o3d.visualization.draw_geometries([source, target])
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    current_transformation = np.identity(4)
    logger.debug("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale] * 10
        radius = voxel_radius[scale]

        logger.debug("3-1. Estimate normal.")
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        voxel_size = radius * 5
        logger.debug("3-2. Downsample with a voxel size %.2f" % voxel_size)
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        # fps_num = 2000
        # source_down = source.farthest_point_down_sample(fps_num)
        # target_down = target.farthest_point_down_sample(fps_num)

        logger.debug("3-3. Applying colored point cloud registration")
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

        current_transformation = result_icp.transformation
        logger.debug("result_icp:", result_icp)
        logger.debug("current_step_matrix:", result_icp.transformation)
        logger.debug("accumulated_matrix:", current_transformation)
        # logger.debug(result_icp.transformation)
    # draw_registration_result_original_color(source, target, result_icp.transformation)
    return current_transformation


def point_cloud_registration_fine(source, target, debug=False):
    # o3d.visualization.draw_geometries([source, target])
    radius = 0.01
    iter = 10000
    init_transformation = np.identity(4)
    logger.debug("3. Colored point cloud registration")

    logger.debug("3-1. Estimate normal.")
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    logger.debug("3-3. Applying colored point cloud registration")
    if debug:
        vis_pcds([source])
        vis_pcds([target])
        vis_pcds([source, target])
    result_icp = o3d.pipelines.registration.registration_generalized_icp(
        source, target, radius * 5, init_transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7,
                                                          relative_rmse=1e-7,
                                                          max_iteration=iter))

    if debug:
        vis_pcds([source.transform(result_icp.transformation), target])
    logger.debug("result_icp:", result_icp)
    logger.debug("current_step_matrix:", result_icp.transformation)
    # logger.debug(result_icp.transformation)
    # draw_registration_result_original_color(source, target, result_icp.transformation)
    return result_icp.transformation


def colored_point_cloud_registration_fine_color(source, target, debug=False):
    # o3d.visualization.draw_geometries([source, target])
    radius = 0.01
    iter = 10000
    init_transformation = np.identity(4)
    logger.debug("3. Colored point cloud registration")

    logger.debug("3-1. Estimate normal.")
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

    logger.debug("3-3. Applying colored point cloud registration")
    if debug:
        vis_pcds([source])
        vis_pcds([target])
        vis_pcds([source, target])
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source, target, radius * 5, init_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-7,
                                                          relative_rmse=1e-7,
                                                          max_iteration=iter))

    if debug:
        vis_pcds([source.transform(result_icp.transformation), target])
    logger.debug("result_icp:", result_icp)
    logger.debug("current_step_matrix:", result_icp.transformation)
    # logger.debug(result_icp.transformation)
    # draw_registration_result_original_color(source, target, result_icp.transformation)
    return result_icp.transformation


def get_nws_points(point_cloud: o3d.geometry.PointCloud) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Optional[Exception]]:
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    logger.debug('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    vis1.add_geometry(point_cloud)
    vis1.update_renderer()
    vis1.run()  # user picks points
    vis1.destroy_window()

    pts_sel = vis1.get_picked_points()
    if len(pts_sel) != 3:
        return (np.array([]), np.array([]), np.array([])), Exception("Please select 3 points")

    pcd_points = np.asarray(point_cloud.points)
    pts = pcd_points[pts_sel]
    base, north, west = pts[0], pts[1], pts[2]

    return (base, north, west), None


def get_nws_points_n(point_cloud: o3d.geometry.PointCloud, n: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[Exception]]:
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    logger.debug('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    vis1.add_geometry(point_cloud)
    vis1.update_renderer()
    vis1.run()  # user picks points
    vis1.destroy_window()

    pts_sel = vis1.get_picked_points()
    if len(pts_sel) != 3 * n:
        return [(np.array([]), np.array([]), np.array([]))], Exception(f"Please select {3 * n} points")

    res = []
    pcd_points = np.asarray(point_cloud.points)
    pts = pcd_points[pts_sel]
    for i in range(n):
        res.append((pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]))

    return res, None


def get_color(point_cloud: o3d.geometry.PointCloud, n: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, np.ndarray]], Optional[Exception]]:
    vis1 = o3d.visualization.VisualizerWithEditing()
    vis1.create_window('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    logger.debug('Please press Shift and click 3 Points: Base - North - West. Press Q to exit.')
    vis1.add_geometry(point_cloud)
    vis1.update_renderer()
    vis1.run()  # user picks points
    vis1.destroy_window()

    pts_sel = vis1.get_picked_points()
    if len(pts_sel) != n:
        return [(np.array([]), np.array([]), np.array([]))], Exception(f"Please select {n} points")

    colors = []
    pcd_colors = np.asarray(point_cloud.colors)
    pts = pcd_colors[pts_sel]

    return pts, None


def get_trans_mat_by_nws_combined(point_cloud: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray, Optional[Exception]]:
    """
    Get transformation matrix by north and west direction, with interaction

    :param pc_helper:
    :return:
    """
    (base, north, west), err = get_nws_points(point_cloud)
    if err is not None:
        return np.array([]), np.array([]), err
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
    rot, trans = rigid_transform_3d(point_set_A, point_set_B)

    return rot, trans, None


def get_marker_by_nws_combined(point_cloud: o3d.geometry.PointCloud, n: int = 1) -> Tuple[List[np.ndarray], Optional[Exception]]:
    """
    Get transformation matrix by north and west direction, with interaction

    :param pc_helper:
    :return:
    """
    points, err = get_nws_points_n(point_cloud, n)
    if err is not None:
        return [np.array([])], err

    res = []
    for (base, north, west) in points:
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
        rot, trans = rigid_transform_3d(point_set_A, point_set_B)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3:4] = trans
        T = np.linalg.inv(T)
        T[:3, 3:4] = base[:, None]
        res.append(T)

    return res, None


def get_workspace_limit_by_interaction(point_cloud: o3d.geometry.PointCloud) -> Tuple[List[np.ndarray], Optional[Exception]]:
    """
    Get workspace limit by interaction

    :param pc_helper:
    :return:
    """
    (base, north, west), err = get_nws_points(point_cloud)
    if err is not None:
        return [np.array([])], Exception("Please select 3 points")

    height_m = py_cli_interaction.must_parse_cli_float("Please input height (m): ")
    north_direction, west_direction = north - base, west - base
    up_direction = np.cross(north_direction, west_direction)
    up_direction /= np.linalg.norm(up_direction)

    sky = base + up_direction * height_m

    return [base, north, west, sky], None


def merge_point_cloud_helpers(raw_pc_by_camera: Iterable[PointCloudHelper]) -> Optional[o3d.geometry.PointCloud]:
    merged_pc = None
    for pc in raw_pc_by_camera:
        if merged_pc is None:
            merged_pc = pc.pcd
        else:
            merged_pc = merged_pc + pc.pcd
    return merged_pc


def remove_green_background(img: np.ndarray):
    img_blur = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 矩形结构
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.erode(mask_open, kernel, mask_open, iterations=1)
    cv2.GaussianBlur(mask_open, (3, 3), 0, mask_open)
    mask_open = cv2.bitwise_not(mask_open)
    res = cv2.bitwise_and(img_blur, img_blur, mask=mask_open)
    return res, mask_open
