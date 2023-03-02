import logging
import os
import subprocess

import cv2
import numpy as np
import open3d as o3d


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
            result = dict()
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
    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    transformed_pcd_all = o3d.geometry.PointCloud()
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    if fake_color:
        rgb_list = [np.tile(__FAKE_COLORS__[i], (xyz_list[i].shape[0], 1)) for i in range(len(xyz_list))]
    else:
        rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]
    transformed_pcd_all.points = o3d.utility.Vector3dVector(np.concatenate(xyz_list, axis=0))
    transformed_pcd_all.colors = o3d.utility.Vector3dVector(np.concatenate(rgb_list, axis=0))
    o3d.visualization.draw_geometries([coordinate, transformed_pcd_all])


def save_pcds(transformed_pcd_list, save_path, seperate=True, fake_color=False, transform_mat=np.eye(4)):
    """
    If you do not want hand-eye calibration, you can simply comment out the following lines.
    transform_mat[:3, :3] = R.from_quat((0.0710936350871877, 0.12186999407, -0.583974393827845, 0.799416854295149)).as_matrix()  # x, y, z, w
    transform_mat[:3, 3] = np.array((0.17857327, 0.5218457133, 0.28518456))  # x,y,z

    :param transformed_pcd_list:
    :param save_path:
    :param seperate:
    :param fake_color:
    :param transform_mat:
    :return:

    TODO: Optimize
    """
    xyz_list = [np.asarray(pcd.points) for pcd in transformed_pcd_list]
    if fake_color:
        rgb_list = [np.repeat(__FAKE_COLORS__[i % len(__FAKE_COLORS__)][None, :], len(xyz_list[i]), axis=0) for i in range(len(xyz_list))]
    else:
        rgb_list = [np.asarray(pcd.colors) for pcd in transformed_pcd_list]

    if seperate:
        for i in range(len(xyz_list)):
            xyzs = xyz_list[i]
            rgbs = rgb_list[i]

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
            o3d.io.write_point_cloud(os.path.join('./output', f'{save_path}_{i}.ply'), masked_pcd)
    else:
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


def rigid_transform_3D(A, B):
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
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
