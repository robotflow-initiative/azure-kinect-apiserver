from typing import Tuple, List, Union

import cv2
import numpy as np
import open3d as o3d

POINT_DEPTH_SCALE_NORMAL = 1
POINT_DEPTH_SCALE_L515 = 0.25


class PointCloudHelper:
    def __init__(self,
                 rgb,
                 depth,
                 depth_scale=None,
                 camera_intrinsic_desc: Tuple[int, int, Union[List[List[float]], np.ndarray]] = None,
                 camera_intrinsic_path=None,
                 transform=None,
                 computed: bool = True, *args, **kwargs) -> None:
        if camera_intrinsic_desc is None:
            self.camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(camera_intrinsic_path)
        else:
            width, height, intrinsic_matrix = camera_intrinsic_desc
            self.camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
        self.transform = transform

        self.color_raw = o3d.geometry.Image(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGRA2RGB))

        depth_raw = depth
        # Uncomment the following lines to remove the edges
        # self.depth_raw[:, :300] = 0
        # self.depth_raw[:, 860:] = 0
        if depth_scale is not None:
            depth_raw = depth_raw * depth_scale
        # Uncomment the following lines to scale / offset the depth
        # self.depth_raw = self.depth_raw * 0.975
        # self.depth_raw = self.depth_raw - 20
        self.depth_raw = o3d.geometry.Image(depth_raw.astype(np.uint16))

        if computed:
            self._pcd = self.rgbd2pc(self.color_raw, self.depth_raw, self.camera_intrinsic, self.transform, *args, **kwargs)
        else:
            self._pcd = None

        self.computed = computed

    @property
    def pcd(self):
        if self._pcd is not None:
            return self._pcd
        else:
            self._pcd = self.rgbd2pc(self.color_raw, self.depth_raw, self.camera_intrinsic, self.transform)
            return self._pcd

    @staticmethod
    def rgbd2pc(color_raw,
                depth_raw,
                camera_intrinsic,
                transform=None,
                depth_scale=1000.,
                depth_trunc=2.0,
                norm_radius=0.01,
                norm_max_nn=30,
                enable_norm_filter=True,
                norm_filter_threshold=0.2,
                enable_denoise=True,
                denoise_radius=0.05,
                denoise_nb_points=16,
                denoise_nb_neighbors=20,
                denoise_std_ratio=2, ):
        if isinstance(color_raw, np.ndarray):
            color_raw = o3d.geometry.Image(cv2.cvtColor(color_raw.astype(np.uint8), cv2.COLOR_BGRA2RGB))  # Convert to RGB
        if isinstance(depth_raw, np.ndarray):
            depth_raw = o3d.geometry.Image(depth_raw.astype(np.uint16))
        if isinstance(camera_intrinsic, tuple):
            width, height, intrinsic_matrix = camera_intrinsic
            camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(int(width), int(height), intrinsic_matrix[0][0], intrinsic_matrix[1][1], intrinsic_matrix[0][2], intrinsic_matrix[1][2])

        rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
            color_raw,
            depth_raw,
            depth_scale,
            depth_trunc,
            convert_rgb_to_intensity=False
        )

        pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
            image=rgbd_image,
            intrinsic=camera_intrinsic,
            extrinsic=np.eye(4),
            # extrinsic=camera_extrinsic,
            # extrinsic=np.linalg.inv(camera_extrinsic),
        )

        # remove according to normals
        if enable_norm_filter:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=norm_radius, max_nn=norm_max_nn)
            )
            pcd_mask = np.linalg.norm(np.array(pcd.normals) * np.array(pcd.points) / np.linalg.norm(pcd.points, axis=1, keepdims=True), axis=1) < norm_filter_threshold
            pcd_mask = np.logical_not(pcd_mask)
            _pcd = o3d.geometry.PointCloud()
            _pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[pcd_mask])
            _pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[pcd_mask])
            pcd = _pcd

        if transform is not None:
            # inv_extrinsic = np.linalg.inv(transform)
            # pcd.transform(inv_extrinsic)
            pcd.transform(transform)

        # denoise
        if enable_denoise:
            _, ind = pcd.remove_radius_outlier(nb_points=denoise_nb_points, radius=denoise_radius)
            pcd = pcd.select_by_index(ind)
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=denoise_nb_neighbors, std_ratio=denoise_std_ratio)
            pcd = pcd.select_by_index(ind)

        return pcd

    def vis(self, size=0.1):
        coordinate_frame = o3d.geometry.TriangleMesh()
        coordinate_frame.create_coordinate_frame(size=size)
        o3d.visualization.draw_geometries([self.pcd] + [coordinate_frame])

    @staticmethod
    def crop_by_xyz_limits(pcd,
                           xlim: Union[Tuple[float, float], List[float], np.ndarray],
                           ylim: Union[Tuple[float, float], List[float], np.ndarray],
                           zlim: Union[Tuple[float, float], List[float], np.ndarray]):
        assert len(xlim) == 2 and len(ylim) == 2 and len(zlim) == 2
        x_min, x_max = xlim
        y_min, y_max = ylim
        z_min, z_max = zlim

        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)

        mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)
        xyz = xyz[mask]
        rgb = rgb[mask]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(xyz)
        cropped_pcd.colors = o3d.utility.Vector3dVector(rgb)

        return cropped_pcd

    @staticmethod
    def crop_by_hsv_limits_reverse(pcd,
                                   hlim: Union[Tuple[float, float], List[float], np.ndarray],
                                   slim: Union[Tuple[float, float], List[float], np.ndarray],
                                   vlim: Union[Tuple[float, float], List[float], np.ndarray]):
        assert len(hlim) == 2 and len(slim) == 2 and len(vlim) == 2
        h_min, h_max = hlim
        s_min, s_max = slim
        v_min, v_max = vlim

        xyz = np.asarray(pcd.points)
        rgb = np.asarray(pcd.colors)
        hsv = cv2.cvtColor((np.expand_dims(rgb, 0) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).squeeze()

        mask = (hsv[:, 0] > h_min) & (hsv[:, 0] < h_max) & (hsv[:, 1] > s_min) & (hsv[:, 1] < s_max) & (hsv[:, 2] > v_min) & (hsv[:, 2] < v_max)
        mask = np.logical_not(mask)
        xyz = xyz[mask]
        rgb = rgb[mask]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(xyz)
        cropped_pcd.colors = o3d.utility.Vector3dVector(rgb)

        return cropped_pcd
