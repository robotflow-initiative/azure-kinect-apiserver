import copy
import dataclasses
import logging
from typing import Union, List, Optional, Dict, Tuple

import cv2
import cv2.aruco as aruco
import numpy as np
import open3d as o3d

logger = logging.getLogger("azure_kinect_apiserver.decoder.aruco")


@dataclasses.dataclass()
class ArucoDetectHelper:
    marker_length: float  # in meter
    aruco_type: int  # e.g. cv2.aruco.DICT_5X5_1000
    camera_matrix: Union[List[List[float]], np.ndarray] = None  # [[ fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    camera_distort: Optional[Union[List[float], np.ndarray]] = None  # [ 0.1927826544288516, -0.34972530095573834, 0.011612480526787846, -0.00393533140166019, -2.9216752723525734 ]
    dof_cache: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None
    font: int = cv2.FONT_HERSHEY_SIMPLEX

    def __post_init__(self):
        self.marker_points = np.array([[-self.marker_length / 2, self.marker_length / 2, 0],
                                       [self.marker_length / 2, self.marker_length / 2, 0],
                                       [self.marker_length / 2, -self.marker_length / 2, 0],
                                       [-self.marker_length / 2, -self.marker_length / 2, 0]])
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        if self.camera_distort is not None:
            if isinstance(self.camera_distort, list):
                self.camera_distort = np.array(self.camera_distort)
            self.camera_distort = self.camera_distort.squeeze()
            assert self.camera_distort.shape[0] in [4, 5, 8, 12, 14]

        if self.camera_matrix is not None:
            if isinstance(self.camera_matrix, list):
                self.camera_matrix = np.array(self.camera_matrix)
            assert self.camera_matrix.shape == (3, 3)

        self.dof_cache = {}
        self.dof_freq = {}

    @staticmethod
    def depth2xyz(u, v, depth, K):
        x = (u - K[0, 2]) * depth / K[0, 0]
        y = (v - K[1, 2]) * depth / K[1, 1]
        return np.array([x, y, depth]).reshape(3, 1)

    @staticmethod
    def apply_polygon_mask_color(img: np.ndarray, polygons: List[np.ndarray]):
        mask = np.zeros(img.shape, dtype=img.dtype)
        for polygon in polygons:
            polygon = polygon.reshape(-1, 2).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, (255, 255, 255))
        res = cv2.bitwise_and(img, mask)
        return res

    @staticmethod
    def apply_polygon_mask_depth(img: np.ndarray, polygons: List[np.ndarray]):
        mask = np.zeros(img.shape, dtype=img.dtype)
        for polygon in polygons:
            polygon = polygon.reshape(-1, 2).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, (np.iinfo(img.dtype).max))
        res = cv2.bitwise_and(img, mask)
        return res

    @classmethod
    def create_roi_masked_point_cloud(cls,
                                      color_frame: np.ndarray,
                                      depth_frame: np.ndarray,
                                      corners: List[np.ndarray],
                                      camera_matrix: np.ndarray) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[Exception]]:
        color_frame_masked = cls.apply_polygon_mask_color(color_frame, [corner.reshape(-1, 2).astype(np.int32) for corner in corners])
        depth_frame_masked = cls.apply_polygon_mask_depth(depth_frame, [corner.reshape(-1, 2).astype(np.int32) for corner in corners])

        if depth_frame_masked.sum() > 0:
            color_frame_o3d = o3d.geometry.Image(cv2.cvtColor(color_frame_masked.astype(np.uint8), cv2.COLOR_BGRA2RGB))
            depth_frame_o3d = o3d.geometry.Image(depth_frame_masked.astype(np.uint16))
            rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
                color_frame_o3d,
                depth_frame_o3d,
                1000.,
                2.,
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
                image=rgbd_image,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    depth_frame_masked.shape[1],
                    depth_frame_masked.shape[0],
                    camera_matrix[0][0],
                    camera_matrix[1][1],
                    camera_matrix[0][2],
                    camera_matrix[1][2],
                ),
                extrinsic=np.eye(4),
            )
            if len(pcd.points) < 4:
                return None, Exception("no enough depth")
            else:
                return pcd, None
        else:
            return None, Exception("no depth")

    def reject_false_rotation(self,
                              color_frame,
                              depth_frame,
                              rvec: np.ndarray,
                              tvec: np.ndarray,
                              corners: np.ndarray,
                              debug=False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Exception]]:
        pcd, err = self.create_roi_masked_point_cloud(color_frame, depth_frame, [corners], self.camera_matrix)
        if err is None:
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=self.marker_length, max_nn=30)
            )
            position = np.asarray(pcd.points).mean(axis=0)
            normals = np.asarray(pcd.normals)
            normal_vec = normals.mean(axis=0)
            normal_vec /= np.linalg.norm(normal_vec)

            z_axis = np.array([[0], [0], [1]])
            candidates = [cv2.Rodrigues(x)[0] @ z_axis for x in rvec]
            score = np.array([float(abs(np.dot(x.T, normal_vec))) for x in candidates])

            if debug:
                arrow = o3d.geometry.TriangleMesh().create_arrow(
                    cylinder_radius=self.marker_length / 20,
                    cone_radius=self.marker_length / 10,
                    cylinder_height=self.marker_length / 2,
                    cone_height=self.marker_length / 2,
                )
                R, _ = cv2.Rodrigues(np.cross(normal_vec, [0, 0, 1]))
                # arrow.scale(3, center=arrow.get_center())
                arrow.translate(position)
                arrow.rotate(R, center=arrow.get_center())

                o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=[0, 0, 0]), pcd, arrow])
                # o3d.visualization.draw_geometries([o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=[0, 0, 0]), arrow])

            return rvec[score.argmax()], tvec[score.argmax()], None
        else:
            return None, None, Exception("no depth")

    def preview_one_frame(self, color_frame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        output_color_frame = copy.deepcopy(color_frame)
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        if ids is not None:
            aruco.drawDetectedMarkers(output_color_frame, corners, ids)
            logger.info(f"detected {len(corners)} markers, with ids {ids}")
        return output_color_frame, corners, ids

    # num = 0
    def process_one_frame(self,
                          color_frame,
                          depth_frame,
                          depth_scale: float = 1000.,
                          undistort=True,
                          debug=False):
        undistort = False if self.camera_distort is None else undistort
        if self.camera_matrix is None:
            return None, None, None, Exception("no camera matrix")

        if undistort:
            color_frame_undistort = cv2.undistort(color_frame, self.camera_matrix, self.camera_distort)
            depth_frame_undistort = cv2.undistort(depth_frame, self.camera_matrix, self.camera_distort)
        else:
            color_frame_undistort = color_frame
            depth_frame_undistort = depth_frame
        output_color_frame = copy.deepcopy(color_frame_undistort)
        depth_frame_undistort = cv2.medianBlur(depth_frame_undistort, 5)

        gray = cv2.cvtColor(color_frame_undistort, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)

        res = {}

        # id found
        if ids is not None:
            aruco.drawDetectedMarkers(output_color_frame, corners, ids) if debug else None
            for marker_id, corner in zip(ids, corners):
                ret, rvecs, tvecs, scores = cv2.solvePnPGeneric(self.marker_points, corner[0], self.camera_matrix, self.camera_distort, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # ret, rvecs, tvecs = cv2.solvePnP(self.marker_points, corner[0], self.camera_matrix, np.array([]), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # print(rvec)
                if ret:
                    rvec, tvec, err = self.reject_false_rotation(color_frame_undistort, depth_frame_undistort, rvecs, tvecs, corner[0])
                    if err is None:
                        u, v = corner[0][0].astype(int).tolist()
                        xyz = self.depth2xyz(u, v, depth_frame_undistort[v][u] / depth_scale, self.camera_matrix)  # 1000 is kinect depth scale
                        res[str(marker_id[0])] = (rvec, xyz, err is None)

            return res, output_color_frame, depth_frame_undistort, None if len(res) > 0 else Exception("no marker found")
        else:
            return None, output_color_frame, depth_frame_undistort, Exception("no marker found")

    def vis_3d(self, detection: Dict[str, Tuple[np.ndarray, np.ndarray]], color_frame=None, depth_frame=None):
        marker_meshes = [o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])]
        for marker_id, (rvec, tvec, is_unique) in detection.items():
            if is_unique:
                new_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=tvec)
                R, _ = cv2.Rodrigues(rvec)
                new_mesh.rotate(R, center=tvec)
                marker_meshes.append(new_mesh)
            else:
                for i in range(len(rvec)):
                    new_mesh = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.1, origin=tvec[i])
                    R, _ = cv2.Rodrigues(rvec[i])
                    new_mesh.rotate(R, center=tvec[i])
                    marker_meshes.append(new_mesh)
        if color_frame is not None:
            color_frame_o3d = o3d.geometry.Image(cv2.cvtColor(color_frame.astype(np.uint8), cv2.COLOR_BGRA2RGB))
            depth_frame_o3d = o3d.geometry.Image(depth_frame.astype(np.uint16))

            rgbd_image = o3d.geometry.RGBDImage().create_from_color_and_depth(
                color_frame_o3d,
                depth_frame_o3d,
                1000.,
                2.,
                convert_rgb_to_intensity=False
            )

            pcd = o3d.geometry.PointCloud().create_from_rgbd_image(
                image=rgbd_image,
                intrinsic=o3d.camera.PinholeCameraIntrinsic(
                    depth_frame.shape[1],
                    depth_frame.shape[0],
                    self.camera_matrix[0, 0],
                    self.camera_matrix[1, 1],
                    self.camera_matrix[0, 2],
                    self.camera_matrix[1, 2],
                ),
                extrinsic=np.eye(4),
            )
            marker_meshes.append(pcd)
        o3d.visualization.draw_geometries(marker_meshes)
        return marker_meshes

    def vis_2d(self, detection: Dict[str, Tuple[np.ndarray, np.ndarray]], color_frame):
        for marker_id, (rvec, tvec, is_unique) in detection.items():
            if is_unique:
                cv2.drawFrameAxes(color_frame, self.camera_matrix, self.camera_distort[:5], rvec, tvec, 0.03, 2)
            else:
                cv2.drawFrameAxes(color_frame, self.camera_matrix, self.camera_distort[:5], rvec[0], tvec[1], 0.03, 2)
        cv2.putText(color_frame, "Id: " + str(list(detection.keys())), (0, 64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        scale = int(color_frame.shape[0] / 480)
        cv2.imshow("frame", color_frame[::scale, ::scale, :])
        key = cv2.waitKey(0)
        return color_frame
