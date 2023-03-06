import json
import logging
import os.path as osp
from typing import Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("azure_kinect_apiserver.common.MultiCamera")


# noinspection PyPep8Naming
class MulticalCameraInfo:
    valid: bool
    master_cam: str
    camera_info: dict

    _cached_realworld_transmat: None
    _cached_camera_extrinsics: dict

    def __init__(self, json_path):
        if osp.isdir(json_path):
            json_path = osp.join(json_path, 'calibration.json')
        self.json_path = json_path
        self.valid = self.load()
        if self.valid:
            _master_cam = [k for k in self.camera_info['camera_poses'].keys() if '_to_' not in k]
            assert (len(_master_cam) == 1), "master cam should be unique"
            self.master_cam = _master_cam[0]
        else:
            self.master_cam = ''

        self._cached_realworld_transmat = None
        self._cached_camera_extrinsics = None

    def load(self):
        try:
            self.camera_info = json.load(open(self.json_path))
            if any([
                'cameras' not in self.camera_info,
                'camera_poses' not in self.camera_info,
                len(self.camera_info['cameras']) == 0,
                len(self.camera_info['camera_poses']) == 0,
            ]):
                logger.error(f"failed to load {self.json_path}, invalid json")
                return False
            else:
                return True
        except Exception as e:
            logger.error(f"failed to load {self.json_path}, {e}")
            return False

    def get_intrinsic(self, serial):
        # Here we use the intrinsic from the multi-cal
        return np.array(self.camera_info['cameras'][serial]['K'])

    def get_distort(self, serial):
        return np.array(self.camera_info['cameras'][serial]['dist'])

    def get_extrinsic(self, serial):
        if self._cached_camera_extrinsics is not None:
            return self._cached_camera_extrinsics[serial]

        else:
            _cache = {}
            for s in self.camera_info['cameras'].keys():
                if s == self.master_cam:
                    _cache[s] = np.eye(4, dtype=float)
                else:

                    R = self.camera_info['camera_poses'][s + '_to_' + self.master_cam]['R']
                    T = self.camera_info['camera_poses'][s + '_to_' + self.master_cam]['T']
                    camera_extrinsic = np.eye(4, dtype=float)
                    camera_extrinsic[:3, :3] = np.array(R)
                    camera_extrinsic[:3, 3] = np.array(T)
                    camera_extrinsic = np.linalg.inv(camera_extrinsic)
                    _cache[s] = camera_extrinsic
            self._cached_camera_extrinsics = _cache
            return _cache[serial]

    def get_resolution(self, serial) -> Tuple[int, int]:
        """
        :param serial: camera's serial number
        :return: (width, height)
        """
        return tuple(self.camera_info['cameras'][serial]['image_size'])

    def get_icp_extrinsic_refinement(self, serial):
        """
        :param serial: camera's serial number
        :return:
        """
        return np.array(
            self.camera_info['icp_extrinsic_refinement'][serial]
        ) if 'icp_extrinsic_refinement' in self.camera_info else None

    def get_realworld_rot_trans(self):
        """
        :param serial: camera's serial number
        :return: (R, T) in real world coordinate
        """
        return {
            "R": np.array(self.camera_info['realworld_rot_trans']["R"]),
            "T": np.array(self.camera_info['realworld_rot_trans']["T"])
        } if 'realworld_rot_trans' in self.camera_info else None

    def get_realworld_transmat(self):
        """
        :param serial: camera's serial number
        :return: (R, T) in real world coordinate
        """
        if self._cached_realworld_transmat is not None:
            return self._cached_realworld_transmat
        else:
            res = np.eye(4)
            res[:3, :3] = np.array(self.camera_info['realworld_rot_trans']["R"])
            res[:3, 3] = np.array(self.camera_info['realworld_rot_trans']["T"]).squeeze()
            self._cached_realworld_transmat = res
            return res

    def get_workspace_limits(self, output_type=dict):
        if output_type == dict:
            return {
                "x_lim": np.array(self.camera_info['workspace_limits']["x_lim"]),
                "y_lim": np.array(self.camera_info['workspace_limits']["y_lim"]),
                "z_lim": np.array(self.camera_info['workspace_limits']["z_lim"]),
            } if 'workspace_limits' in self.camera_info else None
        elif output_type == tuple:
            return (np.array(self.camera_info['workspace_limits']["x_lim"]),
                    np.array(self.camera_info['workspace_limits']["y_lim"]),
                    np.array(self.camera_info['workspace_limits']["z_lim"]))
        else:
            raise NotImplementedError

    def save_to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.camera_info, f, indent=4)
