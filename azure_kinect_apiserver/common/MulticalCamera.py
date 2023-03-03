import json
import logging
from typing import Tuple
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("azure_kinect_apiserver.common.multical_camera")


# noinspection PyPep8Naming
class MulticalCameraInfo:
    valid: bool
    master_cam: str
    camera_info: dict

    def __init__(self, json_path):
        self.json_path = json_path
        self.valid = self.load()
        _master_cam = [k for k in self.camera_info['camera_poses'].keys() if '_to_' not in k]
        assert (len(_master_cam) == 1), "master cam should be unique"
        self.master_cam = _master_cam[0]

    def load(self):
        try:
            self.camera_info = json.load(open(self.json_path))
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
        if serial == self.master_cam:
            return np.eye(4, dtype=float)

        R = self.camera_info['camera_poses'][serial + '_to_' + self.master_cam]['R']
        T = self.camera_info['camera_poses'][serial + '_to_' + self.master_cam]['T']
        camera_extrinsic = np.eye(4, dtype=float)
        camera_extrinsic[:3, :3] = np.array(R)
        camera_extrinsic[:3, 3] = np.array(T)

        camera_extrinsic = np.linalg.inv(camera_extrinsic)
        return camera_extrinsic

    def get_resolution(self, serial) -> Tuple[int, int]:
        """
        :param serial: camera's serial number
        :return: (width, height)
        """
        return tuple(self.camera_info['cameras'][serial]['image_size'])
