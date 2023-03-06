import os.path as osp
import json
import logging
import numpy as np
import sys
import os
from typing import Tuple, List, Dict, Union, Any, Optional
import dataclasses
import pandas as pd
import cv2

from azure_kinect_apiserver.common import MulticalCameraInfo

STATUS_INVALID = -1
STATUS_RAW = 0
STATUS_DECODED = 1
STATUS_ANALYZED = 2


@dataclasses.dataclass()
class AzureKinectDataset:
    kinect_path: str
    multical_calibration_path: str = None
    multical_calibration: Optional[MulticalCameraInfo] = None
    status: int = STATUS_INVALID
    metadata: Dict[str, Any] = None
    calibration: Dict[str, Any] = None
    master_camera_name: str = None
    camera_name_list: List[str] = None
    camera_metadata_collection: Dict[str, pd.DataFrame] = None
    camera_metadata_records: Dict[str, List[Dict[str, Any]]] = None
    color_img_path_collection: Dict[str, List[str]] = None
    depth_img_path_collection: Dict[str, List[str]] = None
    camera_parameter_collection: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        if self.multical_calibration_path is not None:
            self.multical_calibration = MulticalCameraInfo(self.multical_calibration_path)
            if not self.multical_calibration.valid:
                self.multical_calibration = None
        self.status = self._probe_dataset_status()

    def __len__(self):
        if self.status < STATUS_DECODED:
            return 0
        else:
            return len(self.color_img_path_collection[self.camera_name_list[0]])

    def __getitem__(self, item):
        if self.status < STATUS_DECODED:
            return None
        else:
            return ({
                        cam_name: (self.camera_metadata_records[cam_name][item]) for cam_name in self.camera_name_list
                    },
                    {
                        cam_name: (self.color_img_path_collection[cam_name][item], self.depth_img_path_collection[cam_name][item]) for cam_name in self.camera_name_list
                    })

    def _probe_dataset_status(self) -> int:
        status = STATUS_INVALID
        if osp.exists(self.kinect_path) and len(os.listdir(self.kinect_path)) > 0:
            status = STATUS_RAW
            dir_content = os.listdir(self.kinect_path)
            if not all([osp.isfile(osp.join(self.kinect_path, file)) for file in dir_content]):
                if osp.exists(osp.join(self.kinect_path, 'meta.json')):
                    status = STATUS_DECODED
                else:
                    status = STATUS_INVALID
        return status

    def load(self) -> Optional[Exception]:
        if self.status == STATUS_INVALID:
            return Exception("invalid dataset")
        if self.status > STATUS_INVALID:
            possible_calibration = osp.join(self.kinect_path, 'calibration.json')
            if osp.exists(possible_calibration):
                self.set_calibration(possible_calibration)
        if self.status > STATUS_RAW:
            self.metadata = json.load(open(osp.join(self.kinect_path, 'meta.json')))

            self.master_camera_name = list(filter(lambda x: self.metadata['recordings'][x]['is_master'], self.metadata['recordings'].keys()))[0]
            self.camera_name_list = list(self.metadata['recordings'].keys())

            self.camera_metadata_collection = {
                cam_name: pd.read_csv(osp.join(self.kinect_path, cam_name, 'meta.csv')) for cam_name in self.camera_name_list
            }
            self.camera_metadata_records = {
                cam_name: self.camera_metadata_collection[cam_name].to_dict('records') for cam_name in self.camera_name_list
            }

            self.color_img_path_collection = {
                cam_name: list(
                    map(lambda x: osp.join(self.kinect_path, cam_name, 'color', str(x) + '.jpg'), cam_meta['basename'].values)
                ) for cam_name, cam_meta in self.camera_metadata_collection.items()
            }
            self.depth_img_path_collection = {
                cam_name: list(
                    map(lambda x: osp.join(self.kinect_path, cam_name, 'depth', str(x) + '.png'), cam_meta['basename'].values)
                ) for cam_name, cam_meta in self.camera_metadata_collection.items()
            }
            self.camera_parameter_collection = {
                cam_name: json.load(open(osp.join(self.kinect_path, cam_name, 'calibration.kinect.json'))) for cam_name in self.camera_name_list
            }

    def set_calibration(self, path) -> Optional[Exception]:
        self.multical_calibration_path = path
        self.multical_calibration = MulticalCameraInfo(path)
        if not self.multical_calibration.valid:
            self.multical_calibration = None
            return Exception("Invalid calibration")
        else:
            return None

    def get_system_action_start_timestamp(self):
        return float(self.metadata['system_action_start_timestamp'])

    def get_system_timestamp_offset(self):
        return float(self.metadata['system_timestamp_offset'])

    def is_valid_rvec_tvec(self, serial, rvec, tvec):
        origin = np.eye(4)
        # origin[:3, :3] = cv2.Rodrigues(rvec)[0]
        origin[:3, 3] = tvec
        cam_extrinsics = self.multical_calibration.get_extrinsic(serial)
        system_extrinsic = self.multical_calibration.get_realworld_rot_trans()
        real_6dof = system_extrinsic @ cam_extrinsics @ origin

        return real_6dof[0][3] > 0


if __name__ == '__main__':
    dataset = AzureKinectDataset(r'C:\Users\robotflow\Desktop\fast-cloth-pose\data\20230303_205124\kinect')
    dataset.load()
    print(dataset.metadata)
