import dataclasses
import json
import os
import os.path as osp
import glob
import pickle
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import open3d as o3d

from .MulticalCamera import MulticalCameraInfo

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
    point_cloud_path_collection: List[Dict[str, str]] = None
    marker_detections: Dict[str, np.ndarray] = None
    kinect_timestamps_unix: np.ndarray = None

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
            return len(self.color_img_path_collection[self.master_camera_name])

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

    @property
    def start_idx(self):
        if self.status < STATUS_DECODED:
            return 0
        else:
            return np.where(self.kinect_timestamps_unix > self.get_system_action_start_timestamp())[0][0]

    def _probe_dataset_status(self) -> int:
        status = STATUS_INVALID
        if osp.exists(self.kinect_path) and len(os.listdir(self.kinect_path)) > 0:
            status = STATUS_RAW
            dir_content = os.listdir(self.kinect_path)
            if not all([osp.isfile(osp.join(self.kinect_path, file)) for file in dir_content]):
                if osp.exists(osp.join(self.kinect_path, 'meta.json')):
                    status = STATUS_DECODED

                    pcd_path = osp.join(self.kinect_path, 'pcd_s3')
                    if osp.exists(pcd_path) and osp.isdir(pcd_path) and len(os.listdir(pcd_path)) > 0:
                        status = STATUS_ANALYZED

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
            self.kinect_timestamps_unix = self.camera_metadata_collection[self.master_camera_name]['color_dev_ts_usec'].to_numpy() * 1e-6 + self.get_system_timestamp_offset()

        if self.status > STATUS_DECODED:
            point_cloud_path_list = glob.glob(osp.join(self.kinect_path, 'pcd_s3', "*.ply"))
            point_cloud_path_basename_no_ext = list(map(lambda x: osp.splitext(osp.basename(x))[0], point_cloud_path_list))
            self.point_cloud_path_collection = [{} for _ in range(len(self.color_img_path_collection[self.master_camera_name]))]
            for _, (value, path) in enumerate(zip(point_cloud_path_basename_no_ext, point_cloud_path_list)):
                idx, cam_name = value.split('_')
                self.point_cloud_path_collection[int(idx)][cam_name] = path

            self.marker_detections = pickle.load(open(osp.join(self.kinect_path, 'detection_result_s2.pkl'), 'rb'))

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


@dataclasses.dataclass()
class ArizonForceDataset:
    # FIXME: move this class to another repository
    path: str
    status: bool = False
    timestamp_offset: float = 0.0
    recordings: dict = dataclasses.field(default_factory=dict)
    interpolated_force: Optional[list] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        recording_csv_paths = glob.glob(osp.join(self.path, '*.csv'))
        if len(recording_csv_paths) == 0:
            self.status = False
            self.recordings = {}
            self.interpolated_force = None
        else:
            self.status = True

            device_names = list(map(lambda x: osp.splitext(osp.basename(x))[0], recording_csv_paths))
            self.recordings = {
                device_name: pd.read_csv(osp.join(self.path, device_name + '.csv')) for device_name in device_names
            }
            self.interpolated_force = None

    def __len__(self):
        if not self.status:
            return 0
        else:
            if self.interpolated_force is not None:
                return len(self.interpolated_force[self.device_names[0]])
            else:
                return len(self.recordings[self.device_names[0]])

    def __getitem__(self, item):
        if self.interpolated_force is not None:
            return {device_name: self.interpolated_force[device_name][item] for device_name in self.device_names}
        else:
            return {device_name: self.recordings[device_name].iloc[item]['f'] for device_name in self.device_names}

    @property
    def device_names(self):
        return list(self.recordings.keys())

    def compute_interpolated_force(self, timestamps: np.ndarray = None) -> Optional[dict]:
        if timestamps is None:
            self.interpolated_force = {device_name: self.recordings[device_name]['f'] for device_name in self.device_names}
        else:
            self.interpolated_force = {device_name: np.interp(timestamps, self.recordings[device_name]['sys_ts_ns'] * 1e-9, self.recordings[device_name]['f']) for device_name in self.device_names}
        pass


@dataclasses.dataclass()
class JointPointCloudDataset:
    path: str
    time_origin_arizon_minus_kinect_in_sec: float = 0.0

    def __post_init__(self):
        self.kinect_dataset = AzureKinectDataset(osp.join(self.path, 'kinect'))
        self.arizon_dataset = ArizonForceDataset(osp.join(self.path, 'arizon'))
        self.kinect_dataset.load()
        self.arizon_dataset.compute_interpolated_force(self.kinect_dataset.kinect_timestamps_unix + self.time_origin_arizon_minus_kinect_in_sec)

        if not self.arizon_dataset.status:
            self.arizon_dataset = None
        else:
            assert len(self.kinect_dataset) == len(self.arizon_dataset) == len(self.kinect_dataset.kinect_timestamps_unix)
        self._length = len(self.kinect_dataset)
        self.timestamps = self.kinect_dataset.kinect_timestamps_unix

    def __len__(self):
        return self._length

    def __getitem__(self, idx) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], bool, bool, bool, float]:
        point_cloud_paths = self.kinect_dataset.point_cloud_path_collection[idx]
        marker_detections = {k: v[idx]
                             for k, v in self.kinect_dataset.marker_detections.items()}
        force = self.arizon_dataset[idx] if self.arizon_dataset is not None else {None: None}
        if len(point_cloud_paths) == 0:
            point_cloud_collection = {}
        else:
            point_cloud_collection = {k: o3d.io.read_point_cloud(v) for k, v in point_cloud_paths.items() if v is not None}

        return (
            point_cloud_collection,
            marker_detections,
            force,
            any(
                [
                    v is not None for v in point_cloud_paths.values()
                ]
            ),
            any(
                [
                    v is not None for v in marker_detections.values()
                ]
            ),
            any(
                [
                    v is not None for v in force.values()
                ]
            ),
            self.timestamps[idx]
        )

    @property
    def start_idx(self):
        return self.kinect_dataset.start_idx


if __name__ == '__main__':
    dataset = AzureKinectDataset(r'C:\Users\robotflow\Desktop\fast-cloth-pose\data\20230303_205124\kinect')
    dataset.load()
    print(dataset.metadata)
