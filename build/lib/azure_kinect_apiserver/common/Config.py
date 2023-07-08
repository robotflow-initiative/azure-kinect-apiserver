"""
    "Record configuration: \n"
    f"\tcolor_format: {self._handle.color_format} \n\t(0:JPG, 1:NV12, 2:YUY2, 3:BGRA32)\n\n"
    f"\tcolor_resolution: {self._handle.color_resolution} \n\t(0:OFF, 1:720p, 2:1080p, 3:1440p, 4:1536p, 5:2160p, 6:3072p)\n\n"
    f"\tdepth_mode: {self._handle.depth_mode} \n\t(0:OFF, 1:NFOV_2X2BINNED, 2:NFOV_UNBINNED,3:WFOV_2X2BINNED, 4:WFOV_UNBINNED, 5:Passive IR)\n\n"
    f"\tcamera_fps: {self._handle.camera_fps} \n\t(0:5 FPS, 1:15 FPS, 2:30 FPS)\n\n"
    f"\tcolor_track_enabled: {self._handle.color_track_enabled} \n\t(True of False). If Color camera images exist\n\n"
    f"\tdepth_track_enabled: {self._handle.depth_track_enabled} \n\t(True of False). If Depth camera images exist\n\n"
    f"\tir_track_enabled: {self._handle.ir_track_enabled} \n\t(True of False). If IR camera images exist\n\n"
    f"\timu_track_enabled: {self._handle.imu_track_enabled} \n\t(True of False). If IMU samples exist\n\n"
    f"\tdepth_delay_off_color_usec: {self._handle.depth_delay_off_color_usec} us. \n\tDelay between the color image and the depth image\n\n"
    f"\twired_sync_mode: {self._handle.wired_sync_mode}\n\t(0:Standalone mode, 1:Master mode, 2:Subordinate mode)\n\n"
    f"\tsubordinate_delay_off_master_usec: {self._handle.subordinate_delay_off_master_usec} us.\n\tThe external synchronization timing.\n\n"
    f"\tstart_timestamp_offset_usec: {self._handle.start_timestamp_offset_usec} us. \n\tStart timestamp offset.\n\n"
"""
import datetime
import logging
import os.path as osp
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import yaml
from py_cli_interaction import (
    must_parse_cli_int,
    must_parse_cli_string,
    must_parse_cli_bool,
    must_parse_cli_sel,
    must_parse_cli_float,
)

from .functional import probe_device

logger = logging.getLogger(__name__)


class BaseCfg:
    def get_dict(self) -> Dict[Any, Any]:
        raise NotImplemented

    def load_dict(self, src: Dict[str, Any]) -> None:
        raise NotImplemented

    @staticmethod
    def configure_from_keyboard() -> bool:
        raise NotImplemented


@dataclass
class KinectCameraCfg(BaseCfg):
    index: int = -1
    sn: str = ''

    color_format: int = 0
    color_resolution: int = 0
    depth_mode: int = 0
    depth_delay_usec: int = 0
    camera_fps: int = 0
    imu: bool = False
    sync_mode: int = 0
    sync_delay_usec: int = 0
    exposure: int = -12

    __COLOR_RESOLUTION_CANDIDATES__ = ('OFF', '720p', '1080p', '1440p', '1536p', '2160p', '3072p')
    __DEPTH_MODE_CANDIDATES__ = (
        'OFF', 'NFOV_2X2BINNED', 'NFOV_UNBINNED', 'WFOV_2X2BINNED', 'WFOV_UNBINNED', 'Passive IR')
    __SYNC_MODE_CANDIDATES__ = ('Standalone', 'Master', 'Subordinate')

    def __post_init__(self):
        pass

    def get_dict(self) -> Dict[Any, Any]:
        return {
            'index': self.index,
            'sn': self.sn,
            'color_format': self.color_format,
            'depth_mode': self.depth_mode,
            'color_resolution': self.color_resolution,
            'depth_delay_usec': self.depth_delay_usec,
            'camera_fps': self.camera_fps,
            'imu': self.imu,
            'sync_mode': self.sync_mode,
            'sync_delay_usec': self.sync_delay_usec,
            'exposure': self.exposure,
        }

    def load_dict(self, src: Dict[str, Any]) -> None:
        self.index = src['index'] if src['index'] is not None else -1
        self.sn = src['sn'] if src['sn'] is not None else ''
        self.color_format = src['color_format'] if src['color_format'] is not None else 0
        self.color_resolution = src['color_resolution'] if src['color_resolution'] is not None else 5
        self.depth_mode = src['depth_mode'] if src['depth_mode'] is not None else 2
        self.depth_delay_usec = src['depth_delay_usec'] if src['depth_delay_usec'] is not None else 0
        self.camera_fps = src['camera_fps'] if src['camera_fps'] is not None else 30
        self.imu = src['imu'] if src['imu'] is not None else False
        self.sync_mode = src['sync_mode'] if src['sync_mode'] is not None else 0
        self.sync_delay_usec = src['sync_delay_usec'] if src['sync_delay_usec'] is not None else 0
        self.exposure = src['exposure'] if src['exposure'] is not None else -12

    @property
    def valid(self) -> bool:
        if any([
            (self.index < 0),
            (self.color_format < 0 or self.color_format > 3),
            (self.color_resolution < 0 or self.color_resolution > 6),
            (self.depth_mode < 0 or self.depth_mode > 5),
            (self.camera_fps not in [5, 15, 30]),
            (self.sync_mode < 0 or self.sync_mode > 2),
            (self.exposure < -12 or self.exposure > 1),
        ]):
            return False
        else:
            return True

    def get_args(self) -> Tuple[str, Optional[Exception]]:
        """Return the arguments for the kinect camera"""
        if not self.valid:
            return '', Exception('Invalid Kinect Camera Configuration')
        else:
            s = f"--device {self.index} " \
                f"-c {self.__COLOR_RESOLUTION_CANDIDATES__[self.color_resolution]} " \
                f"-d {self.__DEPTH_MODE_CANDIDATES__[self.depth_mode]} " \
                f"--depth-delay {self.depth_delay_usec} " \
                f"-r {str(int(self.camera_fps))} " \
                f"--imu {'ON' if self.imu else 'OFF'} " \
                f"--external-sync {self.__SYNC_MODE_CANDIDATES__[self.sync_mode]} " \
                f"--sync-delay {self.sync_delay_usec} "
            if -11 <= self.exposure <= 1:
                s += f"-e {self.exposure} "
            return s, None


class KinectSystemCfg(BaseCfg):
    config_path: str = ''
    data_path: str = './azure_kinect_data'
    exec_path: str = 'k4arecorder'

    api_port: int = 0
    api_interface: str = '0.0.0.0'
    api_enabled: bool = True
    debug: bool = False

    length_sec: int = 0
    camera_options: List[KinectCameraCfg] = []

    marker_valid_ids: List[str]
    marker_length_m: float
    marker_type: str

    def __init__(self, config_path: str):
        self.config_path = config_path
        try:
            cfg_dict = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
        except Exception as e:
            logger.warning(f'failed to load Azure Kinect Config: {e}')
            return

        if 'azure_kinect' in cfg_dict.keys():
            self.load_dict(cfg_dict['azure_kinect'])
        else:
            logger.warning('failed to load Azure Kinect Config: No azure_kinect key found in config file')

    def get_dict(self) -> Dict[Any, Any]:
        return {
            'data_path': self.data_path,
            'exec_path': self.exec_path,
            'api': {
                'port': self.api_port,
                'interface': self.api_interface,
            },
            'debug': self.debug,
            'length_sec': self.length_sec,
            'camera_options': [cfg.get_dict() for cfg in self.camera_options],
            'marker': {
                'valid_ids': self.marker_valid_ids,
                'length': self.marker_length_m,
                'type': self.marker_type,
            }
        }

    def load_dict(self, src: Dict[str, Any]) -> None:
        self.data_path = src['data_path'] if 'data_path' in src.keys() else './azure_kinect_data'
        self.exec_path = src['exec_path'] if 'exec_path' in src.keys() else 'k4arecorder'
        if 'api' not in src.keys():
            self.api_enabled = False
        else:
            self.api_port = src['api']['port'] if 'port' in src['api'].keys() else 8080
            self.api_interface = src['api']['interface'] if 'interface' in src['api'].keys() else '0.0.0.0'
        self.debug = src['debug'] if 'debug' in src.keys() else False
        self.length_sec = src['length_sec'] if 'length_sec' in src.keys() else 0
        if 'camera_options' in src.keys():
            self.camera_options = []
            for opt in src['camera_options']:
                new_opt = KinectCameraCfg()
                new_opt.load_dict(opt)
                self.camera_options.append(new_opt)
        else:
            self.camera_options = []

        if 'marker' in src.keys():
            self.marker_valid_ids = src['marker']['valid_ids'] if 'valid_ids' in src['marker'].keys() else []
            self.marker_length_m = src['marker']['length'] if 'length' in src['marker'].keys() else -1
            self.marker_type = src['marker']['type'] if 'type' in src['marker'].keys() else ''

    @property
    def valid(self) -> bool:
        if any(
                [
                    (self.api_enabled and (self.api_port <= 0 or self.api_port > 65535)),
                    (self.length_sec < 0),
                    (not len(self.camera_options) > 0),
                    (not all([cfg.valid for cfg in self.camera_options]))
                ]
        ):
            return False
        else:
            return True

    def get_commands(self, tag: str = None) -> Tuple[List[str], Optional[Exception]]:
        """Return the arguments for the kinect camera"""
        if tag is None or tag == '':
            tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        record_path = osp.join(self.data_path, tag, 'kinect')

        if not self.valid:
            return [], Exception('Invalid Kinect System Configuration')
        else:
            args_list = []
            prefix = f"-l {self.length_sec}" if self.length_sec > 0 else ''
            for opt in filter(lambda x: x.sync_mode != 1, self.camera_options):
                postfix = f"{osp.join(record_path, f'{opt.sn}.mkv' if opt.sn is not None and opt.sn != '' else f'output_{opt.index}.mkv')}"
                args, err = opt.get_args()
                if err is not None:
                    return [], err
                else:
                    args_list.append(prefix + args + postfix)

            for opt in filter(lambda x: x.sync_mode == 1, self.camera_options):
                postfix = f"{osp.join(record_path, f'{opt.sn}.mkv' if opt.sn is not None and opt.sn != '' else f'output_{opt.index}.mkv')}"
                args, err = opt.get_args()
                if err is not None:
                    return [], err
                else:
                    args_list.append(prefix + args + postfix)
            return args_list, None

    def get_command_with_exec(self, tag: str = None) -> Tuple[List[str], Optional[Exception]]:
        args_list, err = self.get_commands(tag)
        if err is not None:
            return args_list, err
        else:
            return [f"{self.exec_path} {x}" for x in args_list], None

    @staticmethod
    def configure_from_keyboard() -> bool:
        output_dir = must_parse_cli_string('Output Directory', './config.yaml')
        data_path = must_parse_cli_string('Data Path', './azure_kinect_data')
        exec_path = must_parse_cli_string('Exec Path', 'k4arecorder')
        api_port = must_parse_cli_int('API Port', 1, 65535, 8080)
        api_interface = must_parse_cli_string('API Interface', '0.0.0.0')
        debug = must_parse_cli_bool('Debug', False)

        cams, err = probe_device(exec_path)
        if err is not None:
            print(f'error probing devices using {exec_path}: {err}')
            return False
        else:
            color_format = 0
            color_resolution = must_parse_cli_sel('Color Resolution', ['OFF', '720p', '1080p', '1440p', '1536p', '2160p', '3072p'], default_value=5)
            depth_mode = must_parse_cli_sel('Depth Mode', ['OFF', 'NFOV_2X2BINNED', 'NFOV_UNBINNED', 'WFOV_2X2BINNED', 'WFOV_UNBINNED', 'PASSIVE_IR'], default_value=2)
            depth_delay_usec = must_parse_cli_int('Depth Delay (us)', default_value=0)
            camera_fps_sel = must_parse_cli_sel('Camera FPS', [5, 15, 30], default_value=2)
            camera_fps = [5, 15, 30][camera_fps_sel]
            imu = must_parse_cli_bool('IMU', True)
            sync_mode = 2
            sync_delay_usec = must_parse_cli_int('Sync Delay (us)', default_value=0)
            exposure = must_parse_cli_int('Exposure, -12 for auto exposure', min=-12, max=1, default_value=-12)
            marker_length = must_parse_cli_float('Marker Length (m)', default_value=0.01)
            _marker_type_candidates = [
                'DICT_4X4_100',
                'DICT_4X4_1000',
                'DICT_4X4_250',
                'DICT_4X4_50',
                'DICT_5X5_100',
                'DICT_5X5_1000',
                'DICT_5X5_250',
                'DICT_5X5_50',
                'DICT_6X6_100',
                'DICT_6X6_1000',
                'DICT_6X6_250',
                'DICT_6X6_50',
                'DICT_7X7_100',
                'DICT_7X7_1000',
                'DICT_7X7_250',
                'DICT_7X7_50',
            ]
            _marker_type_sel = must_parse_cli_sel('Marker Type', _marker_type_candidates, default_value=0)
            marker_type = _marker_type_candidates[_marker_type_sel]

            camera_options = [
                {
                    'index': x['Index'],
                    'sn': str(x['Serial']),
                    'color_format': color_format,
                    'color_resolution': color_resolution,
                    'depth_mode': depth_mode,
                    'depth_delay_usec': depth_delay_usec,
                    'camera_fps': camera_fps,
                    'imu': imu,
                    'sync_mode': sync_mode,
                    'sync_delay_usec': sync_delay_usec,
                    'exposure': exposure
                } for x in cams
            ]

            cfg_dict = {
                'azure_kinect': {
                    'data_path': data_path,
                    'exec_path': exec_path,
                    'api': {
                        'port': api_port,
                        'interface': api_interface,
                    },
                    'debug': debug,
                    'camera_options': camera_options,
                    'marker': {
                        'valid_ids': [],
                        'length': marker_length,
                        'type': marker_type
                    }
                }
            }

            with open(output_dir, 'w') as f:
                yaml.dump(cfg_dict, f)


def generate_script_powershell(cfg: KinectSystemCfg, tag: str = None):
    args_list, err = cfg.get_commands(tag)
    if err is not None:
        raise err
    else:
        for i in range(0, len(cfg.camera_options), -1):
            if cfg.camera_options[i].sync_mode == 2:
                args_list.insert(0, args_list.pop(i))

        template = 'Start-Process {} -ArgumentList "{}"'.format(cfg.exec_path, '{}')
        lines = []
        for args in args_list:
            lines.extend([template.format(args), 'Start-Sleep -s 1'])
        content = '\n'.join(lines)
        print(content)
        return content


def generate_script_bash(cfg: KinectSystemCfg, tag: str = None):
    args_list, err = cfg.get_commands(tag)
    if err is not None:
        raise err
    else:
        for i in range(0, len(cfg.camera_options), -1):
            if cfg.camera_options[i].sync_mode == 2:
                args_list.insert(0, args_list.pop(i))

        template = '{} "{}" &'.format(cfg.exec_path, '{}')
        lines = []
        for args in args_list:
            lines.extend([template.format(args), 'sleep 1'])
        content = '\n'.join(lines) + '\n'
        content += 'K4A_EXEC="{}"'.format(cfg.exec_path)
        trapper = """
trap kill_background SIGINT
kill_background() {
  echo "Killing processes"
  ps -aux | grep "$K4A_EXEC" | awk '{print $2}' | xargs kill -9
  exit
}
echo "Press Ctrl-C to stop recording"
wait
"""
        content += trapper
        print(content)
        return content


if __name__ == '__main__':
    c = KinectSystemCfg('manifests/azure_kinect_config/azure_kinect_config.yaml')
    [print(args) for args in c.get_commands('test')]
    print('------------------- Powershell -------------------')
    generate_script_powershell(c, 'test')
    print('------------------- Bash -------------------')
    generate_script_bash(c, 'test')
    print('------------------- End -------------------')
