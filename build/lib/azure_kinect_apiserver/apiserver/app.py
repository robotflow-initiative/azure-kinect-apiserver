import datetime
import json
import logging
import os
import os.path as osp
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, List
import base64

import cv2
import numpy as np

from azure_kinect_apiserver.common import KinectSystemCfg, probe_device, MulticalCameraInfo
from azure_kinect_apiserver.thirdparty import pykinect

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('azure_kinect_apiserver.app')

pykinect.initialize_libraries()


# noinspection PyShadowingNames
class Application:
    option: KinectSystemCfg = None
    state: Dict[str, bool] = None
    lock: threading.RLock = None
    logger: logging.Logger = None

    device_list_info_cache: Optional[List[Dict]] = None
    device_list: Optional[List[pykinect.Device]] = None
    calibrations: Optional[Dict[str, pykinect.Calibration]] = None
    serial_map: Optional[Dict[int, str]] = None
    multical_calibration: Optional[MulticalCameraInfo] = None

    recording_processes: List[subprocess.Popen] = None

    def __init__(self, cfg: KinectSystemCfg):
        self.option = cfg
        self.state = {
            "recording": False,
            "single_shot": False,
        }
        self.lock = threading.RLock()
        self.logger = logging.getLogger('azure_kinect_apiserver.app')

        self.device_list_info_cache = None
        self.device_list = None
        self.serial_map = None
        self.multical_calibration = None

        self.recording_processes = []

    def start_recording(self, tag: Optional[str]) -> Optional[Exception]:
        self.lock.acquire()
        if self.state["single_shot"]:
            self.lock.release()
            return Exception("single shot is running")
        elif self.state["recording"]:
            self.lock.release()
            return Exception("recording is running")
        else:
            # Do something
            if tag is None or tag == "":
                tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

            record_path = osp.join(self.option.data_path, tag, 'kinect')
            if not osp.exists(record_path):
                os.makedirs(record_path, exist_ok=True)

            if self.multical_calibration is not None:
                self.multical_calibration.save_to_json(osp.join(record_path, "calibration.json"))

            # for process in procs:
            #     os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            commands, err = self.option.get_command_with_exec(tag)
            if err is not None:
                self.lock.release()
                return err

            timestamps = []
            for command in commands:
                logging.info(command)
                if hasattr(signal, 'CTRL_C_EVENT'):
                    # windows
                    process = subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                else:
                    process = subprocess.Popen(command, shell=True)
                timestamps.append(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                self.recording_processes.append(process)
                import time
                time.sleep(1)

            if self.option.debug:
                with open(osp.join(record_path, "debug.log"), "w+") as f:
                    f.write("Start recording at: \n\t")
                    f.write("\n\t".join(timestamps))
            self.recording_processes.reverse()

            self.state["recording"] = True
            self.lock.release()
            return None

    def stop_recording(self) -> Optional[Exception]:
        self.lock.acquire()
        if self.state["single_shot"]:
            self.lock.release()
            return Exception("single shot is running")
        elif not self.state["recording"]:
            self.lock.release()
            return Exception("recording is not running")
        else:
            # Do something

            for process in self.recording_processes:
                logging.info(process)
                if hasattr(signal, 'CTRL_C_EVENT'):
                    # windows. Need CTRL_BREAK_EVENT to raise the signal in the whole process group
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    process.kill()
                    # os.kill(process.pid, signal.CTRL_C_EVENT)
                else:
                    os.kill(process.pid, signal.SIGINT)
                # process.terminate()
                # os.kill(process.pid, signal.SIGTERM)
                # os.kill(process.pid, signal.SIGINT)
                # process.send_signal(signal.CTRL_C_EVENT)

            try:
                for process in self.recording_processes:
                    try:
                        process.wait(timeout=3)
                    except Exception as e:
                        logger.warning(e)
            except Exception as e:
                logger.warning(e)

            self.state["recording"] = False
            self.recording_processes = []
            self.lock.release()
            return None

    def list_device(self) -> Tuple[List[Dict], Optional[Exception]]:
        res, err = probe_device(self.option.exec_path)
        if err is not None:
            return res, err
        else:
            self.device_list_info_cache = res
            return res, None

    def __get_device_config__(self, index: int) -> pykinect.Configuration:
        if self.option is None or self.option.camera_options is None or len(self.option.camera_options) <= index:
            device_config = pykinect.default_configuration
            device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
            device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
            device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
            return device_config
        else:
            opt = self.option.camera_options[index]
            device_config = pykinect.default_configuration

            device_config.camera_fps = [5, 10, 30].index(opt.camera_fps)
            # device_config.color_format = opt.color_format
            device_config.color_resolution = int(opt.color_resolution)
            device_config.depth_delay_off_color_usec = int(opt.depth_delay_usec)
            device_config.depth_mode = int(opt.depth_mode)
            device_config.subordinate_delay_off_master_usec = int(opt.sync_delay_usec)
            device_config.wired_sync_mode = int(opt.sync_mode)
            return device_config

    def enter_single_shot_mode(self) -> Optional[Exception]:
        self.lock.acquire()
        if self.state["single_shot"]:
            self.lock.release()
            return Exception("single shot is running")
        elif self.state["recording"]:
            self.lock.release()
            return Exception("recording is running")
        else:
            self.state["single_shot"] = True
            self.lock.release()

            if self.device_list_info_cache is None:
                res, err = self.list_device()
                if err is None:
                    device_info_list = res
                else:
                    return err
            else:
                device_info_list = self.device_list_info_cache

            device_info_list = list(filter(lambda x: x['Serial'] in [x.sn for x in self.option.camera_options], device_info_list))

            # Modify camera configuration
            self.device_list: List[pykinect.Device] = [pykinect.start_device(device_index=i, config=self.__get_device_config__(i)) for i, _ in
                                                       enumerate(device_info_list)]

            print(device_info_list)
            self.calibrations = {device['Serial']: self.device_list[i].get_calibration(self.__get_device_config__(i).depth_mode, self.__get_device_config__(i).color_resolution) for i, device in
                                 enumerate(device_info_list)}
            self.serial_map = {device['Index']: device['Serial'] for device in device_info_list}
            return None

    def exit_single_shot_mode(self):
        self.lock.acquire()
        if not self.state["single_shot"]:
            self.lock.release()
            return Exception("single shot is not running")
        elif self.state["recording"]:
            self.lock.release()
            return Exception("recording is running")
        else:
            if self.device_list is not None and len(self.device_list) > 0:
                for device in self.device_list:
                    device.stop_cameras()
                    device.close()
            self.device_list = None
            self.serial_map = None
            self.state["single_shot"] = False
            self.lock.release()
            return None

    @staticmethod
    def __retrieve_frame__(device_list: List[pykinect.Device], current_frame, current_depth_frame, index):
        try:
            color_image, transformed_colored_depth_image = None, None
            success = False
            while not success:
                # Get capture
                capture = device_list[index].update()
                # Get the color image from the capture
                ret_color, color_image = capture.get_color_image()
                ret_depth, transformed_colored_depth_image = capture.get_transformed_depth_image()
                # TODO: Use nearest interpolation to get the depth image

                success = ret_color and ret_depth

            current_frame[index] = color_image
            current_depth_frame[index] = transformed_colored_depth_image
            logger.debug(color_image.shape, color_image.dtype, transformed_colored_depth_image.shape,
                         transformed_colored_depth_image.dtype)
        except Exception as e:
            logger.error(e)
            return e

    @staticmethod
    def __save_image__(data_path, tag, camera_sn, current_frame, current_depth_frame, index):
        cv2.imwrite(osp.join(data_path, tag, camera_sn, 'color', f"{index}.jpg"),
                    current_frame)
        # np.save(f"./save/{serial_map[i]}/depth/{number}.npy", current_depth_frame[i])
        cv2.imwrite(osp.join(data_path, tag, camera_sn, 'depth', f"{index}.png"),
                    current_depth_frame)
        # load method: cv2.imread('path', cv2.IMREAD_UNCHANGED)

    def single_shot(self, tag: str, index: int) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]], int, Optional[Exception]]:
        if tag is None or tag == "":
            return [], [], -1, Exception("tag is empty")

        if not self.state["single_shot"]:
            err = self.enter_single_shot_mode()
            if err is not None:
                return [], [], -1, err

        for i in range(len(self.device_list)):
            camera_i_path = osp.join(self.option.data_path, tag, self.serial_map[i])
            camera_i_path_color = osp.join(camera_i_path, "color")
            camera_i_path_depth = osp.join(camera_i_path, "depth")

            if not osp.exists(camera_i_path):
                os.makedirs(camera_i_path, exist_ok=True)
                with open(osp.join(camera_i_path, "calibration.kinect.json"), 'w') as f:
                    json.dump({serial: calibration.get_all_parameters() for serial, calibration in self.calibrations.items()}, f, indent=4)
            if not osp.exists(camera_i_path_color):
                os.makedirs(camera_i_path_color, exist_ok=True)
            if not osp.exists(camera_i_path_depth):
                os.makedirs(camera_i_path_depth, exist_ok=True)

        #
        # threads = []
        # for i in range(len(self.device_list)):
        #     t = threading.Thread(target=self.retrieve_frame, args=(self.device_list, current_frames, current_depth_frames, i))
        #     t.start()
        #     threads.append(t)
        # [t.join() for t in threads]

        current_frames = [None for _ in range(len(self.device_list))]
        current_depth_frames = [None for _ in range(
            len(self.device_list))]

        for i in range(len(self.device_list)):
            self.__retrieve_frame__(self.device_list, current_frames, current_depth_frames, i)

        with ThreadPoolExecutor(max_workers=len(self.device_list)) as executor:
            res = []
            for i in range(len(self.device_list)):
                res.append(executor.submit(self.__save_image__, self.option.data_path, tag, self.serial_map[i], current_frames[i], current_depth_frames[i], index))
            [r.result() for r in res]

        return current_frames, current_depth_frames, index, None

    def single_shot_mem(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray], int, Optional[Exception]]:
        if not self.state["single_shot"]:
            err = self.enter_single_shot_mode()
            if err is not None:
                return [], [], -1, err

        current_frames: List[Optional[np.ndarray]] = [None for _ in range(len(self.device_list))]
        current_depth_frames: List[Optional[np.ndarray]] = [None for _ in range(len(self.device_list))]

        for i in range(len(self.device_list)):
            self.__retrieve_frame__(self.device_list, current_frames, current_depth_frames, i)

        return current_frames, current_depth_frames, index, None

    def single_shot_compressed(self, index: int) -> Tuple[List[bytes], List[bytes], int, Optional[Exception]]:
        color_frames, depth_frames, index, err = self.single_shot_mem(index)
        if err is not None:
            return color_frames, depth_frames, index, err
        else:
            color_frames_compressed = [base64.b64encode(cv2.imencode('.jpg', frame)[1].tobytes()) for frame in color_frames]
            depth_frames_compressed = [base64.b64encode(cv2.imencode('.png', frame)[1].tobytes()) for frame in depth_frames]
            return color_frames_compressed, depth_frames_compressed, index, None

    def load_mulitcal_calibration(self, calibration_file: str):
        self.multical_calibration = MulticalCameraInfo(calibration_file)
        if not self.multical_calibration.valid:
            self.logger.warning("invalid multical calibration file")
            self.multical_calibration = None
        else:
            self.logger.info("load multical calibration file")


if __name__ == '__main__':
    import time
    import tqdm

    cfg = KinectSystemCfg('manifests/azure_kinect_config/azure_kinect_config.yaml')

    app = Application(cfg)
    print(app.list_device())

    # app.enter_single_shot_mode()
    # start = time.time()
    # for i in range(10):
    #     app.single_shot('test', i)
    # print(time.time() - start)
    # app.exit_single_shot_mode()

    app.start_recording(None)
    duration_sec = 1800
    with tqdm.tqdm(total=duration_sec) as pbar:
        for i in range(duration_sec):
            time.sleep(1)
            pbar.update(1)
    app.stop_recording()
