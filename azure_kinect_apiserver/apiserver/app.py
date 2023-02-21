import datetime
import logging
import os
import os.path as osp
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

import azure_kinect_apiserver.thirdparty.pyKinectAzure.pykinect_azure as pykinect
from azure_kinect_apiserver.common import KinectSystemCfg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('azure_kinect_apiserver.app')

pykinect.initialize_libraries()


class Application:
    option: KinectSystemCfg = None
    state: Dict[str, bool] = None
    lock: threading.RLock = None

    device_list_info_cache: Optional[List[Dict]] = None
    device_list: Optional[List[pykinect.Device]] = None
    serial_map: Optional[Dict[int, str]] = None

    recording_processes: List[subprocess.Popen] = None

    def __init__(self, cfg: KinectSystemCfg):
        self.option = cfg
        self.state = {
            "recording": False,
            "single_shot": False,
        }
        self.lock = threading.RLock()

        self.device_list_info_cache = None
        self.device_list = None
        self.serial_map = None

        self.recording_processes = []

    def start_recording(self, tag: str) -> Optional[Exception]:
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

            record_path = osp.join(self.option.data_path, tag)
            if not osp.exists(record_path):
                os.makedirs(record_path)

            # for process in procs:
            #     os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            commands, err = self.option.get_command_with_exec(tag)
            if err is not None:
                self.lock.release()
                return err

            timestamps = []
            for command in commands:
                logging.info(command)
                process = subprocess.Popen(command, shell=True)
                timestamps.append(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
                self.recording_processes.append(process)
                time.sleep(1)

            with open(osp.join(record_path, "default.log"), "w+") as f:
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
                process.send_signal(signal.CTRL_C_EVENT)

            try:
                for process in self.recording_processes:
                    try:
                        process.wait(timeout=2)
                    except Exception as e:
                        logger.warning(e)
            except Exception as e:
                logger.warning(e)

            self.state["recording"] = False
            self.lock.release()
            return None

    def list_device(self) -> Tuple[List[Dict], Optional[Exception]]:
        res = []
        process = subprocess.Popen(
            f"{self.option.exec_path} --list",
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
                logger.error(e)
                return [], e
        self.device_list_info_cache = res
        return res, None

    def __get_device_config__(self, index: int) -> pykinect.Configuration:
        # TODO: Implement this function
        device_config = pykinect.default_configuration
        device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE
        device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
        device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
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
                device_info_list = self.list_device()
            else:
                device_info_list = self.device_list_info_cache

            # Modify camera configuration
            self.device_list = [pykinect.start_device(device_index=i, config=self.__get_device_config__(i)) for i, _ in
                                enumerate(device_info_list)]

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

                success = ret_color and ret_depth

            current_frame[index] = color_image
            current_depth_frame[index] = transformed_colored_depth_image
            logger.debug(color_image.shape, color_image.dtype, transformed_colored_depth_image.shape,
                         transformed_colored_depth_image.dtype)
        except Exception as e:
            logger.error(e)
            return e

    # @staticmethod
    # def retrieve_frame(device_list):
    #     current_frame = [None for _ in range(len(device_list))]
    #     current_depth_frame = [None for _ in range(len(device_list))]
    #     success: List[bool] = [False for _ in range(len(device_list))]
    #     captures: List[Optional[pykinect.Capture]] = [None for _ in range(len(device_list))]
    #     try:
    #         color_image, transformed_colored_depth_image = None, None
    #         while not all(success):
    #             for i in range(len(device_list)):
    #                 if not success[i]:
    #                     # Get capture
    #                     captures[i] = device_list[i].update()
    #
    #             for i in range(len(device_list)):
    #                 # Get the color image from the capture
    #                 ret_color, color_image = captures[i].get_color_image()
    #                 ret_depth, transformed_colored_depth_image = captures[i].get_transformed_depth_image()
    #                 if ret_color and ret_depth:
    #                     success[i] = True
    #                     current_frame[i] = color_image
    #                     current_depth_frame[i] = transformed_colored_depth_image
    #                     logger.debug(color_image.shape, color_image.dtype, transformed_colored_depth_image.shape,
    #                                  transformed_colored_depth_image.dtype)
    #
    #         return current_frame, current_depth_frame
    #     except Exception as e:
    #         logger.error(e)
    #         return e

    @staticmethod
    def __save_image__(data_path, tag, camera_sn, current_frame, current_depth_frame, index):
        cv2.imwrite(osp.join(data_path, tag, camera_sn, 'color', f"{index}.png"),
                    current_frame)
        # np.save(f"./save/{serial_map[i]}/depth/{number}.npy", current_depth_frame[i])
        cv2.imwrite(osp.join(data_path, tag, camera_sn, 'depth', f"{index}.png"),
                    current_depth_frame)
        # load method: cv2.imread('path', cv2.IMREAD_UNCHANGED)

    def single_shot(self, tag: str, index: int) -> Optional[Exception]:
        if tag is None or tag == "":
            return Exception("tag is empty")

        if not self.state["single_shot"]:
            err = self.enter_single_shot_mode()
            if err is not None:
                return err

        for i in range(len(self.device_list)):
            camera_i_path = osp.join(self.option.data_path, tag, self.serial_map[i])
            camera_i_path_color = osp.join(camera_i_path, "color")
            camera_i_path_depth = osp.join(camera_i_path, "depth")

            if not osp.exists(camera_i_path):
                os.makedirs(camera_i_path)
            if not osp.exists(camera_i_path_color):
                os.makedirs(camera_i_path_color)
            if not osp.exists(camera_i_path_depth):
                os.makedirs(camera_i_path_depth)

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

        # try:
        #     current_frames.clear()
        #     current_depth_frames.clear()
        #     for i, device in enumerate(self.device_list):
        #         color_image, transformed_colored_depth_image = None, None
        #         success = False
        #         while not success:
        #             # Get capture
        #             capture = device.update()
        #             # Get the color image from the capture
        #             ret_color, color_image = capture.get_color_image()
        #             ret_depth, transformed_colored_depth_image = capture.get_transformed_depth_image()
        #
        #             success = ret_color and ret_depth
        #
        #         current_frames.append(color_image)
        #         current_depth_frames.append(transformed_colored_depth_image)
        #         logger.debug(color_image.shape, color_image.dtype, transformed_colored_depth_image.shape,
        #                      transformed_colored_depth_image.dtype)
        #
        # except Exception as e:
        #     logger.error(e)
        #     return e
        #

        with ThreadPoolExecutor(max_workers=len(self.device_list)) as executor:
            res = []
            for i in range(len(self.device_list)):
                res.append(executor.submit(self.__save_image__, self.option.data_path, tag, self.serial_map[i], current_frames[i], current_depth_frames[i], index))
            [r.result() for r in res]

        return None

    def single_shot_mem(self, tag: str, index: int) -> Tuple[
        List[Optional[np.ndarray]], List[Optional[np.ndarray]], int, Optional[Exception]]:
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
                os.makedirs(camera_i_path)
            if not osp.exists(camera_i_path_color):
                os.makedirs(camera_i_path_color)
            if not osp.exists(camera_i_path_depth):
                os.makedirs(camera_i_path_depth)

        current_frames: List[Optional[np.ndarray]] = [None for _ in range(len(self.device_list))]
        current_depth_frames: List[Optional[np.ndarray]] = [None for _ in range(len(self.device_list))]

        for i in range(len(self.device_list)):
            self.__retrieve_frame__(self.device_list, current_frames, current_depth_frames, i)

        return current_frames, current_depth_frames, index, None


if __name__ == '__main__':
    import yaml
    import time

    cfg_dict = yaml.load(open('manifests/azure_kinect_config/azure_kinect_config.yaml', 'r'), Loader=yaml.FullLoader)
    cfg = KinectSystemCfg()
    cfg.load_dict(cfg_dict['azure_kinect'])

    app = Application(cfg)
    print(app.list_device())

    # app.enter_single_shot_mode()
    # start = time.time()
    # for i in range(10):
    #     app.single_shot('test', i)
    # print(time.time() - start)
    # app.exit_single_shot_mode()

    app.start_recording(None)
    time.sleep(10)
    app.stop_recording()
