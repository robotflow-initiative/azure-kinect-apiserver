import concurrent.futures
import os
import os.path as osp
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(osp.join(os.path.dirname(__file__), '../thirdparty/pyKinectAzure'))

from azure_kinect_apiserver.thirdparty import pykinect

from typing import Optional, Dict, Any, Tuple


def get_mkv_record_meta(path: str) -> Tuple[Dict[str, Any], Optional[Exception]]:
    if not path.endswith('.mkv'):
        return {}, Exception(f'Invalid file extension: {path}')

    pb: pykinect.Playback = pykinect.start_playback(path)
    pb_config = pb.get_record_configuration()
    pb_length = pb.get_recording_length()
    ret, capture = pb.update()
    if ret:
        pb_calibration = {
            'color': capture.calibration.get_matrix(pykinect.K4A_CALIBRATION_TYPE_COLOR),
            'depth': capture.calibration.get_matrix(pykinect.K4A_CALIBRATION_TYPE_DEPTH),
        }
        pb_resolution = {
            'color': (capture.camera_transform.color_resolution.width, capture.camera_transform.color_resolution.height),
            'depth': (capture.camera_transform.depth_resolution.width, capture.camera_transform.depth_resolution.height),
        }
    else:
        pb_calibration = None
        pb_resolution = None

    return {'config': pb_config, 'length': pb_length, 'calibration': pb_calibration, 'resolution': pb_resolution}, None


def decode_thread(index, capture) -> Optional[Dict]:
    # Get color image
    color_obj = capture.get_color_image_object()
    ret_color, color_image = color_obj.to_numpy()

    # Get the colored depth
    depth_obj = capture.get_depth_image_object()
    depth_obj_t = capture.camera_transform.depth_image_to_color_camera(depth_obj)
    ret_depth, depth_color_image = depth_obj_t.to_numpy()

    if not ret_color or not ret_depth:
        return None

    else:
        return {
            'color': color_image,
            'depth': depth_color_image,
            'color_dev_ts_usec': color_obj.device_timestamp_usec,
            'color_sys_ts_nsec': color_obj.system_timestamp_nsec,
            'depth_dev_ts_usec': depth_obj.device_timestamp_usec,
            'depth_sys_ts_nsec': depth_obj.system_timestamp_nsec,
            'index': index
        }


def mkv_record_wrapper(path: str) -> Tuple[Optional[concurrent.futures.Future], Optional[Exception]]:
    if not path.endswith('.mkv'):
        return None, Exception(f'Invalid file extension: {path}')

    pb: pykinect.Playback = pykinect.start_playback(path)

    index = -1
    pool = ThreadPoolExecutor(max_workers=4)

    while True:
        try:
            ret, capture = pb.update()
            index += 1
            if not ret:
                pool.shutdown(wait=True)
                return None, EOFError('end of file')
            else:
                time.sleep(0.01)
                yield pool.submit(decode_thread, index, capture), None
        except EOFError:
            pool.shutdown(wait=True)
            return None, EOFError('end of file')

        # Use capture...
        # Get color image
        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_color_image, 0.3, 0)
        # color_img = frameset[TRACK.COLOR]
