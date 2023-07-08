import concurrent.futures
import os
import os.path as osp
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.append(osp.join(os.path.dirname(__file__), '../thirdparty/pyKinectAzure'))

# noinspection PyPep8
from azure_kinect_apiserver.thirdparty import pykinect

# noinspection PyPep8
from typing import Optional, Dict, Any, Tuple


def get_mkv_record_meta(path: str) -> Tuple[Dict[str, Any], Optional[Exception]]:
    if not path.endswith('.mkv'):
        return {}, Exception(f'Invalid file extension: {path}')

    pb: pykinect.Playback = pykinect.start_playback(path)
    pb_config = pb.get_record_configuration()
    pb_length = pb.get_recording_length()
    ret, capture = pb.update()
    if ret:
        pb_resolution = {
            'color': (capture.camera_transform.color_resolution.width, capture.camera_transform.color_resolution.height),
            'depth': (capture.camera_transform.depth_resolution.width, capture.camera_transform.depth_resolution.height),
        }
    else:
        pb_resolution = None

    return {'config': pb_config.to_dict(), 'length': pb_length, 'resolution': pb_resolution, 'is_master': pb_config.to_dict()['sync_mode'] == 1}, None


def get_mkv_record_calibration(path: str) -> Tuple[Dict[str, Any], Optional[Exception]]:
    if not path.endswith('.mkv'):
        return {}, Exception(f'Invalid file extension: {path}')

    pb: pykinect.Playback = pykinect.start_playback(path)
    ret, capture = pb.update()
    if ret:
        pb_calibration = capture.calibration.get_all_parameters()
    else:
        pb_calibration = None

    return pb_calibration, None


def decode_thread(index, capture) -> Optional[Dict]:
    # Get color image
    color_obj = capture.get_color_image_object()
    ret_color, color_image = color_obj.to_numpy()

    # Get the colored depth
    depth_obj = capture.get_depth_image_object()
    ret_depth, depth_image = depth_obj.to_numpy()

    if not ret_color or not ret_depth:
        return None
    else:
        depth_image_custom16 = pykinect.Image.create_custom16_from_numpy(depth_image)
        depth_obj_t = capture.camera_transform.depth_image_to_color_camera_custom(depth_obj, depth_image_custom16, pykinect.K4A_TRANSFORMATION_INTERPOLATION_TYPE_LINEAR)
        _, depth_image = depth_obj_t.to_numpy()

        # FIXME: cannot align color to depth
        # print(color_image.shape, depth_image.shape)
        # color_obj_bgra32 = pykinect.Image.create_bgra32_from_shape(color_image.shape[1], color_image.shape[0])
        # color_obj_t = capture.camera_transform.color_image_to_depth_camera(depth_obj, color_obj_bgra32)
        # _, color_image = color_obj_t.to_numpy()
        # color_image = color_image[:, :, :3]

        return {
            'color': color_image,
            'depth': depth_image,
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
