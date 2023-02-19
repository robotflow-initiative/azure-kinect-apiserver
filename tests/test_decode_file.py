from azure_kinect_apiserver.thirdparty import MKVReader, TRACK
from azure_kinect_apiserver.thirdparty import pykinect
# import pykinect_azure as pykinect

from typing import Optional, Dict, Any, List, Tuple
import cv2
import numpy as np


def clip_depth_image(depth_image: np.ndarray, min_depth: int = 0, max_depth: int = 10000) -> np.ndarray:
    """Converts a depth image to a color image.

    Args:
        depth_image: A depth image as a numpy array.

    Returns:
        A color image as a numpy array.
    """
    depth_image = depth_image - min_depth
    depth_image = depth_image / (max_depth - min_depth) * 255
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)


def decode_file_v1(path: str) -> Optional[Exception]:
    if not path.endswith('.mkv'):
        return Exception(f'Invalid file extension: {path}')

    pb = pykinect.start_playback(path)
    pb_config = pb.get_record_configuration()
    pb_length = pb.get_recording_length()
    print("---- playback.is_valid() ----")
    print(pb.is_valid())
    print("---- playback.config----")
    print(pb_config)
    print("---- playback.calibration ----")
    print(pb.calibration)
    print("---- playback.length ----")
    print(pb_length)
    while True:
        try:
            ret, capture = pb.update()
        except EOFError:
            break
        if not ret:
            break
        # Use capture...
        # Get color image
        ret_color, color_image = capture.get_color_image()
        # ret_color, color_image = capture.get_transformed_color_image()

        # Get the colored depth
        # ret_depth, depth_color_image = capture.get_colored_depth_image()
        ret_depth, depth_color_image = capture.get_transformed_depth_image()

        if not ret_color or not ret_depth:
            continue

        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_color_image, 0.3, 0)
        # color_img = frameset[TRACK.COLOR]
        cv2.imshow('color', color_image[::4, ::4, :])
        cv2.imshow('depth', clip_depth_image(depth_color_image)[::4, ::4, :])
        cv2.waitKey(1)

    print('done')


def decode_file_v2(path: str) -> Optional[Exception]:
    max_depth = 10000
    min_depth = 0
    if not path.endswith('.mkv'):
        return Exception(f'Invalid file extension: {path}')

    # Initialize MKVReader object
    reader = MKVReader(path)
    calib = reader.get_calibration()
    while True:
        try:
            frameset = reader.get_next_frameset()
        except EOFError:
            break

        # Use frameset...
        color_img = frameset[TRACK.COLOR]
        cv2.imshow('color', color_img[::4, ::4, :])
        if TRACK.DEPTH in frameset.keys():
            depth_img = frameset[TRACK.DEPTH]
            depth_img = depth_img - min_depth
            depth_img = depth_img / (max_depth - min_depth) * 255
            cv2.imshow('depth', cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=1), cv2.COLORMAP_JET))
        cv2.waitKey(1)

    print('done')


if __name__ == '__main__':
    pykinect.initialize_libraries()
    decode_file_v1(r"/azure_kinect_data/20230216_232955/output_0.mkv")
