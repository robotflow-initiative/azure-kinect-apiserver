import logging
import subprocess

import cv2
import numpy as np


def clip_depth_image(depth_image: np.ndarray, min_depth: int = 0, max_depth: int = 10000) -> np.ndarray:
    """Converts a depth image to a color image.

    Args:
        depth_image: A depth image as a numpy array.

    Returns:
        A color image as a numpy array.
        :param depth_image:
        :param min_depth:
        :param max_depth:
    """
    depth_image = depth_image - min_depth
    depth_image = depth_image / (max_depth - min_depth) * 255
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)


def probe_device(exec_path: str):
    res = []
    process = subprocess.Popen(
        f"{exec_path} --list",
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
            logging.getLogger('azure_kinect_apiserver.common.functional').error(e)
            return [], e
    return res, None
