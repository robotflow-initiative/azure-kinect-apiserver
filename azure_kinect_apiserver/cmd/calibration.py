import argparse
import datetime
import logging
from os import path as osp

import cv2
import numpy as np
from py_cli_interaction import must_parse_cli_string

from azure_kinect_apiserver.apiserver import Application
from azure_kinect_apiserver.common import KinectSystemCfg


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


def interaction_loop(app: Application):
    tag = must_parse_cli_string("Please enter a tag for the calibration: ", default_value="cali_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    app.enter_single_shot_mode()
    print("--------\nPress 's' or Enter to take a single shot, 'c' or space to refresh preview, 'q' or ESC to quit\n--------")
    index = 0
    while True:
        color_frames, depth_frames, _, err = app.single_shot_mem(tag, index)
        if err is not None:
            logging.error(f"error: {err}")
            break

        color_preview = np.vstack([x[::8, ::8, :] for x in color_frames])
        cv2.putText(color_preview, f"tag: {tag}, index: {index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(f"color", color_preview)

        depth_preview = np.vstack([x[::8, ::8, None] for x in depth_frames])
        depth_preview = clip_depth_image(depth_preview)
        cv2.putText(depth_preview, f"tag: {tag}, index: {index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(f"depth", depth_preview)

        k = cv2.waitKey()
        if k == ord('q') or k == 27:
            break
        elif k == ord('s') or k == 13:
            app.single_shot(tag, index)
            index += 1
        elif k == ord('c') or k == ord(' '):
            continue


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("azure_kinect_apiserver.cmd.calibration")
    logger.info("using config file: {}".format(osp.realpath(args.config)))

    cfg = KinectSystemCfg(args.config)
    if cfg.valid is False:
        logging.error(f"invalid config file {osp.realpath(args.config)}")
        exit(1)
    return interaction_loop(Application(cfg))


def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./azure_kinect_config.yaml')
    print(argv)

    args = parser.parse_args(argv)
    return main(args)


if __name__ == '__main__':
    import sys

    exit(entry_point(sys.argv))
