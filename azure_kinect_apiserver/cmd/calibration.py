import argparse
import datetime
import logging
from os import path as osp

import cv2
import numpy as np
import plyer
from py_cli_interaction import must_parse_cli_string

from azure_kinect_apiserver.apiserver.server import Application
from azure_kinect_apiserver.common import KinectSystemCfg
from azure_kinect_apiserver.decoder import ArucoDetectHelper


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
        color_frames, depth_frames, _, err = app.single_shot_mem(index)
        if err is not None:
            logging.error(f"error: {err}")
            break

        scale = int(color_frames[0].shape[0] / 360)
        color_preview = np.vstack([x[::scale, ::scale, :] for x in color_frames])
        cv2.putText(color_preview, f"tag: {tag}, index: {index}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(f"color", color_preview)

        depth_preview = np.vstack([x[::scale, ::scale, None] for x in depth_frames])
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

    with open(osp.join(app.option.data_path, tag, "calibration.json"), "w") as f:
        f.write('{}')
    logging.info(f"calibration data saved to {osp.join(app.option.data_path, tag)}")


def aruco_preview(app: Application):
    app.enter_single_shot_mode()
    ctx = ArucoDetectHelper(
        app.option.marker_length_m,
        getattr(cv2.aruco, app.option.marker_type),
    )
    while True:
        color_frames, depth_frames, _, err = app.single_shot_mem(0)
        if err is not None:
            logging.error(f"error: {err}")
            continue

        scale = int(color_frames[0].shape[0] / 540)
        color_frames = [ctx.preview_one_frame(x)[0] for x in color_frames]
        color_preview = np.vstack([x[::scale, ::scale, :] for x in color_frames])
        cv2.imshow(f"color", color_preview)

        k = cv2.waitKey(1)
        if k == ord('q') or k == 27:
            break
        elif k == ord('c') or k == ord(' '):
            continue


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("azure_kinect_apiserver.cmd.calibration")
    logger.info("using config file: {}".format(osp.realpath(args.config)))

    cfg = KinectSystemCfg(args.config)
    if cfg.valid is False:
        logger.error(f"invalid config file {osp.realpath(args.config)}, please select from current directory:")
        try:
            f = plyer.filechooser.open_file(title="Select a config file", filters="*.yaml")
            if len(f) > 0:
                cfg = KinectSystemCfg(f[0])
                if cfg.valid is False:
                    logger.error(f"invalid config file {osp.realpath(f[0])}")
                    return 1
            else:
                logger.info("abort")
                return 1
        except Exception as err:
            logger.error(f"error: {err}")
            return 1

    if args.aruco_preview:
        return aruco_preview(Application(cfg))
    else:
        return interaction_loop(Application(cfg))


def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--aruco_preview', action='store_true')
    print(argv)

    args = parser.parse_args(argv)
    return main(args)


if __name__ == '__main__':
    import sys

    exit(entry_point(sys.argv))
