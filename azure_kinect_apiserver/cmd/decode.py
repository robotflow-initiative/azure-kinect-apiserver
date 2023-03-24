import argparse
import concurrent.futures
import datetime
import glob
import json
import logging
import os
import threading
from os import path as osp
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd
import plyer
import tqdm

from azure_kinect_apiserver.decoder import (
    mkv_record_wrapper,
    StateMachine,
    state_machine_save_thread_v1,
    get_mkv_record_meta,
    get_mkv_record_calibration,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("azure_kinect_apiserver.cmd.decode")


def compute_timestamp_offset(calibration_pairs: List[Tuple[int, str]]):
    calibration_pairs = np.array(list(
        map(
            lambda x: [int(x[0]) * 1e-6, datetime.datetime.strptime(x[1], "%Y-%m-%d_%H:%M:%S.%f").timestamp()],
            calibration_pairs
        )
    ))
    timestamp_offset = np.mean(calibration_pairs[:, 1] - calibration_pairs[:, 0])
    logger.info(f"calibration pairs:\n {calibration_pairs}")
    return float(timestamp_offset)


def get_timestamp_offset_from_decoded(tagged_path: str,
                                      cam_name: str,
                                      examine_n_frames: int = 600,
                                      maximum_no_detect_count: int = 30,
                                      debug: bool = True) -> Tuple[float, float, Optional[Exception]]:
    # Read metadata and check if all color images are present
    _color_img_path_collection = sorted(glob.glob(os.path.join(tagged_path, cam_name, 'color', '*.jpg')), key=lambda x: int(osp.splitext(osp.basename(x))[0]))
    color_img_meta = pd.read_csv(os.path.join(tagged_path, cam_name, 'meta.csv'))
    assert len(color_img_meta) == len(_color_img_path_collection), f"len(color_img_meta)={len(color_img_meta)}"
    color_img_path_collection = [osp.join(tagged_path, cam_name, 'color', str(filename_no_ext) + '.jpg') for filename_no_ext in color_img_meta['basename']]
    assert all([osp.exists(x) for x in color_img_path_collection]), f"some color images are missing"

    decoder = cv2.QRCodeDetector()
    calibration_pairs = []
    last_qrcode_data = None
    no_detect_count = 0

    with tqdm.tqdm(total=len(color_img_path_collection)) as pbar:
        for img_idx, img_path in enumerate(color_img_path_collection):
            # Read color image
            if img_idx >= examine_n_frames or no_detect_count >= maximum_no_detect_count:
                break
            color_img = cv2.imread(img_path)
            data, bbox, straight_qrcode = decoder.detectAndDecode(color_img)
            if len(data) > 0:
                if debug:
                    print(f"Found QR code at {bbox} with data {data}")
                    if bbox is not None:
                        cv2.rectangle(color_img, tuple(bbox[0][0].astype(int)), tuple(bbox[0][2].astype(int)), (0, 255, 0), 2)
                    cv2.putText(color_img, data, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    cv2.imshow('img', color_img)
                    cv2.waitKey(0)
                if last_qrcode_data is not None:
                    if data == last_qrcode_data:
                        continue
                    else:
                        calibration_pairs.append((color_img_meta['basename'][img_idx], data))
                        last_qrcode_data = data
                        logger.debug(f"found calibration pair: {color_img_meta['basename'][img_idx]} -> {data}")
                else:
                    last_qrcode_data = data
            else:
                if len(calibration_pairs) > 0:
                    no_detect_count += 1
            pbar.update(1)
    if len(calibration_pairs) < 2:
        logger.error(f"found {len(calibration_pairs)} calibration pairs, which is less than 2")
        return 0, 0, Exception("found less than 2 calibration pairs")
    else:
        timestamp_offset = compute_timestamp_offset(calibration_pairs)
        logger.info(f"found {len(calibration_pairs)} calibration pairs, timestamp offset is {timestamp_offset}")
        return timestamp_offset, datetime.datetime.strptime(calibration_pairs[-1][-1], "%Y-%m-%d_%H:%M:%S.%f").timestamp(), None


def mkv_worker(kinect_dir: str):
    files = glob.glob(kinect_dir + "/*.mkv")
    names = [osp.basename(f).split('.')[0] for f in files]
    wrappers = [
        mkv_record_wrapper(f) for f in files
    ]
    m = StateMachine(names)
    t = threading.Thread(target=state_machine_save_thread_v1, args=(m, kinect_dir, names))
    t.start()
    with tqdm.tqdm() as pbar:
        num_closed = 0
        while num_closed < len(wrappers):
            frame_futures: Dict[str, Optional[concurrent.futures.Future]] = {k: None for k in names}
            for idx, w in enumerate(wrappers):
                try:
                    # noinspection PyTypeChecker
                    frame_futures[names[idx]], err = next(w)
                    if err is not None:
                        raise err
                except StopIteration:
                    num_closed += 1
                    continue

            frames = {k: frame_futures[k].result() for k in names if frame_futures[k] is not None}

            for stream_id, frame in frames.items():
                if frame is not None:
                    m.push(stream_id, frame)
                    pbar.set_description(f"pressure: {len(m.frame_buffer[names[1]])}")
                    pbar.update(1)

        m.close()
        t.join()

    for i, file in enumerate(files):
        with open(osp.join(kinect_dir, names[i], f"calibration.kinect.json"), "w") as f:
            json.dump(get_mkv_record_calibration(file)[0], f, indent=4, sort_keys=True)

    metadata = {'recordings': {names[i]: get_mkv_record_meta(file)[0] for i, file in enumerate(files)}}
    master_camera = list(filter(lambda x: x[1]['is_master'] is True, metadata['recordings'].items()))[0][0]
    ts, ts_action_start, err = get_timestamp_offset_from_decoded(tagged_path=kinect_dir, cam_name=master_camera, debug=False)
    ret: Optional[Exception] = None
    if err is not None:
        logging.error(str(err))
        ret = Exception("failed to get timestamp offset")
        metadata['system_timestamp_offset'] = 0
        metadata['system_action_start_timestamp'] = 0
    else:
        metadata['system_timestamp_offset'] = ts
        metadata['system_action_start_timestamp'] = ts_action_start

    with open(osp.join(kinect_dir, "meta.json"), "w") as f:
        json.dump(metadata, f, indent=4, sort_keys=True)

    return ret


def main(args: argparse.Namespace):
    logger.info("processing directory: {}".format(osp.realpath(args.data_dir)))

    kinect_dir = osp.join(args.data_dir, "kinect")

    return mkv_worker(kinect_dir)


def entry_point(argv):
    if len(argv) < 1:
        try:
            f = plyer.filechooser.choose_dir(title="Select a recording")
            if len(f) > 0:
                args = argparse.Namespace()
                args.data_dir = f[0]
                return main(args)
            else:
                logger.info("abort")
                return 1
        except Exception as err:
            logger.error(f"error: {err}")
            print("Usage: python -m azure_kinect_apiserver decode <path>")
            return 1

    else:
        data_dir = argv[0]
        args = argparse.Namespace()
        args.data_dir = data_dir
        return main(args)
