import argparse
import base64
import logging
import os
import os.path as osp
import pickle
import threading
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse

from azure_kinect_apiserver.apiserver.app import Application
from azure_kinect_apiserver.common import KinectSystemCfg

APPLICATION: Optional[Application] = None
controller: FastAPI = FastAPI()

controller.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_response(status_code, **kwargs):
    data = {'code': status_code, 'timestamp': time.time()}
    data.update(**kwargs)
    json_compatible_data = jsonable_encoder(data)
    resp = JSONResponse(content=json_compatible_data, status_code=status_code)
    return resp


@controller.get("/")
def root():
    return RedirectResponse(url='/docs')


@controller.get("/v1/azure/status")
def get_status():
    global APPLICATION
    return make_response(200, recording=APPLICATION.state['recording'], single_shot=APPLICATION.state['single_shot'], status=True)


@controller.get("/v1/azure/single_shot")
def single_shot(tag: str, index: int, save: bool = True):
    """

    :param tag:
    :param index:
    :param save:
    :return: json response, depth and color are base64 encoded pickle string
    """
    global APPLICATION
    if save:
        _, _, index, err = APPLICATION.single_shot(tag, index)
        return make_response(200, single_shot=APPLICATION.state['single_shot'], depth=None, color=None, index=index, err=str(err))
    else:
        # 4 x 2160p color + depth consumes 210M bandwidth
        # TODO: use a better way to transfer data
        color_frames, depth_frames, index, err = APPLICATION.single_shot_mem(tag, index)
        return make_response(200, single_shot=APPLICATION.state['single_shot'], depth=base64.b64encode(pickle.dumps(depth_frames)), color=base64.b64encode(pickle.dumps(color_frames)), index=index,
                             ret=str(err))


@controller.delete("/v1/azure/single_shot")
def stop_single_shot():
    global APPLICATION
    APPLICATION.exit_single_shot_mode()
    return make_response(200, single_shot=APPLICATION.state['single_shot'])


@controller.post("/v1/azure/start")
def start_recording(tag: str):
    global APPLICATION
    ret = APPLICATION.start_recording(tag)
    return make_response(200, message="recording started", recording=APPLICATION.state['recording'], err=str(ret))


@controller.post("/v1/azure/stop")
def stop_recording():
    global APPLICATION
    ret = APPLICATION.stop_recording()
    return make_response(200, message="recording stopped", recording=APPLICATION.state['recording'], err=str(ret))


# noinspection PyUnresolvedReferences
def main(args):
    global APPLICATION
    logging.basicConfig(level=logging.INFO)

    cfg = KinectSystemCfg(args.config)
    if cfg.valid is False:
        logging.error(f"invalid config file {osp.realpath(args.config)}")
        exit(1)
    APPLICATION = Application(cfg)

    if args.multical_calibration is not None:
        APPLICATION.load_mulitcal_calibration(args.multical_calibration)
    # Prepare system
    APPLICATION.logger.info(f"azure apiserver service listen at {cfg.api_port}")
    APPLICATION.logger.info(f"azure apiserver config {cfg}")

    try:
        thread = threading.Thread(target=uvicorn.run, kwargs={'app': controller, 'port': cfg.api_port, 'host': cfg.api_interface})
        thread.start()

        while True:
            time.sleep(86400)
        # uvicorn.run(app=controller, port=cfg.api_port, host=cfg.api_interface)
    except KeyboardInterrupt:
        APPLICATION.logger.info(f"got KeyboardInterrupt")
        if APPLICATION.state['recording']:
            APPLICATION.logger.info(f"stop recording")
            APPLICATION.stop_recording()
        APPLICATION.stop_recording()
        if APPLICATION.state['single_shot']:
            APPLICATION.logger.info(f"exit single shot mode")
            APPLICATION.exit_single_shot_mode()
        # noinspection PyProtectedMember
        os._exit(1)


def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./azure_kinect_config.yaml', help='path to azure kinect config file')
    parser.add_argument('--multical_calibration', type=str, default=None, help='path to multical calibration file, can be its parent directory')
    args = parser.parse_args(argv)
    main(args)


if __name__ == '__main__':
    import sys

    entry_point(sys.argv[1:])
