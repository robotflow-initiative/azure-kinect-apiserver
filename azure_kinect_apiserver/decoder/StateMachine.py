import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from os import path as osp
from typing import List, Dict, Any, Optional

import cv2

logger = logging.getLogger("azure_kinect_apiserver.decoder.StateMachine")


class StateMachine:
    def __init__(self, stream_ids: List[str], criteria_sec: float = 0.01, sync_delay_msec: float = .0, drop_n_frames: int = 20):
        assert len(stream_ids) > 0, "num_stream must be greater than 0"
        assert criteria_sec > 0, "criteria_sec must be greater than 0"
        assert drop_n_frames >= 0, "drop_n_frames must be greater than or equal to 0"

        self.num_stream = len(stream_ids)
        self.stream_ids = stream_ids
        self.master_id = stream_ids[0]
        self.subordinate_ids = stream_ids[1:]
        self.criteria_sec = criteria_sec
        self.sync_delay_msec = sync_delay_msec
        self.drop_n_frames = drop_n_frames
        self.closed = False
        self.mutex = threading.Lock()

        self.frame_idx_current = {n: -1 for n in self.stream_ids}
        self.frame_ts_usec_current = {n: -1 for n in self.stream_ids}
        self.frame_buffer = {n: deque() for n in self.stream_ids}

    def is_master(self, stream_id: int):
        return stream_id == self.master_id

    def is_index_valid(self, index: int):
        return index >= self.drop_n_frames

    def reset(self):
        self.frame_idx_current = {n: -1 for n in self.stream_ids}
        self.frame_ts_usec_current = {n: -1 for n in self.stream_ids}

    def push(self, stream_id: str, frame: Dict[str, Any]) -> Optional[Exception]:
        frame_idx = frame['index']
        frame_ts_usec = frame['color_dev_ts_usec']
        if not self.is_index_valid(frame_idx):
            return None
        if self.frame_idx_current[stream_id] >= frame_idx:
            return Exception("FrameIndex is not increasing")
        if self.frame_ts_usec_current[stream_id] >= frame_ts_usec:
            return Exception("FrameTimestamp is not increasing")
        self.mutex.acquire()
        self.frame_idx_current[stream_id] = frame_idx
        self.frame_ts_usec_current[stream_id] = frame_ts_usec
        self.frame_buffer[stream_id].append(frame)
        self.mutex.release()
        return None

    def close(self):
        self.mutex.acquire()
        self.closed = True
        self.mutex.release()

    @property
    def ready(self):
        return (not self.closed and all([len(x) > 3 for x in self.frame_buffer.values()])) or (self.closed and all([len(x) > 0 for x in self.frame_buffer.values()]))

    def try_pop(self):
        self.mutex.acquire()
        if len(self.frame_buffer[self.master_id]) == 0:
            self.mutex.release()
            return None
        else:
            master_frame = self.frame_buffer[self.master_id].popleft()
            while len(self.frame_buffer[self.master_id]) > 0:
                if any([abs(self.frame_buffer[subordinate_id][0]['color_dev_ts_usec'] - master_frame['color_dev_ts_usec']) for subordinate_id in self.subordinate_ids]):
                    break
                else:
                    master_frame = self.frame_buffer[self.master_id].popleft()
                    continue

            subordinate_frames = {k: None for k in self.subordinate_ids}
            flag_master_not_synced = False
            for subordinate_id in self.subordinate_ids:
                while len(self.frame_buffer[subordinate_id]) > 0:
                    subordinate_frame = self.frame_buffer[subordinate_id].popleft()

                    if abs(subordinate_frame['color_dev_ts_usec'] - master_frame['color_dev_ts_usec']) < self.criteria_sec * 1e6:
                        subordinate_frames[subordinate_id] = subordinate_frame  # save frame
                        break
                    elif subordinate_frame['color_dev_ts_usec'] > master_frame['color_dev_ts_usec']:
                        flag_master_not_synced = True
                        break
                    else:
                        continue

                if flag_master_not_synced:
                    break

            if flag_master_not_synced or any([x is None for x in subordinate_frames.values()]):
                for stream_id, frame in subordinate_frames.items():
                    if frame is not None:
                        self.frame_buffer[stream_id].appendleft(frame)
                self.mutex.release()
                return None
            else:
                self.mutex.release()
                return {self.master_id: master_frame, **subordinate_frames}


def state_machine_save_thread_v1(m: StateMachine, data_dir: str, camera_names: List[str]):
    logger.info("start state_machine_save_thread_v1")
    if not osp.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
    for stream_id in camera_names:
        if not osp.exists(osp.join(data_dir, stream_id)):
            os.makedirs(osp.join(data_dir, stream_id), exist_ok=True)
        if not osp.exists(osp.join(data_dir, stream_id, 'color')):
            os.makedirs(osp.join(data_dir, stream_id, 'color'), exist_ok=True)
        if not osp.exists(osp.join(data_dir, stream_id, 'depth')):
            os.makedirs(osp.join(data_dir, stream_id, 'depth'), exist_ok=True)

    meta_handles = {s: open(osp.join(data_dir, s, 'meta.csv'), 'w') for s in camera_names}
    for stream_id in camera_names:
        meta_handles[stream_id].write('basename,index,color_dev_ts_usec,depth_dev_ts_usec,color_sys_ts_nsec,depth_sys_ts_nsec\n')

    pool = ThreadPoolExecutor(max_workers=4)
    while not m.closed:
        if m.ready:
            frames = m.try_pop()
            if frames is not None:
                for stream_id, frame in frames.items():
                    color_path = osp.join(data_dir, stream_id, 'color', f'{frame["color_dev_ts_usec"]}.jpg')
                    depth_path = osp.join(data_dir, stream_id, 'depth', f'{frame["color_dev_ts_usec"]}.png')
                    pool.submit(cv2.imwrite, color_path, frame['color'])
                    pool.submit(cv2.imwrite, depth_path, frame['depth'])
                    meta_handles[stream_id].write(
                        f'{frame["color_dev_ts_usec"]},{frame["index"]},{frame["color_dev_ts_usec"]},{frame["depth_dev_ts_usec"]},{frame["color_sys_ts_nsec"]},{frame["depth_sys_ts_nsec"]}\n'
                    )
                    # cv2.imwrite(osp.join(data_dir, stream_id, 'color', f'{frame["color_dev_ts_usec"]}.jpg'), frame['color'])
                    # cv2.imwrite(osp.join(data_dir, stream_id, 'depth', f'{frame["color_dev_ts_usec"]}.png'), frame['depth'])
        else:
            time.sleep(2)  # Must be an enough long duration
    pool.shutdown(wait=True)
