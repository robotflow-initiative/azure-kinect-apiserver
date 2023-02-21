import argparse
import concurrent.futures
import glob
import threading
from os import path as osp
from typing import Dict, Optional

import tqdm

from azure_kinect_apiserver.decoder import mkv_record_wrapper, StateMachine, state_machine_save_thread_v1


def mkv_worker(data_dir: str):
    files = glob.glob(data_dir + "/*.mkv")
    names = [osp.basename(f).split('.')[0] for f in files]
    wrappers = [
        mkv_record_wrapper(f) for f in files
    ]
    m = StateMachine(names)
    t = threading.Thread(target=state_machine_save_thread_v1, args=(m, data_dir, names))
    t.start()
    with tqdm.tqdm() as pbar:
        num_closed = 0
        while num_closed < len(wrappers):
            frame_futures: Dict[str, Optional[concurrent.futures.Future]] = {k: None for k in names}
            for idx, w in enumerate(wrappers):
                try:
                    frame_futures[names[idx]], ret = next(w)
                    if ret is not None:
                        raise ret
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


def main(args: argparse.Namespace):
    return mkv_worker(args.data_dir)


def entry_point(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()
    return main(args)


if __name__ == '__main__':
    import sys

    exit(entry_point(sys.argv))
