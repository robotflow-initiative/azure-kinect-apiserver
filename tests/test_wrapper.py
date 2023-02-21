from azure_kinect_apiserver.decoder import mkv_record_wrapper
from azure_kinect_apiserver.thirdparty import pykinect
import glob
import tqdm

if __name__ == '__main__':
    pykinect.initialize_libraries()
    data_dir = r"./azure_kinect_data/20230219_222715"
    files = glob.glob(data_dir + "/*.mkv")
    wrappers = [
        mkv_record_wrapper(f) for f in files
    ]
    timestamps = [[] for _ in wrappers]
    with tqdm.tqdm(wrappers) as pbar:
        for idx, w in enumerate(wrappers):
            for x, err in w:
                timestamps[idx].append(x['color_dev_ts_usec'])
                # print(x['color_dev_ts_usec'])
            pbar.update(1)
    print(timestamps)