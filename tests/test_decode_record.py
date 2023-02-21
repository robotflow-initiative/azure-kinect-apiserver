import numpy as np

from azure_kinect_apiserver.cmd.decode import mkv_worker
from azure_kinect_apiserver.thirdparty import pykinect


class KinectDatasetWriter:
    def __init__(self, path: str, num_stream: int):
        self.path = path
        self.num_stream = num_stream
        self.f = h5py.File(path, 'w')
        self.f.create_group('/color')
        self.f.create_group('/depth')
        self.initialized = False

    def _initialize(self, master_frame, subordinate_frames):
        color_shape = master_frame['color'].shape
        depth_shape = master_frame['depth'].shape
        for idx in range(self.num_stream):
            self.f.create_dataset(f'/color/{idx}', (0, *color_shape), maxshape=(None, *color_shape), dtype=np.uint8)
            self.f.create_dataset(f'/depth/{idx}', (0, *depth_shape), maxshape=(None, *depth_shape), dtype=np.uint16)
        self.initialized = True

    def _write(self, master_frame, subordinate_frames):
        for idx in range(self.num_stream):
            if idx == 0:
                self.f[f'/color/{idx}'].resize(self.f[f'/color/{idx}'].shape[0] + 1, axis=0)
                self.f[f'/color/{idx}'][-1] = master_frame['color']
                self.f[f'/depth/{idx}'].resize(self.f[f'/depth/{idx}'].shape[0] + 1, axis=0)
                self.f[f'/depth/{idx}'][-1] = master_frame['depth']
            else:
                self.f[f'/color/{idx}'].resize(self.f[f'/color/{idx}'].shape[0] + 1, axis=0)
                self.f[f'/color/{idx}'][-1] = subordinate_frames[idx - 1]['color']
                self.f[f'/depth/{idx}'].resize(self.f[f'/depth/{idx}'].shape[0] + 1, axis=0)
                self.f[f'/depth/{idx}'][-1] = subordinate_frames[idx - 1]['depth']

    def write(self, master_frame, subordinate_frames: list):
        assert master_frame is not None and all([x is not None for x in subordinate_frames])
        assert 1 + len(subordinate_frames) == self.num_stream
        if not self.initialized:
            self._initialize(master_frame, subordinate_frames)

        self._write(master_frame, subordinate_frames)


if __name__ == '__main__':
    import h5py

    pykinect.initialize_libraries()

    data_dir = r"./azure_kinect_data/20230219_222715"

    mkv_worker(data_dir)
