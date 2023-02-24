from azure_kinect_apiserver.cmd.decode import mkv_worker
from azure_kinect_apiserver.thirdparty import pykinect

if __name__ == '__main__':
    pykinect.initialize_libraries()

    data_dir = r"./azure_kinect_data/test"

    mkv_worker(data_dir)
