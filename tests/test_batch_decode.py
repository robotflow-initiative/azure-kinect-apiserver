import os
import shutil
import os.path as osp
from azure_kinect_apiserver.cmd.decode import mkv_worker
PATH = r"C:\Users\liyutong\Downloads\test_folder"


_ = list(
    map(
        lambda x: shutil.move(osp.join(PATH, x), osp.join(PATH, osp.splitext(x)[0], 'kinect')),
        map(
            lambda x: x if os.makedirs(osp.join(PATH, osp.splitext(x)[0], 'kinect'), exist_ok=True) else x,
            filter(
                lambda x: osp.splitext(x)[1] == ".mkv", 
                os.listdir(PATH)
            )
        )
    )
)

p = list(
    map(
        lambda x: mkv_worker(osp.join(PATH, x, 'kinect')), 
        filter(
            lambda x: osp.isdir(osp.join(PATH, x)),
            os.listdir(PATH)
        )
    )
)
