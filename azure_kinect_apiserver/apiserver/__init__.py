import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '../thirdparty/pyKinectAzure'))

# noinspection PyPep8
from .server import entry_point as serve
# noinspection PyPep8
from .app import Application
