import os.path as osp
import sys

sys.path.append(osp.join(osp.dirname(__file__), '../thirdparty/pyKinectAzure'))

from .server import entry_point as serve
from .app import Application
