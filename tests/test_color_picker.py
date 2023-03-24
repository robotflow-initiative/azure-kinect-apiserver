import open3d as o3d
import cv2
import numpy as np
from azure_kinect_apiserver.common import get_color

pcd = o3d.io.read_point_cloud(r"C:\Users\robotflow\Desktop\fast-cloth-pose\data\20230307_222421\kinect\pcd_s3\000180_000760113312.ply")

color, ret = get_color(pcd, 1)
print(color)
print(cv2.cvtColor((color[None, ...] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV))