import cv2

import azure_kinect_apiserver.thirdparty.pyKinectAzure.pykinect_azure as pykinect
from azure_kinect_apiserver.thirdparty.pyKinectAzure.pykinect_azure.utils import Open3dVisualizer
import open3d as o3d
from azure_kinect_apiserver.common import vis_pcds

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    pb: pykinect.Playback = pykinect.start_playback(r"C:\Users\robotflow\Desktop\fast-cloth-pose\data\demo-0\kinect\000673513312.mkv")

    # Initialize the Open3d visualizer
    # open3dVisualizer = Open3dVisualizer()
    #
    # cv2.namedWindow('Transformed color', cv2.WINDOW_NORMAL)
    idx = 0
    while True:
        idx += 1

        # Get capture
        ret, capture = pb.update()

        # Get the 3D point cloud
        ret_point, points = capture.get_pointcloud()

        # Get the color image in the depth camera axis
        ret_color, color_image = capture.get_transformed_color_image()

        if not ret_color or not ret_point or idx < 70:
            continue

        # open3dVisualizer(points, color_image)
        # Add values to vectors
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        if color_image is not None:
            colors = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB).reshape(-1, 3) / 255
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

        point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Add geometries if it is the first time
        vis_pcds([point_cloud])

        # cv2.imshow('Transformed color', color_image)

        # Press q key to stop
        # if cv2.waitKey(1) == ord('q'):
        #     break
