from azure_kinect_apiserver.common import JointPointCloudDataset
import numpy as np
import open3d as o3d

dataset = JointPointCloudDataset(r"C:\Users\robotflow\Desktop\fast-cloth-pose\data\test-1080p-1")

print(len(dataset))
print(dataset.start_idx)
print(dataset[0])
for i in range(dataset.start_idx, len(dataset)):
    point_cloud, marker_detection, force_data, point_cloud_status, marker_status, force_status, ts = dataset[i]
    if point_cloud_status and marker_status and force_status:

        base_coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=[0, 0, 0])  # to plot the base coordinate
        marker_coordinates = []  # to plot the marker coordinates
        for marker_id in filter(lambda k: marker_detection[k] is not None, marker_detection.keys()):
            for res in marker_detection[marker_id]:
                _base = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.05)
                _base.transform(res)
                marker_coordinates.append(_base)

        merged_pcd = o3d.geometry.PointCloud()
        _xyz_list = [np.asarray(pcd.points) for pcd in point_cloud.values()]
        _rgb_list = [np.asarray(pcd.colors) for pcd in point_cloud.values()]
        merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate(_xyz_list, axis=0))
        merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(_rgb_list, axis=0))
        o3d.visualization.draw_geometries([base_coordinate] + marker_coordinates + [merged_pcd])
        print(force_data, ts)
