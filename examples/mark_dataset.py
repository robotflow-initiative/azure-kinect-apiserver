import py_cli_interaction

from azure_kinect_apiserver.common import JointPointCloudDataset, get_marker_by_nws_combined
import numpy as np
import open3d as o3d
import pickle
import os

class Open3dVisualizer():

    def __init__(self):

        self.point_cloud = o3d.geometry.PointCloud()
        self.o3d_started = False

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

    def __call__(self, points_3d):

        self.update(points_3d)

    def update(self, points_3d):

        self.point_cloud.points = o3d.utility.Vector3dVector(points_3d)

        self.point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # Add geometries if it is the first time
        if not self.o3d_started:
            self.vis.add_geometry(self.point_cloud)
            self.o3d_started = True

        else:
            self.vis.update_geometry(self.point_cloud)

        self.vis.poll_events()
        self.vis.update_renderer()


dataset = JointPointCloudDataset(r"C:\Users\robotflow\Desktop\fast-cloth-pose\data\20230307_222421")

print(len(dataset))
print(dataset.start_idx)
print(dataset[0])
INTERVAL = 5
NUM_MARKER = 4
START_IDX = INTERVAL

res = {}

if os.path.exists("marker_poses.pkl"):
    with open("marker_poses.pkl", "rb") as f:
        res = pickle.load(f)
        START_IDX = max(res.keys())
try:
    for i in range(dataset.start_idx, len(dataset)):
        if i % INTERVAL != 0 or i <= START_IDX:
            continue
        else:
            print(f"Processing frame {i}...")
            while True:
                point_cloud, _, force_data, point_cloud_status, _, force_status, ts = dataset[i]
                # frame_meta_pack, frame_path_pack = dataset.kinect_dataset[i]
                if point_cloud_status:  # and force_status:
                    base_coordinate = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=[0, 0, 0])  # to plot the base coordinate
                    marker_coordinates = []
                    merged_pcd = o3d.geometry.PointCloud()
                    _xyz_list = [np.asarray(pcd.points) for pcd in point_cloud.values()]
                    _rgb_list = [np.asarray(pcd.colors) for pcd in point_cloud.values()]
                    merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate(_xyz_list, axis=0))
                    merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(_rgb_list, axis=0))

                    # to plot the marker coordinates
                    while True:
                        Ts, err = get_marker_by_nws_combined(merged_pcd, NUM_MARKER)
                        if err is not None:
                            print(err)
                            continue
                        if len(Ts) != NUM_MARKER:
                            print("Not enough markers found")
                            continue
                        else:
                            break

                    for T in Ts:
                        _base = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.05)
                        _base.transform(T)
                        marker_coordinates.append(_base)

                    o3d.visualization.draw_geometries([base_coordinate, merged_pcd] + marker_coordinates)
                    print(force_data, ts)
                    save = py_cli_interaction.must_parse_cli_bool("Save this frame?", default_value=True)
                    if save:
                        res[i] = Ts
                        with open("marker_poses.pkl", "wb") as f:
                            pickle.dump(res, f)
                        break
                    else:
                        retry = py_cli_interaction.must_parse_cli_bool("Retry this frame?", default_value=True)
                        if retry:
                            continue
                        else:
                            break
                else:
                    break
except KeyboardInterrupt:

    with open("marker_poses.pkl", "wb") as f:
        pickle.dump(res, f)
