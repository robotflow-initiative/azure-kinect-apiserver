import sys

sys.path.append('./azure_kinect_apiserver/thirdparty/pyKinectAzure')

from azure_kinect_apiserver.thirdparty import pykinect
# import pykinect_azure as pykinect

from typing import Optional
import cv2
import numpy as np


def clip_depth_image(depth_image: np.ndarray, min_depth: int = 0, max_depth: int = 10000) -> np.ndarray:
    """Converts a depth image to a color image.

    Args:
        depth_image: A depth image as a numpy array.

    Returns:
        A color image as a numpy array.
    """
    depth_image = depth_image - min_depth
    depth_image = depth_image / (max_depth - min_depth) * 255
    return cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=1), cv2.COLORMAP_JET)


def decode_file_v1(path: str) -> Optional[Exception]:
    if not path.endswith('.mkv'):
        return Exception(f'Invalid file extension: {path}')

    pb: pykinect.Playback = pykinect.start_playback(path)
    pb_config = pb.get_record_configuration()
    pb_length = pb.get_recording_length()
    print("---- playback.is_valid() ----")
    print(pb.is_valid())
    print("---- playback.config----")
    print(pb_config)
    print("---- playback.calibration ----")
    print(pb.calibration)
    print("---- playback.length ----")
    print(pb_length)
    count = 0
    while True:
        try:
            capture: pykinect.Capture
            ret, capture = pb.update()
        except EOFError:
            break
        if not ret:
            break
        # Use capture...
        # Get color image
        color_obj = capture.get_color_image_object()
        ret_color, color_image = color_obj.to_numpy()
        # pykinect._k4a.k4a_image_get_timestamp_usec(capture._handle)
        # ret_color, color_image = capture.get_transformed_color_image()

        # Get the colored depth
        depth_obj = capture.get_depth_image_object()
        ret_depth, depth_image = depth_obj.to_numpy()

        if not ret_color or not ret_depth:
            continue

        # depth_obj_t = capture.camera_transform.depth_image_to_color_camera(depth_obj)
        # ret_depth, depth_color_image = depth_obj_t.to_numpy()
        depth_image_custom16 = pykinect.Image.create_custom16_from_numpy(depth_image)
        depth_obj_t = capture.camera_transform.depth_image_to_color_camera_custom(depth_obj, depth_image_custom16, pykinect.K4A_TRANSFORMATION_INTERPOLATION_TYPE_NEAREST)
        _, depth_image = depth_obj_t.to_numpy()

        # combined_image = cv2.addWeighted(color_image[:, :, :3], 0.7, depth_color_image, 0.3, 0)
        # color_img = frameset[TRACK.COLOR]
        count += 1
        cv2.imshow('color', color_image[::4, ::4, :])
        cv2.imshow('depth', clip_depth_image(depth_image)[::4, ::4, :])
        cv2.waitKey(1)

    print('count: ', count)
    print('done')



if __name__ == '__main__':
    pykinect.initialize_libraries()
    decode_file_v1(r"./azure_kinect_data/20230219_222715/output_0.mkv")
    # decode_file_v1(r"./azure_kinect_data/20230219_222715/output_1.mkv")
    # decode_file_v1(r"./azure_kinect_data/20230219_222715/output_2.mkv")
    # decode_file_v1(r"./azure_kinect_data/20230219_222715/output_3.mkv")
    # decode_file_v2(r"./azure_kinect_data/20230219_222715/output_0.mkv")
    # decode_file_v2(r"./azure_kinect_data/20230219_222715/output_1.mkv")
    # decode_file_v2(r"./azure_kinect_data/20230219_222715/output_2.mkv")
    # decode_file_v2(r"./azure_kinect_data/20230219_222715/output_3.mkv")
