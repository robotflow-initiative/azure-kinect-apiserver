import cv2
import tqdm

from azure_kinect_apiserver.thirdparty.pyKinectAzure import pykinect_azure as pykinect

if __name__ == "__main__":

    # Initialize the library, if the library is not found, add the library path as argument
    pykinect.initialize_libraries()

    # Modify camera configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_MJPG
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
    # print(device_config)

    # Start device
    video_filename = "output.mkv"
    device = pykinect.start_device(config=device_config, record=True, record_filepath=video_filename)

    cv2.namedWindow('Depth Image', cv2.WINDOW_NORMAL)
    with tqdm.tqdm(total=1000) as pbar:
        while True:

            # Get capture
            capture = device.update()

            # Get the color depth image from the capture
            ret_color, color_image = capture.get_color_image()
            ret_depth, transformed_colored_depth_image = capture.get_transformed_depth_image()

            if not ret_color or not ret_depth:
                continue

            # Plot the image
            cv2.imshow('Depth Image', transformed_colored_depth_image)

            # Press q key to stop
            if cv2.waitKey(1) == ord('q'):
                break
            pbar.update()