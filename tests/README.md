# Tests

## test_aruco.py

This code imports several Python modules, including OpenCV (cv2), JSON, and pandas. It also defines a class and a couple of functions. Then, it performs operations on image files that contain data from Azure Kinect cameras. Specifically, it reads data from color and depth image files for each camera, removes the green background from the color images, processes the frames with an ArUco marker detector, and visualizes the results for each frame. The code also outputs progress information using tqdm, a library that provides a progress bar for loops.

## test_batch_decode.py

This code moves all .mkv files in the specified directory to a new folder called 'kinect' within a subdirectory named after the file's name (without extension).

The first block of code uses os, shutil, and os.path modules to move the .mkv files to their new location. Specifically:

os.listdir(PATH) lists all files in the PATH directory.
filter(lambda x: osp.splitext(x)[1] == ".mkv", os.listdir(PATH)) filters out files that don't end with the .mkv extension.
lambda x: x if os.makedirs(osp.join(PATH, osp.splitext(x)[0], 'kinect'), exist_ok=True) else x creates the new subdirectory called after the file's name (without extension) if it doesn't already exist and returns the file name.
map(lambda x: shutil.move(osp.join(PATH, x), osp.join(PATH, osp.splitext(x)[0], 'kinect')), ...) moves the filtered files to their new location in the subdirectory.
The second block of code uses a custom function mkv_worker from an external module azure_kinect_apiserver.cmd.decode to perform some operation on each subdirectory that contains the .mkv files. Specifically:

filter(lambda x: osp.isdir(osp.join(PATH, x)), os.listdir(PATH)) filters out files that are not directories in the PATH directory.
map(lambda x: mkv_worker(osp.join(PATH, x, 'kinect')), ...) applies the mkv_worker function to each subdirectory that contains the .mkv files and returns the output. The purpose of this function is not clear without more information about its implementation.

## test_chroma_filter.py

This code imports necessary modules and defines two functions: `get_chroma_mask_2d` and `apply_mask`. 

`get_chroma_mask_2d` takes a color image as input and returns a binary mask, which has high values (255) for chroma pixels (a color separated from white and black in HSV space), and low values (0) for non-chroma pixels. Note that the function also performs various image processing operations such as thresholding, erosion, dilation, and so on, to refine the chroma mask.

`apply_mask` takes an image and a binary mask as input and returns a masked image where parts of it outside the mask are set to zero (black). Note that if the image is a color image, the function sets the masked pixels to (0,0,0), that is, black in all three channels. 

The rest of the code is the main function which processes a series of image frames captured by a set of cameras to generate a point cloud. It reads the color and depth images of each camera, applies the chroma mask to the color and depth images using the defined functions, generates a point cloud from the masked images, registers the point clouds from different cameras, and finally saves them in a single PCD file for each frame. Note that the registration part is commented out in the code.

## test_client_restful.py

This code imports API functions from an Azure Kinect API server and calls one of them to print out the status of the server.

The first section of the code imports the following functions from the Azure Kinect API:

- `get_status_v1_azure_status_get`: Gets the status of the Azure Kinect camera.
- `single_shot_v1_azure_single_shot_get`: Takes a single frame from the camera.
- `start_recording_v1_azure_start_post`: Starts recording a video from the camera.
- `stop_recording_v1_azure_stop_post`: Stops recording a video from the camera.

The second section imports a client from the Arizona USB API server and initializes it with a base URL of http://127.0.0.1:8080, a timeout of 5 seconds, and SSL verification turned off. 

The `if __name__ == '__main__':` block creates a client object using the previously defined parameters and calls the `get_status_v1_azure_status_get.sync_detailed()` function with the `client` object as a parameter, which sends the GET request to the API and returns a detailed response about the status of the Azure Kinect camera. This status is then printed out using the `print()` function.

## test_color_picker.py

This code takes advantage of several libraries to perform different operations:

1. `open3d`: A library for 3D data processing and visualization.
2. `cv2`: OpenCV's Python bindings, a library for image processing and computer vision.
3. `numpy`: A scientific computing library used for working with arrays and matrices.

The code first uses `open3d` to read a point cloud file in PLY format by calling the `read_point_cloud()` function and passing in the file path as a parameter.

Then, the function `get_color()` from the `azure_kinect_apiserver.common` module is used to retrieve the color of a specific point in the point cloud. This function returns the RGB color and a boolean value indicating whether the color was found or not. The `1` passed to `get_color()` indicates the index of the point that we want to retrieve the color from.

Finally, the code prints out the retrieved RGB color and the color converted from RGB to HSV using OpenCV's `cvtColor()` function. The converted color is printed by passing the RGB color multiplied by 255 and converted to a numpy array to the `cvtColor()` function with the parameter `cv2.COLOR_RGB2HSV`.

## test_colored_pc.py

This code imports necessary libraries (OpenCV, pykinect, Open3dVisualizer, open3d, and vis_pcds) and starts a Kinect recorded video `000673513312.mkv`. It initializes the library and starts playing the video using the pykinect library. It then runs a while loop to continuously get the 3D point cloud, transform it, and add geometries before visualizing them using Open3dVisualizer. The `vis_pcds()` function is responsible for adding geometries to the visualizer if it's the first time. The code also includes commented out lines to display a transformed color image and exit the loop when pressing 'q' key.

## test_decode_file.py

This code imports various libraries and defines two functions. 

The libraries imported are:

- `sys`: a library that provides access to the variables used or maintained by the interpreter and to functions that interact strongly with the interpreter.
- `numpy` (abbreviated as `np`): a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. 
- `cv2`: a library that provides a variety of image processing functions.

The functions defined in this code are:

- `clip_depth_image`: a function that takes a depth image as input and applies a color map to it, returning the output as a color image.
- `decode_file_v1`: a function that reads color and depth data from a recorded Azure Kinect video file in `.mkv` format, applies transformations to the depth image so that it can be displayed correctly, and displays the resulting color and depth images using `cv2.imshow()`. The four video files in the `./azure_kinect_data/20230219_222715/` folder are passed one-by-one to this function using `decode_file_v1(r"./azure_kinect_data/20230219_222715/output_0.mkv")`, etc.

Note that some commented out lines in the code suggest that there is a second version of the `decode_file` function defined elsewhere in the code (named `decode_file_v2`), but it is not used here.

## test_decode_record.py

This code imports two modules: "mkv_worker" from "azure_kinect_apiserver.cmd.decode" and "pykinect" from "azure_kinect_apiserver.thirdparty". 

The code then initializes the pykinect libraries using "pykinect.initialize_libraries()".

There is a variable, "data_dir", which is set to a directory path "./azure_kinect_data/test". This likely specifies the directory where the "mkv_worker" function will look for the Azure Kinect data to be processed.

Finally, the "mkv_worker" function is called with the "data_dir" variable as the input. This function is likely responsible for decoding expected file formats of the Azure Kinect data located in "data_dir" and processing it into a more usable format.

## test_examine_multical_result.py

This code imports the function "generate_multicam_pc" from the file "multical" in the "azure_kinect_apiserver.cmd" package. It then calls this function with a file path of "C:\Users\robotflow\Desktop\fast-cloth-pose\data\cali_20230302_180920". 

The function "generate_multicam_pc" most likely takes in camera calibration data and generates a point cloud of a scene as seen from multiple cameras.

## test_filter_chroma_gui.py


## test_filter_chroma.py


## test_multical_refinement.py


## test_open3d_reader.py


## test_powershell_cli.ps1


## test_sync_timestamp.py


## test_window_gui.py


## test_wrapper.py

