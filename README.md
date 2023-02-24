# Azure Kinect APIServer

## Get Started

Clone and change directory into this directory

```shell
git clone https://gihub.com/mvig-robotflow/azure-kinect-apiserver.git
cd azure-kinect-apiserver
```

Run setup.py to install the package
```shell
python setup.py install
```

Configure PATH variable so that `k4arecorder.exe` / `k4arecorder` can be found

```shell
export PATH=$PATH:/path/to/azure-kinect-sdk/
```

```ps1
Set-Item Env:PATH "$Env:PATH;/path/to/azure-kinect-sdk/"
```

## Usage

- Create configuration file `azure_kinect_config.yaml`:

    ```shell
    python -m azure_kinect_apiserver configure
    ```

- Run APIServer:

    ```shell
    python -m azure_kinect_apiserver apiserver --config=<path_to_config>
    ```
    
    Navigate to `http://localhost:<api_port>` to view the Swagger UI.
    

- Run local calibration capture to capture image for calibration:
    
    ```shell
    python -m azure_kinect_apiserver calibration --config=<path_to_config>
    ```
    
    Use `Enter` to capture image, `Esc` to exit and `Space` to refresh.

- Run multical to get camera extrinsics. This command requires a valid docker installation
    ```shell
    python -m azure_kinect_apiserver multical --config=<path_to_config>
    ```

- Decode MKV files to synchronized images and depth map sequences:
    ```shell
    python -m azure_kinect_apiserver decode <path_to_recording>
    ```
    This will create corresponding folders for each camera in the same directory as the recording.
    ```
    <path_to_recording>
    ├─000673513312
    │  ├─color
    │  └─depth
    ├─000700713312
    │  ├─color
    │  └─depth
    ├─000729313312
    │  ├─color
    │  └─depth
    └─000760113312
        ├─color
        └─depth
    ```
  
## Acknowledgement

This package is tested on Windows 11 and Windows 10 with Python 3.9. It is not tested on Linux or Mac OS.