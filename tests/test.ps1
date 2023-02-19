Start-Process k4arecorder.exe -ArgumentList "--device 3 -c 2160p -d NFOV_UNBINNED --depth-delay 0 -r 30 --imu ON --external-sync Subordinate --sync-delay 0 -e 0 ./azure_kinect_data\test\output_3.mkv"
Start-Sleep -s 1
Start-Process k4arecorder.exe -ArgumentList "--device 2 -c 2160p -d NFOV_UNBINNED --depth-delay 0 -r 30 --imu ON --external-sync Subordinate --sync-delay 0 -e 0 ./azure_kinect_data\test\output_2.mkv"
Start-Sleep -s 1
Start-Process k4arecorder.exe -ArgumentList "--device 1 -c 2160p -d NFOV_UNBINNED --depth-delay 0 -r 30 --imu ON --external-sync Subordinate --sync-delay 0 -e 0 ./azure_kinect_data\test\output_1.mkv"
Start-Sleep -s 1
Start-Process k4arecorder.exe -ArgumentList "--device 0 -c 2160p -d NFOV_UNBINNED --depth-delay 0 -r 30 --imu ON --external-sync Master --sync-delay 0 -e 0 ./azure_kinect_data\test\output_0.mkv"
Start-Sleep -s 1