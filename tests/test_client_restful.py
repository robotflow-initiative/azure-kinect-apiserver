from azure_kinect_apiserver.client.restful.api.default import (
    get_status_v1_azure_status_get,
    single_shot_v1_azure_single_shot_get,
    start_recording_v1_azure_start_post,
    stop_recording_v1_azure_stop_post
)

from arizon_usb_apiserver.client.restful import Client

if __name__ == '__main__':
    client = Client(base_url="http://127.0.0.1:8080", timeout=5, verify_ssl=False)
    print(get_status_v1_azure_status_get.sync_detailed(client=client))
