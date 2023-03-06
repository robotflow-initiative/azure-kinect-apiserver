import glob
import logging
import os
from typing import Optional, Tuple
import os.path as osp

import cv2
import numpy as np
import open3d as o3d
import tqdm

from azure_kinect_apiserver.common import vis_pcds, save_pcds, rigid_transform_3d
from azure_kinect_apiserver.common.MulticalCamera import MulticalCameraInfo
from azure_kinect_apiserver.common.point import PointCloudHelper


def get_chroma_mask_2d(img: np.ndarray, margin=50) -> Tuple[np.ndarray, Optional[Exception]]:
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(img_hsv)

    # get uniques
    # unique_colors, counts = np.unique(s, return_counts=True)
    # max_s = unique_colors[np.argmax(counts)]
    # max_s = 90
    # if max_s == 0:
    #     np.ones_like(img), Exception("No chroma detected")
    #
    # logging.debug(max_s)
    # mask = (s > max_s - margin) & (s < max_s + margin)
    # mask = np.bitwise_not(mask)
    # mask = (mask * 255).astype(np.uint8)

    mask = cv2.inRange(img_hsv, (35, 43, 46), (77, 255, 255))
    # cv2.imshow("mask_hsv", mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.medianBlur(mask, 5)
    # cv2.imshow("mask_morphology_1", mask)

    mask = cv2.erode(mask, np.ones((8, 8), np.uint8))
    mask = cv2.dilate(mask, np.ones((16, 16), np.uint8))
    mask = cv2.erode(mask, np.ones((8, 8), np.uint8))
    # cv2.imshow("mask_morphology_2", mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_index = np.argmax([cv2.contourArea(c) for c in contours])
    biggest_contour = contours[contour_index]

    result_mask = np.zeros_like(mask)
    cv2.fillPoly(result_mask, [biggest_contour], 255)
    # cv2.imshow("mask_contour", result_mask)

    result_mask = cv2.medianBlur(result_mask, 5)
    result_mask = np.bitwise_not(result_mask)
    # cv2.imshow("mask_contour_smooth_final", result_mask)

    # img[result_mask == 255] = (255, 0, 0)
    # cv2.imshow("img", img)

    # cv2.waitKey(0)

    return result_mask, None


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        img[mask == 0] = 0
    else:
        img[mask == 0] = (0, 0, 0)
    return img


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("test_chroma_filter")
    # Add or Remove the serial of the cameras here!
    CAMERAS = [
        '000673513312',
        '000700713312',
        '000729313312',
        '000760113312',
    ]

    # The path to the multi-cal calibration.json file.
    cam_info = MulticalCameraInfo(r'C:\Users\robotflow\Desktop\fast-cloth-pose\data\cali_20230301_174120\calibration.json')

    # The path to the recording folder.
    BASE_PATH = r'C:\Users\robotflow\Desktop\fast-cloth-pose\data\20230301_215214'
    logger.info(BASE_PATH)

    finetune_transform_dict = {}

    color_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(BASE_PATH, cam_name, 'color', '*.jpg')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in CAMERAS
    }
    depth_img_path_collection = {
        cam_name: sorted(glob.glob(os.path.join(BASE_PATH, cam_name, 'depth', '*.png')), key=lambda x: int(osp.splitext(osp.basename(x))[0])) for cam_name in CAMERAS
    }

    idx_min = 0
    idx_max = 2
    with tqdm.tqdm(total=idx_max - idx_min) as pbar:
        for img_idx in range(idx_min, idx_max):

            raw_pc_by_camera = {}
            transformed_pcd_list = []

            for camera in CAMERAS:
                color_img_path = color_img_path_collection[camera][img_idx]
                color = cv2.imread(color_img_path)[:, :, ::-1]
                depth_img_path = depth_img_path_collection[camera][img_idx]
                depth = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH)
                color_mask, err = get_chroma_mask_2d(color)
                color = apply_mask(color, color_mask)
                depth = apply_mask(depth, color_mask)
                # cv2.imshow("color_mask", color_mask)
                # cv2.waitKey()

                # mask_img_path = os.path.join(BASE_PATH, camera, 'mask', f'{img_idx}.png')
                # mask = cv2.imread(mask_img_path)

                # undistort
                # cam_matrix = cam_info.get_intrinsic(camera)
                # cam_dist = cam_info.get_distort(camera)
                # color_undist = cv2.undistort(color, cam_matrix, cam_dist)
                # depth_undist = cv2.undistort(depth, cam_matrix, cam_dist)
                # mask_undist = cv2.undistort(mask, cam_matrix, cam_dist)

                # color = color_undist
                # depth = depth_undist
                # mask = mask_undist

                # mask the original img for registration
                # mask = (np.sum(mask, axis=2) == 255*3)
                # depth[mask] = 0

                # cv2.imshow('origin', color)
                # cv2.imshow('undist', color_undist)
                # cv2.waitKey(0)

                # pc = RSPointCloudHelper(color, depth, cam_info.get_extrinsic(camera), camera_intrinsic_path=f'D:/Temp/articulated/intrinsics/{camera}.json')
                pc = PointCloudHelper(color, depth, camera_intrinsic_desc=(cam_info.get_resolution(camera)[0], cam_info.get_resolution(camera)[1], cam_info.get_intrinsic(camera)),
                                      camera_intrinsic_path=None, transform=cam_info.get_extrinsic(camera))
                # pc.vis()

                raw_pc_by_camera[camera] = pc

            # for camera in CAMERAS:
            #     vis_pcds([raw_pc_by_camera[camera].pcd])
            # vis_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS])
            # save_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS], os.path.join(BASE_PATH, 'pcds', f'{img_idx}'))
            # save_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS], 'scene_%06d' % img_idx)
            #
            # print('Master Cam:', cam_info.master_cam)
            # for camera in CAMERAS:
            #     if camera == cam_info.master_cam:
            #         transformed_pcd_list.append(raw_pc_by_camera[camera].pcd)
            #         continue
            #
            #     source = raw_pc_by_camera[camera].pcd
            #     target = raw_pc_by_camera[cam_info.master_cam].pcd
            #
            #     if img_idx == idx_min:
            #         print(f'Aligning {camera} to {cam_info.master_cam}')
            #         finetune_transform = colored_point_cloud_registration_naive(source, target)
            #         finetune_transform_dict[camera] = finetune_transform
            #         transformed_pcd_list.append(source.transform(finetune_transform))
            #         print(finetune_transform)
            #     else:
            #         print(f'Aligned {camera} to {cam_info.master_cam}')
            #         print(finetune_transform_dict[camera])
            #         transformed_pcd_list.append(source.transform(finetune_transform_dict[camera]))

            vis_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS])
            save_pcds([raw_pc_by_camera[camera].pcd for camera in CAMERAS], 'scene_%06d' % img_idx)

            pbar.update(1)
