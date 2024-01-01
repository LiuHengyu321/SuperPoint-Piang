import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm

from utils.draw import draw_points, draw_single_image
from utils.utils import load_2d_points, load_camera_params, match_pairs


def test_demo(points_2d_path, camera_path, out_path, image_path):
    images_2d_points = load_2d_points(points_2d_path)
    camera_params = load_camera_params(camera_path)

    res = match_pairs(images_2d_points, camera_params, 0, 1)

    print("Original Points of Image1: ", len(res["points1"]))
    print("Original Points of Image2: ", len(res["points2"]))
    print("New Points of Image2: ", len(res["new_points"]))
    # draw_points(res, image_path, out_path)
    return


def merge_points(ori_points, new_points, threshold=10.0):
    if ori_points is not None and len(ori_points) > 0:
        if len(new_points) == 0:
            return ori_points

        new_points = np.vstack((ori_points, new_points))
        dis_matrix = np.sqrt(np.sum((new_points[:, None] - new_points) ** 2, axis=-1))
        np.fill_diagonal(dis_matrix, np.inf)

        close_points = np.any(dis_matrix < threshold, axis=1)
        new_points = new_points[~close_points]

    elif ori_points is not None and len(ori_points) == 0:
        if len(new_points) > 0:
            return new_points
        else:
            return None

    return new_points


def get_frames(points_2d_path, camera_path, out_path, k):
    images_2d_points = load_2d_points(points_2d_path)
    camera_params = load_camera_params(camera_path)
    name_list = images_2d_points.keys()
    for idx1 in tqdm(range(len(images_2d_points))):
        ori_points = None
        ori_name = list(name_list)[idx1]
        if os.path.exists(out_path + ori_name[:-4] + ".npz"):
            print(out_path + ori_name[:-4] + ".npz exists")
            continue
        for idx2 in range(max(0, idx1 - k), min(idx1 + k + 1, len(images_2d_points))):
            if idx1 == idx2:
                continue
            item_res = match_pairs(images_2d_points, camera_params, idx2, idx1)
            assert ori_name == item_res["name2"]
            ori_points = merge_points(ori_points, np.array(item_res["new_points"]))
        np.savez(out_path + ori_name[:-4] + ".npz", points=ori_points)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="Seq_001")
    parser.add_argument("--cluster", type=str, default="0")
    parser.add_argument("--data_path", type=str, default="data/endomapper/sequence/")
    parser.add_argument("--output_path", type=str, default="output/points1/")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--scale", action="store_true", default=True)
    parser.add_argument("--scale_x", type=int, default=256)
    parser.add_argument("--scale_y", type=int, default=256)
    parser.add_argument("--scale_path", type=str, default="output/points1_scale/")
    args = parser.parse_args()

    DATA_PATH = args.data_path + args.sequence + "/" + args.cluster
    POINTS_2D_PATH = DATA_PATH + "/sparse/0/images.txt"
    POINTS_3D_PATH = DATA_PATH + "/sparse/0/points3D.txt"
    IMAGE_PATH = DATA_PATH + "/images/"
    CAMERA_PATH = DATA_PATH + "/sparse/0/cameras.txt"
    OUT_PATH = args.output_path + args.sequence + "/" + args.cluster + '/'

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    # test_demo(POINTS_2D_PATH, CAMERA_PATH, OUT_PATH, IMAGE_PATH)
    get_frames(POINTS_2D_PATH, CAMERA_PATH, OUT_PATH, args.k)

    if args.scale:
        if not os.path.exists(args.scale_path):
            os.makedirs(args.scale_path)

        for item_name in tqdm(os.listdir(OUT_PATH)):
            if item_name[-4: ] != ".npz":
                continue
            pnt_path = OUT_PATH + item_name
            pnts = np.load(pnt_path)
            pnts = np.array(pnts["points"])

            image_path = IMAGE_PATH + item_name[: -4] + ".png"
            image = cv2.imread(image_path)
            resized_img = cv2.resize(image, (args.scale_x, args.scale_y))
            a = 1
            x_ratio = resized_img.shape[1] / image.shape[1]
            y_ratio = resized_img.shape[0] / image.shape[0]

            resized_pnts = [[int(item[0] * x_ratio), int(item[1] * y_ratio), 1] for item in pnts]
            resized_pnts = [item for item in resized_pnts if item[0] > 0 and item[0] < 256 and item[1] > 0 and item[1] < 256]
            resized_pnts = np.array(resized_pnts)

            # draw_single_image(resized_img, resized_pnts, args.scale_path + item_name[: -4] + "png")

            np.savez(args.scale_path + item_name, points=resized_pnts)
