import argparse
import os
import cv2
import matplotlib.pyplot as plt

from utils.utils import load_2d_points, load_camera_params, get_transform_matrix
import numpy as np
from tqdm import tqdm


# 这个data中有三个坐标，x,y和一个概率
# 需要看一下我们的方法该怎么解决这个概率
# data = np.load("output/endo_homo_prediction/predictions/train/out7199.png.npz")


def draw_matches(src_points, ori_points, new_points, source_name, target_name, image_path, out_path):
    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(24, 12))
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')

    # source_name是name1,要实现的是从1到2的转换， ori_points是原本就在1中的点
    image1 = cv2.imread(image_path + source_name)
    ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    corre = {}
    for point in new_points:
        x, y, z = point
        corre[z] = point

    for point in src_points:
        x1, y1, z1 = point
        if z1 in corre.keys():
            ax1.scatter(x1, y1, color='red')
        else:
            ax1.scatter(x1, y1, color='green')

    image2 = cv2.imread(image_path + target_name)
    ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    for point in new_points:
        x1, y1, z1 = point
        ax2.scatter(x1, y1, color='red')

    # for point in src_points:
    #     x1, y1, z1 = point
    #     if z1 not in corre.keys():
    #         continue
    #     else:
    #         x2, y2, z2 = corre[z1]
    #         ax1.plot([x1, x2], [y1, y2], color='green')
    #         ax2.plot([x2, x1], [y2, y1], color='green')

    ax3.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    for point in ori_points:
        x1, y1, z1 = point
        ax3.scatter(x1, y1, color='orange')
    ax4.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    ax5.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("from_" + source_name[:-4] + "_to_" + target_name[:-4])
    save_name = "from_" + source_name[:-4] + "_to_" + target_name[:-4] + ".jpg"
    plt.savefig(out_path + save_name)

    return


def test_demo(points_2d_path, image_path, camera_path, out_path, src_idx=2, tar_idx=1):
    points_2d = load_2d_points(points_2d_path)
    camera = load_camera_params(camera_path)
    name_list = points_2d.keys()

    name1 = list(name_list)[src_idx]
    params1 = points_2d[name1]["params"]
    points1 = points_2d[name1]["points"]

    name2 = list(name_list)[tar_idx]
    params2 = points_2d[name2]["params"]
    points2 = points_2d[name2]["points"]

    rot_1to2, t_1to2 = get_transform_matrix(params1, params2)

    points1_homogeneous = np.column_stack((points1, np.ones(len(points1))))
    points1_cam_coords = np.linalg.inv(camera).dot(points1_homogeneous.T).T
    # transformed_points = rot_1to2.dot(points1_cam_coords.T).T + t_1to2
    transformed_points = rot_1to2.dot(points1_cam_coords.T).T
    transformed_points = camera.dot(transformed_points.T).T / transformed_points[:, 2][:, np.newaxis]
    points = transformed_points[:, :2]

    points = [[item[0], item[1], i] for i, item in enumerate(points)]
    new_points = [item for item in points if item[0] > 0 and item[1] > 0 and item[0] < 1080 and item[1] < 1440]

    points1 = [[item[0], item[1], i] for i, item in enumerate(points1)]
    points2 = [[item[0], item[1], i] for i, item in enumerate(points2)]
    draw_matches(points1, points2, new_points, name1, name2, image_path, out_path)

    return points


# 需要看一下特征点是否对应
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, default="Seq_001")
    parser.add_argument("--cluster", type=str, default="0")
    parser.add_argument("--data_path", type=str, default="data/endomapper/sequence/")
    parser.add_argument("--output_path", type=str, default="output/check/")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    DATA_PATH = args.data_path + args.sequence + "/" + args.cluster
    POINTS_2D_PATH = DATA_PATH + "/sparse/0/images.txt"
    POINTS_3D_PATH = DATA_PATH + "/sparse/0/points3D.txt"
    IMAGE_PATH = DATA_PATH + "/images/"
    CAMERA_PATH = DATA_PATH + "/sparse/0/cameras.txt"
    OUT_PATH = args.output_path + args.sequence + "/" + args.cluster + '/'

    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    tar_idx = 1
    for src_idx in tqdm(range(max(0, tar_idx - 5), min(231, tar_idx + 5))):
        if src_idx == tar_idx:
            continue
        test_demo(POINTS_2D_PATH, IMAGE_PATH, CAMERA_PATH, OUT_PATH, src_idx, tar_idx)


