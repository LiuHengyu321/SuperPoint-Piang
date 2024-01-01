import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_points(transfer_res, image_path, output_path):
    image1 = cv2.imread(image_path + transfer_res['name1'])
    image2 = cv2.imread(image_path + transfer_res['name2'])

    plt.figure(figsize=(36, 9))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.title('Image1 and its original points')
    plt.axis('off')
    for point in transfer_res["points1"]:
        plt.scatter(point[0], point[1], color='red', s=1)

    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Image2 and its new points")
    plt.axis('off')
    for point in transfer_res["new_points"]:
        plt.scatter(point[0], point[1], color='green', s=1)

    plt.subplot(1, 4, 3)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Image2 and its original points")
    plt.axis('off')
    for point in transfer_res["points2"]:
        plt.scatter(point[0], point[1], color='red', s=1)

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.title("Image2 and its all points")
    plt.axis('off')
    for point in transfer_res["new_points"]:
        plt.scatter(point[0], point[1], color='green', s=1)
    for point in transfer_res["points2"]:
        plt.scatter(point[0], point[1], color='red', s=1)

    plt.savefig(output_path + "1.jpg")
    return


def draw_key_points(img, corners, color=(0, 255, 0), radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple((s * c[:2]).astype(int)), radius, color, thickness=-1)
    return img


def draw_single_image(image, points, save_path):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for point in points:
        plt.scatter(point[0], point[1], color='red', s=1)
    plt.axis("off")
    plt.savefig(save_path + "1.jpg")
    return


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img

