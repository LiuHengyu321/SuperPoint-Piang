import argparse
import yaml
import os
import logging
from pathlib import Path

import numpy as np
from imageio import imread
from tqdm import tqdm

import torch
import torch.optim
import torch.utils.data

from utils.draw import draw_key_points
from utils.utils import save_image, inv_warp_image_batch
from utils.loader import data_loader_test as data_loader

from models.model_wrap import SuperPointFrontend_torch, PointTracker


def combine_heatmap(heatmap, inv_homographies, mask_2D, device="cpu"):
    # multiply heatmap with mask_2D
    heatmap = heatmap * mask_2D

    heatmap = inv_warp_image_batch(
        heatmap, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )

    # check
    mask_2D = inv_warp_image_batch(
        mask_2D, inv_homographies[0, :, :, :], device=device, mode="bilinear"
    )
    heatmap = torch.sum(heatmap, dim=0)
    mask_2D = torch.sum(mask_2D, dim=0)
    return heatmap / mask_2D
    pass


# input 1 images, output pseudo ground truth by homography adaptation.
# labels: ‘prob’(key points): np (N1, 3)
@torch.no_grad()
def export_detector_homo_adapt_gpu(config, output_dir, args):

    # basic setting
    task = config["data"]["dataset"]
    export_task = config["data"]["export_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logging.info("train on device: %s", device)

    with open(os.path.join(output_dir, "config.yml"), "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # parameters
    nms_dist = config["model"]["nms"]  # 4
    top_k = config["model"]["top_k"]
    homoAdapt_iter = config["data"]["homography_adaptation"]["num"]
    conf_thresh = config["model"]["detection_threshold"]
    nn_thresh = 0.7
    count = 0
    max_length = 5
    output_images = args.outputImg
    check_exist = True

    # save data
    save_path = Path(output_dir)
    save_output = save_path
    save_output = save_output / "predictions" / export_task
    save_path = save_path / "checkpoints"
    logging.info("=> will save everything to {}".format(save_path))
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_output, exist_ok=True)

    # data loading
    data = data_loader(config, dataset=task, export_task=export_task)
    test_set, test_loader = data["test_set"], data["test_loader"]

    # model loading
    # load pretrained
    try:
        path = config["pretrained"]
        print("==> Loading pre-trained network.")
        print("path: ", path)

        # This class runs the SuperPoint network and processes its outputs.
        fe = SuperPointFrontend_torch(
            config=config,
            weights_path=path,
            nms_dist=nms_dist,
            conf_thresh=conf_thresh,
            nn_thresh=nn_thresh,
            cuda=False,
            device=device,
        )
        print("==> Successfully loaded pre-trained network.")
        fe.net_parallel()
        print(path)

        # save to files
        save_file = save_output / "export.txt"
        with open(save_file, "a") as myfile:
            myfile.write("load model: " + path + "\n")

    except Exception:
        print(f"load model: {path} failed! ")
        raise

    def load_as_float(path):
        return imread(path).astype(np.float32) / 255

    tracker = PointTracker(max_length, nn_thresh=fe.nn_thresh)
    with open(save_file, "a") as myfile:
        myfile.write("homography adaptation: " + str(homoAdapt_iter) + "\n")

    # loop through all images
    for i, sample in tqdm(enumerate(test_loader)):
        img, mask_2D = sample["image"], sample["valid_mask"]
        img = img.transpose(0, 1)
        img_2D = sample["image_2D"].numpy().squeeze()
        mask_2D = mask_2D.transpose(0, 1)

        inv_homographies, homographies = (
            sample["homographies"],
            sample["inv_homographies"],
        )
        img, mask_2D, homographies, inv_homographies = (
            img.to(device),
            mask_2D.to(device),
            homographies.to(device),
            inv_homographies.to(device),
        )

        name = sample["name"][0]
        logging.info(f"name: {name}")
        if check_exist:
            p = Path(save_output, "{}.npz".format(name))
            if p.exists():
                logging.info("file %s exists. skip the sample.", name)
                continue

        # pass through network
        heatmap = fe.run(img, onlyHeatmap=True, train=False)
        outputs = combine_heatmap(heatmap, inv_homographies, mask_2D, device=device)
        pts = fe.getPtsFromHeatmap(outputs.detach().cpu().squeeze())  # (x,y, prob)

        # subpixel prediction
        if config["model"]["subpixel"]["enable"]:
            fe.heatmap = outputs  # tensor [batch, 1, H, W]
            print("outputs: ", outputs.shape)
            print("pts: ", pts.shape)
            pts = fe.soft_argmax_points([pts])
            pts = pts[0]

        # top K points
        pts = pts.transpose()
        print("total points: ", pts.shape)
        print("pts: ", pts[:5])
        if top_k:
            if pts.shape[0] > top_k:
                pts = pts[:top_k, :]
                print("topK filter: ", pts.shape)

        # save keypoints
        pred = {}
        pred.update({"pts": pts})

        # make directories
        filename = str(name)
        if task == "Kitti" or "Kitti_inh":
            scene_name = sample["scene_name"][0]
            os.makedirs(Path(save_output, scene_name), exist_ok=True)

        path = Path(save_output, "{}.npz".format(filename))
        np.savez_compressed(path, **pred)

        # output images for visualization labels
        if output_images and i < 10:
            img_pts = draw_key_points(img_2D * 255, pts.transpose())
            f = save_output / (str(count) + ".png")
            if task == "Coco" or "Kitti":
                f = save_output / (name + ".png")
            save_image(img_pts, str(f))
        count += 1

    print("output pseudo ground truth: ", count)
    save_file = save_output / "export.txt"
    with open(save_file, "a") as myfile:
        myfile.write("Homography adaptation: " + str(homoAdapt_iter) + "\n")
        myfile.write("output pairs: " + str(count) + "\n")
    pass


if __name__ == "__main__":
    torch.set_default_tensor_type(torch.FloatTensor)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # using homography adaptation to export detection psuedo ground truth
    p_train = argparse.ArgumentParser()
    p_train.add_argument("--command", type=str, default="export_detector_homoAdapt")
    p_train.add_argument("--config", type=str, default="configs/magicpoint_simcol_export.yaml")
    p_train.add_argument("--exper_name", type=str, default="simcol_homo_prediction_256")
    p_train.add_argument("--eval", action="store_true")
    p_train.add_argument("--outputImg", action="store_true", default=True, help="output image for visualization")
    p_train.add_argument("--debug", action="store_true", default=False, help="turn on debuging mode")
    p_train.add_argument("--output_path", type=str, default="/data/hyliu/simcol_out/")
    p_train.set_defaults(func=export_detector_homo_adapt_gpu)

    args = p_train.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = args.output_path + args.exper_name
    os.makedirs(output_dir, exist_ok=True)

    logging.info("Running command {}".format(args.command.upper()))
    args.func(config, output_dir, args)
