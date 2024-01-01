
# losses
import torch
import numpy as np
from utils.utils import to_numpy


def print_var(points):
    print("points: ", points.shape)
    print("points: ", points)
    pass


# from utils.losses import pts_to_bbox
def pts_to_bbox(points, patch_size):
    """
    input:
        points: (y, x)
    output:
        bbox: (x1, y1, x2, y2)
    """

    shift_l = (patch_size + 1) / 2
    shift_r = patch_size - shift_l
    pts_l = points - shift_l
    pts_r = points + shift_r + 1
    bbox = torch.stack((pts_l[:, 1], pts_l[:, 0], pts_r[:, 1], pts_r[:, 0]), dim=1)
    return bbox
    pass


# torchvision roi pooling
def _roi_pool(pred_heatmap, rois, patch_size=8):
    from torchvision.ops import roi_pool
    patches = roi_pool(pred_heatmap, rois.float(), (patch_size, patch_size), spatial_scale=1.0)
    return patches
    pass


# from utils.losses import norm_patches
def norm_patches(patches):
    patch_size = patches.shape[-1]
    patches = patches.view(-1, 1, patch_size * patch_size)
    d = torch.sum(patches, dim=-1).unsqueeze(-1) + 1e-6
    patches = patches / d
    patches = patches.view(-1, 1, patch_size, patch_size)
    # print("patches: ", patches.shape)
    return patches


def extract_patches(label_idx, image, patch_size=7):
    """
    return:
        patches: tensor [N, 1, patch, patch]
    """
    rois = pts_to_bbox(label_idx[:,2:], patch_size).long()
    # filter out??
    rois = torch.cat((label_idx[:,:1], rois), dim=1)
    # print_var(rois)
    # print_var(image)
    patches = _roi_pool(image, rois, patch_size=patch_size)
    return patches




# from utils.losses import extract_patch_from_points
def extract_patch_from_points(heatmap, points, patch_size=5):
    """
    this function works in numpy
    """

    # numpy
    if type(heatmap) is torch.Tensor:
        heatmap = to_numpy(heatmap)
    heatmap = heatmap.squeeze()  # [H, W]
    # padding
    pad_size = int(patch_size / 2)
    heatmap = np.pad(heatmap, pad_size, 'constant')
    # crop it
    patches = []
    ext = lambda img, pnt, wid: img[pnt[1]:pnt[1] + wid, pnt[0]:pnt[0] + wid]
    print("heatmap: ", heatmap.shape)
    for i in range(points.shape[0]):
        # print("point: ", points[i,:])
        patch = ext(heatmap, points[i, :].astype(int), patch_size)
        # print("patch: ", patch.shape)
        patches.append(patch)
    # extract points
    return patches



def soft_argmax_2d(patches, normalized_coordinates=True):
    """
    params:
        patches: (B, N, H, W)
    return:
        coor: (B, N, 2)  (x, y)

    """
    import torchgeometry as tgm
    m = tgm.contrib.SpatialSoftArgmax2d(normalized_coordinates=normalized_coordinates)
    coords = m(patches)  # 1x4x2
    return coords


# log on patches
# from utils.losses import do_log
def do_log(patches):
    patches[patches < 0] = 1e-6
    patches_log = torch.log(patches)
    return patches_log

