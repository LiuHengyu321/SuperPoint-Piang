import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.d2s import DepthToSpace, SpaceToDepth


def img_overlap(img_r, img_g, img_gray):  # img_b repeat
    def to_3d(img):
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        return img
    img_r, img_g, img_gray = to_3d(img_r), to_3d(img_g), to_3d(img_gray)
    img = np.concatenate((img_gray, img_gray, img_gray), axis=0)
    img[0, :, :] += img_r[0, :, :]
    img[1, :, :] += img_g[0, :, :]
    img[img > 1] = 1
    img[img < 0] = 0
    return img


def load_2d_points(path_2d):
    with open(path_2d, 'r') as f:
        lines = f.readlines()
    lines = lines[4:]
    images = {}
    name = None

    for idx, line in enumerate(lines):
        line = line.replace("\n", "").split(" ")
        if idx % 2 == 0:
            name = line[-1]
            params = [float(i) for i in line[1: -2]]
            item = {
                "params": params,
                "points": []
            }
            images[name] = item
        else:
            points = []
            line = [float(i) for i in line]
            for i in range(0, len(line), 3):
                if line[i + 2] != -1:
                    line[i + 2] = int(line[i + 2])
                    points.append(line[i: i + 2])
            images[name]["points"] = points

    sorted_index = sorted(images)
    sorted_images = {k: images[k] for k in sorted_index}
    return sorted_images


def load_camera_params(path_camera):
    with open(path_camera, 'r') as f:
        lines = f.readlines()
    line = lines[-1].replace("\n", "").split(" ")
    params = [float(i) for i in line[4: 8]]
    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    k = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])
    return k


# transfer from 1 to 2
def get_transform_matrix(params1, params2):
    pose1 = np.array(params1)
    pose2 = np.array(params2)

    q1, t1 = pose1[: 4], pose1[4:]
    q2, t2 = pose2[: 4], pose2[4:]

    rot1 = R.from_quat(q1).as_matrix()
    rot2 = R.from_quat(q2).as_matrix()

    rot = rot2 @ rot1.T
    t = t2 - rot @ t1
    return rot, t


def transfer_points(points1, camera, rot_1to2, translation_1to2):
    points1_homogeneous = np.column_stack((points1, np.ones(len(points1))))
    points1_cam_coords = np.linalg.inv(camera).dot(points1_homogeneous.T).T

    transformed_points = rot_1to2.dot(points1_cam_coords.T).T
    # transformed_points = rot_1to2.dot(points1_cam_coords.T).T + translation_1to2
    transformed_points = camera.dot(transformed_points.T).T / transformed_points[:, 2][:, np.newaxis]

    points = transformed_points[:, :2]
    points = [item for item in points if item[0] > 0 and item[1] > 0]
    return points


def match_pairs(points, camera, idx1=0, idx2=1):
    name_list = points.keys()

    name1 = list(name_list)[idx1]
    params1 = points[name1]['params']
    points1 = points[name1]['points']

    name2 = list(name_list)[idx2]
    params2 = points[name2]['params']
    points2 = points[name2]['points']

    rot_1to2, t_1to2 = get_transform_matrix(params1, params2)
    points_1to2 = transfer_points(points1, camera, rot_1to2, t_1to2)
    transfer_res = {
        "name1": name1,
        "name2": name2,
        "points1": points1,
        "points2": points2,
        "new_points": points_1to2
    }
    return transfer_res


def save_image(img, filename):
    cv2.imwrite(filename, img)


def pltImshow(img):
    plt.imshow(img)
    plt.show()


# 给定一系列点和homo转移矩阵，输出转移之后点的坐标
def warp_points(points, homographies, device='cpu'):
    """
    Warp a list of points with the given homography.

    Arguments:
        points: list of N points, shape (N, 2(x, y))).
        homography: batched or not (shapes (B, 3, 3) and (...) respectively).

    Returns: a Tensor of shape (N, 2) or (B, N, 2(x, y)) (depending on whether the homography
            is batched) containing the new coordinates of the warped points.

    """
    # expand points len to (x, y, 1)
    no_batches = len(homographies.shape) == 2
    homographies = homographies.unsqueeze(0) if no_batches else homographies
    batch_size = homographies.shape[0]

    points = torch.cat((points.float(), torch.ones((points.shape[0], 1)).to(device)), dim=1)
    points = points.to(device)

    homographies = homographies.view(batch_size * 3, 3)

    warped_points = homographies@points.transpose(0, 1)
    warped_points = warped_points.view([batch_size, 3, -1])
    warped_points = warped_points.transpose(2, 1)

    warped_points = warped_points[:, :, :2] / warped_points[:, :, 2:]
    return warped_points[0, :, :] if no_batches else warped_points


def compute_valid_mask(image_shape, inv_homography, device='cpu', erosion_radius=0):
    """
    Compute a boolean mask of the valid pixels resulting from a homography applied to
    an image of a given shape. Pixels that are False correspond to bordering artifacts.
    A margin can be discarded using erosion.

    Arguments:
        input_shape: Tensor of rank 2 representing the image shape, i.e. `[H, W]`.
        homography: Tensor of shape (B, 8) or (8,), where B is the batch size.
        `erosion_radius: radius of the margin to be discarded.

    Returns: a Tensor of type `tf.int32` and shape (H, W).
    """
    if inv_homography.dim() == 2:
        inv_homography = inv_homography.view(-1, 3, 3)
    batch_size = inv_homography.shape[0]
    mask = torch.ones(batch_size, 1, image_shape[0], image_shape[1]).to(device)

    mask = inv_warp_image_batch(mask, inv_homography, device=device, mode='nearest')
    mask = mask.view(batch_size, image_shape[0], image_shape[1])
    mask = mask.cpu().numpy()
    if erosion_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_radius*2,)*2)
        for i in range(batch_size):
            mask[i, :, :] = cv2.erode(mask[i, :, :], kernel, iterations=1)

    return torch.tensor(mask).to(device)


def normPts(pts, shape):
    """
    normalize pts to [-1, 1]
    :param pts:
        tensor (y, x)
    :param shape:
        tensor shape (y, x)
    :return:
    """
    pts = pts/shape*2 - 1
    return pts


def inv_warp_image_batch(img, mat_homo_inv, device='cpu', mode='bilinear'):
    """
        inverse warp images in batch
        :param img:
            batch of images
            tensor [batch_size, 1, H, W]
        :param mat_homo_inv:
            batch of homography matrices
            tensor [batch_size, 3, 3]
        :param device:
            GPU device or CPU
        :return:
            batch of warped images
            tensor [batch_size, 1, H, W]
    """
    # compute inverse warped points
    if len(img.shape) == 2 or len(img.shape) == 3:
        img = img.view(1, 1, img.shape[0], img.shape[1])
    if len(mat_homo_inv.shape) == 2:
        mat_homo_inv = mat_homo_inv.view(1, 3, 3)

    Batch, channel, H, W = img.shape
    coor_cells = torch.stack(torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)), dim=2)
    coor_cells = coor_cells.transpose(0, 1)
    coor_cells = coor_cells.to(device)
    coor_cells = coor_cells.contiguous()

    src_pixel_coords = warp_points(coor_cells.view([-1, 2]), mat_homo_inv, device)
    src_pixel_coords = src_pixel_coords.view([Batch, H, W, 2])
    src_pixel_coords = src_pixel_coords.float()

    # 将原来图片中旧坐标的值填充到新坐标中
    warped_img = F.grid_sample(img, src_pixel_coords, mode=mode, align_corners=True)
    return warped_img


def inv_warp_image(img, mat_homo_inv, device='cpu', mode='bilinear'):
    """
    Inverse warp images in batch

    :param img:
        batch of images
        tensor [H, W]
    :param mat_homo_inv:
        batch of homography matrices
        tensor [3, 3]
    :param device:
        GPU device or CPU
    :return:
        batch of warped images
        tensor [H, W]
    """
    warped_img = inv_warp_image_batch(img, mat_homo_inv, device, mode)
    return warped_img.squeeze()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def labels_2d_to_3d(labels, cell_size, add_dustbin=True):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)
    if add_dustbin:
        dustbin = labels.sum(dim=1)
        dustbin = 1 - dustbin
        dustbin[dustbin < 1.] = 0
        # print('dust: ', dustbin.shape)
        # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
        ## norm
        dn = labels.sum(dim=1)
        labels = labels.div(torch.unsqueeze(dn, 1))
    return labels


def flatten_detection(semi, tensor=False):
    """
    Flatten detection output

    :param semi:
        output from detector head
        tensor [65, Hc, Wc]
        :or
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        np (1, H, C)
        :or
        tensor (batch_size, 65, Hc, Wc)

    """
    batch = False
    if len(semi.shape) == 4:
        batch = True
        batch_size = semi.shape[0]

    if batch:
        dense = nn.functional.softmax(semi, dim=1)  # [batch, 65, Hc, Wc]
        # Remove dustbin.
        nodust = dense[:, :-1, :, :]
    else:
        dense = nn.functional.softmax(semi, dim=0)  # [65, Hc, Wc]
        nodust = dense[:-1, :, :].unsqueeze(0)

    depth2space = DepthToSpace(8)
    heatmap = depth2space(nodust)
    heatmap = heatmap.squeeze(0) if not batch else heatmap
    return heatmap

def labels2Dto3D_flattened(labels, cell_size):
    '''
    Change the shape of labels into 3D. Batch of labels.

    :param labels:
        tensor [batch_size, 1, H, W]
        keypoint map.
    :param cell_size:
        8
    :return:
         labels: tensors[batch_size, 65, Hc, Wc]
    '''
    batch_size, channel, H, W = labels.shape
    Hc, Wc = H // cell_size, W // cell_size
    space2depth = SpaceToDepth(8)
    # labels = space2depth(labels).squeeze(0)
    labels = space2depth(labels)
    # print("labels in 2Dto3D: ", labels.shape)
    # labels = labels.view(batch_size, H, 1, W, 1)
    # labels = labels.view(batch_size, Hc, cell_size, Wc, cell_size)
    # labels = labels.transpose(1, 2).transpose(3, 4).transpose(2, 3)
    # labels = labels.reshape(batch_size, 1, cell_size ** 2, Hc, Wc)
    # labels = labels.view(batch_size, cell_size ** 2, Hc, Wc)

    dustbin = torch.ones((batch_size, 1, Hc, Wc)).cuda()
    # labels = torch.cat((labels, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.cat((labels*2, dustbin.view(batch_size, 1, Hc, Wc)), dim=1)
    labels = torch.argmax(labels, dim=1)
    return labels


def load_checkpoint(load_path, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    filename = '{}__{}'.format(file_prefix[0], filename)
    checkpoint = torch.load(load_path/filename)
    print("load checkpoint from ", filename)
    return checkpoint


def save_checkpoint(save_path, net_state, epoch, filename='checkpoint.pth.tar'):
    file_prefix = ['superPointNet']
    # torch.save(net_state, save_path)
    filename = '{}_{}_{}'.format(file_prefix[0], str(epoch), filename)
    torch.save(net_state, save_path/filename)
    print("save checkpoint to ", filename)


def get_writer_path(task='train', exper_name='', date=True):
    import datetime
    prefix = 'runs/'
    str_date_time = ''
    if exper_name != '':
        exper_name += '_'
    if date:
        str_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    return prefix + task + '/' + exper_name + str_date_time


def precision_recall_torch(pred, labels):
    offset = 10**-6
    assert pred.size() == labels.size(), 'Sizes of pred, labels should match when you get the precision/recall!'
    precision = torch.sum(pred*labels) / (torch.sum(pred)+ offset)
    recall = torch.sum(pred*labels) / (torch.sum(labels) + offset)
    if precision.item() > 1.:
        print(pred)
        print(labels)
        import scipy.io.savemat as savemat
        savemat('pre_recall.mat', {'pred': pred, 'labels': labels})
    assert precision.item() <=1. and precision.item() >= 0.
    return {'precision': precision, 'recall': recall}


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


def box_nms(prob, size, iou=0.1, min_prob=0.01, keep_top_k=0):
    # requires https://github.com/open-mmlab/mmdetection.
    # Warning : BUILD FROM SOURCE using command MMCV_WITH_OPS=1 pip install -e
    # from mmcv.ops import nms as nms_mmdet
    from torchvision.ops import nms

    """Performs non maximum suppression on the heatmap by considering hypothetical
    bounding boxes centered at each pixel's location (e.g. corresponding to the receptive
    field). Optionally only keeps the top k detections.
    Arguments:
    prob: the probability heatmap, with shape `[H, W]`.
    size: a scalar, the size of the bouding boxes.
    iou: a scalar, the IoU overlap threshold.
    min_prob: a threshold under which all probabilities are discarded before NMS.
    keep_top_k: an integer, the number of top scores to keep.
    """
    pts = torch.nonzero(prob > min_prob).float() # [N, 2]
    prob_nms = torch.zeros_like(prob)
    if pts.nelement() == 0:
        return prob_nms
    size = torch.tensor(size/2.).cuda()
    boxes = torch.cat([pts-size, pts+size], dim=1) # [N, 4]
    scores = prob[pts[:, 0].long(), pts[:, 1].long()]
    if keep_top_k != 0:
        indices = nms(boxes, scores, iou)
    else:
        raise NotImplementedError

    pts = torch.index_select(pts, 0, indices)
    scores = torch.index_select(scores, 0, indices)
    prob_nms[pts[:, 0].long(), pts[:, 1].long()] = scores
    return prob_nms


def get_pts_from_heatmap(heatmap, conf_thresh, nms_dist):
    '''
    :param self:
    :param heatmap:
        np (H, W)
    :return:
    '''

    border_remove = 4

    H, W = heatmap.shape[0], heatmap.shape[1]
    xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
    sparsemap = (heatmap >= conf_thresh)
    if len(xs) == 0:
        return np.zeros((3, 0))
    pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
    pts[0, :] = ys
    pts[1, :] = xs
    pts[2, :] = heatmap[xs, ys]
    pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
    inds = np.argsort(pts[2, :])
    pts = pts[:, inds[::-1]]  # Sort by confidence.
    # Remove points along border.
    bord = border_remove
    toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
    toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
    toremove = np.logical_or(toremoveW, toremoveH)
    pts = pts[:, ~toremove]
    return pts


def homography_scaling_torch(homography, H, W):
    trans = torch.tensor([[2./W, 0., -1], [0., 2./H, -1], [0., 0., 1.]])
    homography = (trans.inverse() @ homography @ trans)
    return homography


def filter_points(points, shape, return_mask=False):
    ### check!
    points = points.float()
    shape = shape.float()
    mask = (points >= 0) * (points <= shape-1)
    mask = (torch.prod(mask, dim=-1) == 1)
    if return_mask:
        return points[mask], mask
    return points[mask]
    # return points [torch.prod(mask, dim=-1) == 1]


def crop_or_pad_choice(in_num_points, out_num_points, shuffle=False):
    # Adapted from https://github.com/haosulab/frustum_pointnet/blob/635c938f18b9ec1de2de717491fb217df84d2d93/fpointnet/data/datasets/utils.py
    """Crop or pad point cloud to a fixed number; return the indexes
    Args:
        points (np.ndarray): point cloud. (n, d)
        num_points (int): the number of output points
        shuffle (bool): whether to shuffle the order
    Returns:
        np.ndarray: output point cloud
        np.ndarray: index to choose input points
    """
    if shuffle:
        choice = np.random.permutation(in_num_points)
    else:
        choice = np.arange(in_num_points)
    assert out_num_points > 0, 'out_num_points = %d must be positive int!'%out_num_points
    if in_num_points >= out_num_points:
        choice = choice[:out_num_points]
    else:
        num_pad = out_num_points - in_num_points
        pad = np.random.choice(choice, num_pad, replace=True)
        choice = np.concatenate([choice, pad])
    return choice

