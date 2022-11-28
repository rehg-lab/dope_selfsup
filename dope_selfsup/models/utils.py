import os
import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
import gc
import cv2
import torch.nn.functional as F
import matplotlib
import math

from prettytable import PrettyTable
from dope_selfsup.data.data_utils import fps

class AverageMeter(object):
    """Computes and stores the average and current value.

    Examples::
        >>> # Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # Update meter after every minibatch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def viz_local_heatmaps(tag, viz_dir, image1, image2, uv_c1, uv_c2, f1, f2, epoch, idx):

    f1 = f1.squeeze()
    f2 = f2.squeeze()

    H, W, _ = image1.shape
    C, fH, fW = f1.shape
    assert fH == fW
    feat_dim = fH

    ratio = H / fH
    f1 = f1.reshape(C, fH * fW).T
    f2 = f2.reshape(C, fH * fW).T

    uv_c1 = torch.floor(uv_c1 / ratio).long().squeeze().numpy()
    uv_c2 = uv_c2.long().squeeze().numpy()

    dist = torch.mm(f1, f2.T)

    max_idx = torch.argmax(dist, dim=1)

    row1, col1 = uv_c1
    row2_gt, col2_gt = uv_c2

    i = int(row1 * fH + col1)
    row2, col2 = np.divmod(max_idx[i].item(), feat_dim)
    euc_dist = dist[i, :].reshape(fH, fW).cpu().numpy()
    distance_im = np.clip(euc_dist, 0.0, 1)

    row1 = np.floor(row1 * ratio).astype(int)
    col1 = np.floor(col1 * ratio).astype(int)

    row2 = np.floor(row2 * ratio).astype(int)
    col2 = np.floor(col2 * ratio).astype(int)

    img = np.zeros((H, W * 3, 3))
    im1 = (image1 * 255).astype(np.uint8)[:, :, [2, 1, 0]]
    im2 = (image2 * 255).astype(np.uint8)[:, :, [2, 1, 0]]
    im_tgt = (image2 * 255).astype(np.uint8)[:, :, [2, 1, 0]]

    im = copy.deepcopy(im1)
    img_out_1 = np.zeros(im.shape)
    img_out_1[:, :, :] = im[:, :, :]
    cv2.circle(img_out_1, (col1+int(ratio//2), row1+int(ratio//2)), 1, (0, 255, 255), thickness=2)
    cv2.rectangle(img_out_1, (col1, row1), (col1+int(ratio), row1+int(ratio)), (0, 255, 255), thickness=1)
    
    #leftmost image
    im = copy.deepcopy(im2)
    img_out_2 = np.zeros(im.shape)
    img_out_2[:, :, :] = im[:, :, :]
    cv2.circle(img_out_2, (col2_gt+int(ratio/2), row2_gt+int(ratio/2)), 1, (0, 255, 0), thickness=2)
    cv2.rectangle(img_out_2, (col2_gt, row2_gt), (col2_gt+int(ratio), row2_gt+int(ratio)), (0, 255, 0), thickness=1)

    #middle image
    im2 = (distance_im * 255).astype(np.uint8)
    im2 = cv2.resize(im2, (H, W), interpolation=cv2.INTER_NEAREST)
    im2 = np.stack([im2, im2, im2], axis=2)
    heatmap_img = cv2.applyColorMap(im2, cv2.COLORMAP_INFERNO)
    img_out_3 = cv2.addWeighted(heatmap_img, 0.7, im_tgt, 0.3, 0)
    cv2.circle(img_out_3, (col2+int(ratio/2), row2+int(ratio/2)), 1, (255, 0, 0), thickness=2)

    cv2.rectangle(img_out_3, (col2, row2), (col2+int(ratio), row2+int(ratio)), (255, 0, 0), thickness=1)

    #last image
    cv2.circle(img_out_3, (col2_gt+int(ratio/2), row2_gt+int(ratio/2)), 1, (0, 255, 0), thickness=2)
    cv2.rectangle(img_out_3, (col2_gt, row2_gt), (col2_gt+int(ratio), row2_gt+int(ratio)), (0, 255, 0), thickness=1)
    
    # writing
    img[:, :W, :] = img_out_1
    img[:, W : 2 * W, :] = img_out_2
    img[:, 2 * W :, :] = img_out_3

    p = os.path.join(viz_dir, "local_{}_e{:05d}_im{:04d}.jpeg".format(tag, epoch, idx))

    cv2.imwrite(p, img)

def viz_segmentations(tag, viz_dir, mask_gt, mask_pred, epoch, idx):
    mask_gt = (mask_gt.squeeze().numpy() * 255).astype(int)
    mask_pred = (mask_pred.cpu().numpy() * 255).astype(int)

    H, W = mask_gt.shape

    out_im = np.zeros((H, W * 2))
    out_im[:, :W] = mask_gt
    out_im[:, W:] = mask_pred

    out_im = cv2.resize(out_im, (W * 2 * 4, H * 4), interpolation=cv2.INTER_NEAREST)
    p = os.path.join(viz_dir, "mask_{}_e{:05d}_im{:04d}.jpeg".format(tag, epoch, idx))

    cv2.imwrite(p, out_im)


def compute_PCK(f1, f2, uv_c1, uv_c2, H):
    C, fH, fW = f1.shape
    assert fH == fW
    feat_dim = fH
    ratio = H / fH

    im2_coords = []
    for row1, col1 in uv_c1:

        f = f1[:,row1,col1]
        sim = torch.mul(f.view(C,1,1),f2).sum(0)
        row2, col2 = torch.where(sim==sim.max())
        row2=row2[0].item()
        col2=col2[0].item()

        im2_coords.append([row2, col2])

    im2_coords = np.array(im2_coords)

    im2_coords = np.floor(im2_coords * ratio).astype(int)

    im2_coords_target = uv_c2 * ratio
    mean_l2_dist = np.linalg.norm(im2_coords - im2_coords_target, axis=1)
    pck = PCK(im2_coords, im2_coords_target, H)

    return pck, mean_l2_dist

def PCK(co1, co2, H):
    diff = np.abs(co1 - co2)
    thres = np.round(0.05 * H).astype(int)

    boolean = diff < thres
    boolean = np.logical_and(boolean[:, 0], boolean[:, 1])

    return np.mean(boolean)

