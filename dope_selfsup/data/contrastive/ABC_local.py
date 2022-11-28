import numpy as np
import torch
import os
import json
import itertools
import cv2

from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset
from PIL import Image
from pprint import pprint

from dope_selfsup.data import data_utils

DATA_ROOT = "./dataset_directory/abc"


class ABC(Dataset):
    def __init__(
        self,
        split=None,
        n_pts=None,
        augmentation_file=None,
        mask_size=None,
    ):

        split_path = os.path.join(DATA_ROOT, "split", f"{split}.txt")

        with open(split_path, "r") as f:
            objects = f.readlines()
            objects = [x.strip() for x in objects]

        self.objects = objects
        self.split = split
        self.n_pts = n_pts
        self.transform = None

        if augmentation_file is not None and augmentation_file != "none":
            with open(f"./dope_selfsup/data/{augmentation_file}", "r") as f:
                aug_params = json.load(f)
            print("Augmentation parameters:")
            pprint(aug_params)
            self.transform = data_utils.ContrastiveAugmentation(aug_params)

        self.mask_size = mask_size

        print("Total {} samples: {}".format(split, len(self.objects)))

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx):
        obj = self.objects[idx]

        obj_path = os.path.join(DATA_ROOT, obj)
        corr_path = os.path.join(DATA_ROOT, obj, "corr.npy")
        corr = np.load(corr_path)
        
        # find good pairs out of the 20 views per object
        # the data is generated so that (0,1), (2,3), (4,5) etc are pairs of views
        # so we find the ones with at least self.n_pts correspondences

        good_pair_idxs = np.where((corr != -1)[:, 0, :, 0].sum(-1) > self.n_pts)[0]
        p_idx = np.random.choice(good_pair_idxs)
        pair = np.arange(20).reshape(10, 2)[p_idx]
        
        # corresponding pixels in each view
        uv_c1, uv_c2 = corr[p_idx].astype(int)
        
        # farthest point sampling
        uv_idx = data_utils.fps(uv_c1, self.n_pts)

        uv_c1 = uv_c1[uv_idx]
        uv_c2 = uv_c2[uv_idx]

        img1_path = os.path.join(DATA_ROOT, obj, "RGB", "{:04d}.jpeg".format(pair[0]))
        img2_path = os.path.join(DATA_ROOT, obj, "RGB", "{:04d}.jpeg".format(pair[1]))

        obj_inst = obj.replace(".glb", ".obj")
        seg1_path = os.path.join(
            DATA_ROOT,
            obj,
            "segmentations",
            obj_inst,
            "{}_{:04d}.jpeg".format(obj_inst, pair[0]),
        )
        seg2_path = os.path.join(
            DATA_ROOT,
            obj,
            "segmentations",
            obj_inst,
            "{}_{:04d}.jpeg".format(obj_inst, pair[1]),
        )

        uv_c1 = torch.tensor(uv_c1)
        uv_c2 = torch.tensor(uv_c2)

        ## loading image pair
        with open(img1_path, "rb") as f:
            img1 = Image.open(f).convert("RGB")

        with open(img2_path, "rb") as f:
            img2 = Image.open(f).convert("RGB")

        ## loading image pair
        seg1 = data_utils.read_segmentation(seg1_path)
        seg2 = data_utils.read_segmentation(seg2_path)

        seg1 = np.stack([seg1, seg1, seg1], axis=-1)
        seg2 = np.stack([seg2, seg2, seg2], axis=-1)

        seg1 = Image.fromarray(seg1)
        seg2 = Image.fromarray(seg2)
        
        # apply augmentation
        if self.transform is not None:
            choice_list = [[True, False], [False, True], [True, True], [True, True]]
            bool1, bool2 = choice_list[np.random.randint(4)]

            img1, seg1, uv_c1 = self.transform(img1, seg1, uv_c1, bool1)
            img2, seg2, uv_c2 = self.transform(img2, seg2, uv_c2, bool2)

        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        seg1 = TF.to_tensor(seg1)
        seg2 = TF.to_tensor(seg2)

        seg1 /= seg1.max()
        seg2 /= seg2.max()

        seg1 = seg1[0, :, :]  # make single channel
        seg2 = seg2[0, :, :]  # make single channel

        # resize to feature dimension
        # have to use cv2 for cv2.INTER_AREA which gives 
        # us better binary segmentations when the grid is coarse

        seg1 = cv2.resize(
            seg1.numpy(), (self.mask_size, self.mask_size), interpolation=cv2.INTER_AREA
        )
        seg2 = cv2.resize(
            seg2.numpy(), (self.mask_size, self.mask_size), interpolation=cv2.INTER_AREA
        )

        seg1[seg1 > 0] = 1
        seg2[seg2 > 0] = 1

        seg1 = torch.from_numpy(seg1)
        seg2 = torch.from_numpy(seg2)

        # randomly give either view 1 or view 2 to encoder/momentum encoder
        if np.random.choice([0, 1]):
            out_dct = dict(
                image1=img1,
                image2=img2,
                seg1=seg1,
                seg2=seg2,
                uv_c1=uv_c1,
                uv_c2=uv_c2,
                instance=obj,
            )
        else:
            out_dct = dict(
                image1=img2,
                image2=img1,
                seg1=seg2,
                seg2=seg1,
                uv_c1=uv_c2,
                uv_c2=uv_c1,
                instance=obj,
            )

        return out_dct


if __name__ == "__main__":
    SEED = 1234
    import random
    import matplotlib.pyplot as plt
    import matplotlib

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    seen_train = ABC(
        split="train",
        n_pts=32,
        augmentation_file="augmentation_settings.json",
        mask_size=56,
    )

    ## viz dataloading
    out_dir = "dataset_test_outputs/ABC_local_dataset"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for i in range(80):
        print(i)
        data = seen_train.__getitem__(i)
        image1, image2 = data["image1"], data["image2"]
        uv1_ls, uv2_ls = data["uv_c1"], data["uv_c2"]
        seg1, seg2 = data["seg1"], data["seg2"]
        name = data["instance"]

        image1 = image1.numpy().transpose(1, 2, 0)
        image2 = image2.numpy().transpose(1, 2, 0)

        fig = matplotlib.figure.Figure()
        (ax1, ax2), (ax3, ax4) = fig.subplots(2, 2)
        ax1.axis("off")
        ax2.axis("off")

        ax1.imshow(image1)
        ax2.imshow(image2)
        ax3.imshow(seg1, cmap="gray")
        ax4.imshow(seg2, cmap="gray")
        fig.suptitle(name)

        for uv1, uv2 in zip(uv1_ls, uv2_ls):
            ax1.plot(uv1[1], uv1[0], ".", markersize=2.5)
            ax2.plot(uv2[1], uv2[0], ".", markersize=2.5)

        fig.suptitle(name)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "{:04d}.png".format(i)))
