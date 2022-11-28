import torch
import numpy as np
import random
import os
import sys
import time
import yaml
import shutil
import torch.distributed as dist

from argparse import ArgumentParser, Namespace
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from dope_selfsup import models

from dope_selfsup.data.lowshot.sampler import CategoriesSampler
from dope_selfsup.data.lowshot.modelnet import ModelNet as ModelNet_lowshot

from dope_selfsup.data.contrastive.ABC_local import ABC as ABC_local
from dope_selfsup.data.contrastive.modelnet_local import ModelNet as modelnet_local

torch.backends.cudnn.benchmark = True

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def get_datasets_local(data_args):

    if data_args.dataset == "modelnet":

        train_dataset = modelnet_local(
            split="train",
            augmentation_file=data_args.aug_file,
            n_pts=data_args.n_pts,
            mask_size=data_args.mask_size,
        )

        train_dataset_eval = modelnet_local(
            split="train",
            augmentation_file="none",
            n_pts=data_args.n_pts,
            mask_size=data_args.mask_size,
        )

        val_dataset = modelnet_local(
            split="val", 
            n_pts=data_args.n_pts, 
            mask_size=data_args.mask_size,
        )

        ls_val_dataset = ModelNet_lowshot("val")

        datasets = {
            "train_dataset": train_dataset,
            "train_dataset_eval": train_dataset_eval,
            "val_dataset": val_dataset,
            "ls_val_dataset": ls_val_dataset,
        }

    if data_args.dataset == "ABC":

        train_dataset = ABC_local(
            split="train",
            augmentation_file=data_args.aug_file,
            n_pts=data_args.n_pts,
            mask_size=data_args.mask_size,
        )

        train_dataset_eval = ABC_local(
            split="train",
            augmentation_file="none",
            n_pts=data_args.n_pts,
            mask_size=data_args.mask_size,
        )

        val_dataset = ABC_local(
            split="val",
            n_pts=data_args.n_pts,
            mask_size=data_args.mask_size,
        )

        ls_val_dataset = ModelNet_lowshot("val")

        datasets = {
            "train_dataset": train_dataset,
            "train_dataset_eval": train_dataset_eval,
            "val_dataset": val_dataset,
            "ls_val_dataset": ls_val_dataset,
        }

    return datasets


def get_dataloaders(datasets, data_args):

    train_dataset = datasets["train_dataset"]
    train_dataset_eval = datasets["train_dataset_eval"]
    val_dataset = datasets["val_dataset"]
    ls_val_dataset = datasets["ls_val_dataset"]

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True
    )

    # use len(val_dataset) worth of training samples for
    # evaluating on the training data
    train_sampler_eval = torch.utils.data.SubsetRandomSampler(
        torch.randperm(len(train_dataset))[: len(val_dataset)]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        prefetch_factor=data_args.prefetch_factor,
    )

    train_loader_eval = torch.utils.data.DataLoader(
        train_dataset_eval,
        num_workers=data_args.num_workers,
        batch_size=data_args.batch_size,
        pin_memory=True,
        sampler=train_sampler_eval,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=data_args.batch_size,
        num_workers=data_args.num_workers,
        pin_memory=True,
    )

    train_sampler_eval_viz = torch.utils.data.SubsetRandomSampler(
        torch.randperm(len(train_dataset))[:20]
    )

    train_loader_eval_viz = torch.utils.data.DataLoader(
        train_dataset_eval,
        batch_size=1,
        sampler=train_sampler_eval_viz,
        num_workers=8,
        pin_memory=True,
    )

    val_sampler_viz = torch.utils.data.SubsetRandomSampler(
        torch.randperm(len(val_dataset))[:20]
    )

    val_loader_viz = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler_viz,
        num_workers=8,
        pin_memory=True,
    )

    # low-shot loader
    ls_sampler_params = [
        ls_val_dataset.labels,
        data_args.n_ls_val_iters,  # n iters
        5,  # way
        1,  # shot
        15,  # query
        "multi_instance_shots",
    ]

    ls_sampler = CategoriesSampler(*ls_sampler_params)

    ls_val_loader = torch.utils.data.DataLoader(
        ls_val_dataset, num_workers=6, batch_sampler=ls_sampler, pin_memory=True
    )

    loaders = {
        "train_loader": train_loader,
        "train_loader_eval": train_loader_eval,
        "train_loader_eval_viz": train_loader_eval_viz,
        "val_loader": val_loader,
        "val_loader_viz": val_loader_viz,
        "ls_val_loader": ls_val_loader,
    }

    return loaders


def main(cfg, ckpt, cfg_p):

    ## DDP stuff
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    cfg["data_args"]["batch_size"] = cfg["data_args"]["batch_size"] // world_size

    model_args = Namespace(**cfg["model_args"])
    data_args = Namespace(**cfg["data_args"])
    optim_args = Namespace(**cfg["optim_args"])

    log_dir = os.path.join(
        cfg["exp_meta"]["log_dir"],
        cfg["exp_meta"]["exp_name"],
        datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
    )

    torch.manual_seed(cfg["exp_meta"]["seed"])
    np.random.seed(cfg["exp_meta"]["seed"])
    random.seed(cfg["exp_meta"]["seed"])

    dist.init_process_group(
        backend="nccl", rank=local_rank, world_size=int(os.environ["WORLD_SIZE"])
    )
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed_all(cfg["exp_meta"]["seed"])

    print(f"Rank {local_rank + 1}/{world_size} process initialized.\n")

    model = getattr(models, model_args.model_type)(model_args, optim_args, data_args)

    if ckpt != "":
        model.load(ckpt)
        log_dir = os.path.split(ckpt)[0]
        log_dir = os.path.join(log_dir + "_continued")

    start_epoch = model.lr_scheduler.state_dict()["last_epoch"]

    if local_rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        # book keeping
        shutil.copy(cfg_p, log_dir)

        # book keeping
        Tee(os.path.join(log_dir, "log.txt"), "a")
    else:
        writer = None

    datasets = get_datasets_local(data_args)
    loaders = get_dataloaders(datasets, data_args)

    model.train_full_ddp(
        loaders, writer, start_epoch, optim_args.epochs, cfg["exp_meta"]["val_freq"]
    )


def add_arguments(parser):
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--local_rank", type=int, default=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    main(cfg, args.ckpt, args.cfg)
