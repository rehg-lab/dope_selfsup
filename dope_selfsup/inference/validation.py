import os
import torch
import torchvision
import numpy as np
import random
import torch.nn.functional as F

from torch import nn
from argparse import ArgumentParser, Namespace

import dope_selfsup.nets

from dope_selfsup.nets import dope_net
from dope_selfsup.data.lowshot.modelnet import ModelNet
from dope_selfsup.data.lowshot.sampler import CategoriesSampler
from dope_selfsup.inference.utils import evaluate

def main(
    encoder_path,
    out_p,
    dataset,
    split_dir,
    n_neighbors,
    n_shots_list,
    n_ways_list,
    n_queries,
    n_iters,
    shot_type,
):

    enc_name = encoder_path.split("/")[-1].replace(".pt", "")
    print(enc_name)
    encoder = dope_net.DOPENetworkCNN(128)
    projector = dope_net.DOPEProjector(128, 1024, 256) # hard coded but matches config

    ckpt_dict = torch.load(encoder_path, map_location="cpu")

    encoder_dict = ckpt_dict["encoder_dict"]
    encoder_dict = {k.replace("module.", ""): v for k, v in encoder_dict.items()}

    projector_dict = ckpt_dict["projector_dict"]
    projector_dict = {k.replace("module.", ""): v for k, v in projector_dict.items()}

    encoder.load_state_dict(encoder_dict)
    projector.load_state_dict(projector_dict)

    encoder = encoder.cuda()
    projector = projector.cuda()

    if dataset == "modelnet":
        test_dataset = ModelNet("val", split_dir)

    for n_ways in n_ways_list:
        for n_shots in n_shots_list:

            torch.manual_seed(1234)
            np.random.seed(1234)
            random.seed(1234)

            test_sampler_params = [
                test_dataset.labels,
                n_iters,
                n_ways,
                n_shots,
                n_queries,
                shot_type,
            ]

            test_sampler = CategoriesSampler(*test_sampler_params)

            test_loader = torch.utils.data.DataLoader(
                test_dataset, num_workers=6, batch_sampler=test_sampler, pin_memory=True
            )

            mean_acc, conf_interval = evaluate(
                encoder, projector, test_loader, n_neighbors
            )

            out_str = (
                f"{n_ways:03d}w{n_shots:03d}s{n_queries:03d}q   "
                f"acc:{mean_acc:.4f}({conf_interval:.4f})\n"
            )

            out_file = os.path.join(
                out_p, f"{n_ways:03d}w{n_shots:03d}s{n_queries:03d}q_{enc_name}.txt"
            )

            with open(out_file, "w") as f:
                f.write(out_str)

            print(out_str)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--enc_p", type=str, default="")
    parser.add_argument("--dataset", type=str, default="modelnet")
    parser.add_argument("--shot_type", type=str, default="multi_instance_shots")
    parser.add_argument("--n_neighbors", type=int, default=20)
    parser.add_argument("--n_iters", type=int, default=2500)
    parser.add_argument("--n_queries", type=int, default=15)
    parser.add_argument("--n_ways", type=str, default="5,10")
    parser.add_argument("--n_shots", type=str, default="1,5")
    parser.add_argument("--split_dir", type=str, default="")
    args = parser.parse_args()

    encoders = sorted([x for x in os.listdir(args.enc_p) if x.endswith(".pt")])[::-1]

    for encoder in encoders:
        encoder_path = os.path.join(args.enc_p, encoder)

        dest = os.path.join(
            args.enc_p,
            "eval_output",
            f"eval_{encoder}_{args.dataset}_validation_{args.split_dir}_{args.shot_type}_{args.n_iters}_iterations",
        )
        if not os.path.exists(dest):
            os.makedirs(dest)
        n_shots = [int(x) for x in args.n_shots.split(",")]
        n_ways = [int(x) for x in args.n_ways.split(",")]

        main(
            encoder_path,
            dest,
            args.dataset,
            args.split_dir,
            args.n_neighbors,
            n_shots,
            n_ways,
            args.n_queries,
            args.n_iters,
            args.shot_type,
        )
