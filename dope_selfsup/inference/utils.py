import os
import torch
import torchvision
import numpy as np
import random

from dope_selfsup.data.data_utils import fps
from tqdm import tqdm

def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm

def compute_episode_accuracy_local_knn(shot_embeds, query_embeds, shot_labels, query_labels, k=20, mode="max"):

    predicted_labels = []

    shot_labels = shot_labels[:,0]

    for query in query_embeds:
        rows, cols = torch.where(query.norm(dim=0)>=0.90)
        
        # use randomly sampled features if we don't have enough features with norm bigger than 0.9
        if len(rows) < k:

            rows = torch.randint(56, size=(k,))
            cols = torch.randint(56, size=(k,))

        uvs = torch.stack([rows, cols],dim=1).cpu().numpy()
        query_idxs = fps(uvs, k)
        uvs = torch.from_numpy(uvs[query_idxs])

        query_features = query[:,uvs[:,0],uvs[:,1]]

        # how similar are these query features, to the
        # features in each of the shots for each class?

        shot_scores = []
        for shots in shot_embeds:

            n_shots, C, H, W = shots.shape
            # n_shots x C x H x W -> n_shots x C x H*W
            shots = shots.reshape(n_shots, C, H*W)
            # K x H*W x n_shots

            # ((N_shots x H*W x C) @ (128 x K)).T = K x H*W x N_shots
            similarity = ((shots.transpose(1,2) @ query_features).T)

            # take max over H*W dimension
            scores = similarity.max(dim=1)[0]
            per_shot_scores = scores.sum(axis=0)
            shot_scores.append(per_shot_scores.cpu())

        shot_scores = torch.stack(shot_scores)

        if mode == "sum":
            per_label_scores = shot_scores.sum(axis=1)
        if mode == "max":
            per_label_scores = shot_scores.max(axis=1)[0]

        pred_label = shot_labels[torch.argmax(per_label_scores)]
        predicted_labels.append(pred_label.item())

    acc = (torch.tensor(predicted_labels)==query_labels).float().mean()

    return acc

@torch.no_grad()
def evaluate(encoder, projector, loader, n_neighbors):
    encoder.eval()
    projector.eval()

    n_shot = loader.batch_sampler.n_shot
    n_way = loader.batch_sampler.n_way
    n_query = loader.batch_sampler.n_query

    per_episode_accuracy = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images, labels = batch
            images = images.cuda()

            encoder_output = encoder(images.cuda())
            embeds = encoder_output["local_feat_pre_proj"]
            masks = encoder_output["mask"]

            embeds = projector(embeds)
            embeds = masks.unsqueeze(1) * embeds

            shot_labels = labels[:n_shot*n_way]
            query_labels = labels[n_shot*n_way:]

            shot_embeds = embeds[:n_shot*n_way]
            query_embeds = embeds[n_shot*n_way:]

            _, C, fH, fW = shot_embeds.shape

            shot_embeds = shot_embeds.reshape(n_way, n_shot, C, fH, fW)
            shot_labels = shot_labels.reshape(n_way, n_shot)

            accuracy = compute_episode_accuracy_local_knn(
                    shot_embeds, query_embeds, shot_labels, query_labels, k=n_neighbors
                )

            accuracy = accuracy.item()
            per_episode_accuracy.append(accuracy)

            m, pm = compute_confidence_interval(per_episode_accuracy)
            p_str = f"{idx:04d}/{len(loader):04d} - curr epi:{accuracy:.4f}  avg:{m:.4f} ci:{pm:.4f}"
            print(p_str, end='\r')

    m, pm = compute_confidence_interval(per_episode_accuracy)

    return m, pm


