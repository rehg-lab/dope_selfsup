import torch
import time
import numpy as np
import os

from dope_selfsup.models.utils import (
        AverageMeter,
        count_parameters,
        compute_PCK,
        viz_local_heatmaps,
        viz_segmentations)

from dope_selfsup.nets.moco_func_utils import (
    moment_update,
    NCESoftmaxLoss,
    batch_shuffle_ddp,
    batch_unshuffle_ddp,
)

from dope_selfsup.inference.utils import compute_episode_accuracy_local_knn

from dope_selfsup.nets.dope_net import DOPEContrast, DOPENetworkCNN, DOPEProjector
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

BATCH_LOG_STEP = 10
MODEL_LOG_STEP = 25

class DOPE:
    def __init__(self, model_args, optim_args, data_args):
        self.build_net(model_args, data_args)
        self.build_optimizer(optim_args)

        self.batch_iter = 0
        self.loss = NCESoftmaxLoss()
        
    def build_net(self, model_args, data_args):
       
        # define model
        self.encoder = DOPENetworkCNN(
                model_args.fpn_dim)
        self.projector = DOPEProjector(
                model_args.fpn_dim,
                model_args.fpn_dim*8,
                model_args.proj_dim)
        self.encoder_ema = DOPENetworkCNN(
                model_args.fpn_dim)
        self.projector_ema = DOPEProjector(
                model_args.fpn_dim,
                model_args.fpn_dim*8,
                model_args.proj_dim)

        moment_update(self.encoder, self.encoder_ema, 0)
        moment_update(self.projector, self.projector_ema, 0)

        for name, p in self.encoder_ema.named_parameters():
            p.requires_grad = False

        for name, p in self.projector_ema.named_parameters():
            p.requires_grad = False

        print("Encoder")
        count_parameters(self.encoder)
        print("Projector")
        count_parameters(self.projector)
        print("Momentum Encoder")
        count_parameters(self.encoder_ema)
    
        if dist.is_initialized():
            local_rank = int(os.environ["LOCAL_RANK"])

            self.encoder = self.encoder.to(local_rank)
            self.encoder_ema = self.encoder_ema.to(local_rank)
            self.projector = self.projector.to(local_rank)
            self.projector_ema = self.projector_ema .to(local_rank)

            print("encoder ema check")
            print(local_rank, next(self.encoder_ema.parameters()).device)
            
            self.encoder = DDP(
                self.encoder, device_ids=[local_rank], output_device=local_rank
            )
            self.projector = DDP(
                self.projector, device_ids=[local_rank], output_device=local_rank
            )


        self.local_contrast = DOPEContrast(
            model_args.proj_dim, 
            T=model_args.contrast_T, 
            negative_source=model_args.negative_source,
            n_pos_pts=data_args.n_pts, 
            n_obj=data_args.batch_size
        )

        torch.cuda.empty_cache()

    def build_optimizer(self, optim_args):

        if self.projector is not None:
            params = [
                    {"params":self.encoder.parameters(), "lr":optim_args.learning_rate},
                    {"params":self.projector.parameters(), "lr":optim_args.learning_rate}
                ]
        else:
            params = [
                    {"params":self.encoder.parameters(), "lr":optim_args.learning_rate},
                ]

        if optim_args.optim_type == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                weight_decay=optim_args.weight_decay
            )

        if optim_args.scheduler_type == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=optim_args.epochs, 
                eta_min=0, 
                last_epoch=-1, 
                verbose=True)

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler


    def train_epoch(self, loader, writer, epoch):
        
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])

        ### defining meters
        loss_meter = AverageMeter()
        pos_l_meter = AverageMeter()  ## obj1 in im1 with obj1 in im2
        neg_l_2nd_view_meter  = AverageMeter() 
        neg_l_other_obj_meter = AverageMeter() 

        seg_loss_meter = AverageMeter()
        local_loss_meter = AverageMeter()

        model_time_meter = AverageMeter()
        batch_time_meter = AverageMeter()

        self.encoder.train()
        loader.sampler.set_epoch(epoch)

        epoch_start = time.time()
        
        load_start = time.time()
        for idx, batch in enumerate(loader):
            load_end = time.time()
            
            # don't train if we get a batch that's different than batch size
            # which usually happens at the end of an epoch
            if len(batch['image1']) != loader.batch_size:
                continue

            image1 = batch['image1'].cuda(non_blocking=True)
            image2 = batch['image2'].cuda(non_blocking=True)
            
            seg1 = batch['seg1'].cuda(non_blocking=True)
            seg2 = batch['seg2'].cuda(non_blocking=True)

            B, C, H, W = image1.shape

            uv_c1 = batch['uv_c1']
            uv_c2 = batch['uv_c2']

            # zero the parameter gradients
            self.optimizer.zero_grad()
            
            model_start = time.time()

            # Shuffle and reverse so that batch norm shortcut doesn't happen
            output_q = self.encoder(image1)
            
            with torch.no_grad():
                # shuffle for making use of BN
                image2, idx_unshuffle = batch_shuffle_ddp(image2)
                output_k = self.encoder_ema(image2)

                # undo shuffle
                output_k = {k:batch_unshuffle_ddp(v, idx_unshuffle) for k,v in output_k.items()}
                
            # get key and query features
            local_feat_grid_q = output_q['local_feat_pre_proj']
            local_feat_grid_k = output_k['local_feat_pre_proj']

            _, _, fH, fW = local_feat_grid_q.shape

            assert H == W
            
            # convert uv's to feature H/W from pixel H/W
            ratio = H/fH
            uv_c1 = torch.floor(uv_c1/ratio).long()
            uv_c2 = torch.floor(uv_c2/ratio).long()

            local_feat_q, local_feat_k = (
                self.local_contrast.extract_local_features(
                    local_feat_grid_q, 
                    local_feat_grid_k,
                    uv_c1,
                    uv_c2)
                )

            B, n_pts, C = local_feat_q.shape
            
            # apply projector
            local_feat_q = local_feat_q.view(B*n_pts, C, 1, 1)
            local_feat_k = local_feat_k.view(B*n_pts, C, 1, 1)
            
            local_feat_q = self.projector(local_feat_q).squeeze()

            with torch.no_grad():
                local_feat_k = self.projector_ema(local_feat_k).squeeze()

            mask_q = output_q['mask']
            mask_k = output_k['mask']

            # mask out projected features
            local_mask_q, local_mask_k = self.local_contrast.extract_local_features(
                mask_q.unsqueeze(1), mask_k.unsqueeze(1), uv_c1, uv_c2)

            local_mask_q = local_mask_q.view(B*n_pts, 1)
            local_mask_k = local_mask_k.view(B*n_pts, 1)
            
            local_feat_q = local_feat_q * local_mask_q
            local_feat_k = local_feat_k * local_mask_k

            out, sim_dct = self.local_contrast(
                    local_feat_q, 
                    local_feat_k)

            segmentation_loss = torch.nn.functional.binary_cross_entropy(
                    output_q['mask'], seg1
                )
            
            local_loss  = self.loss(out)
            
            ## updating the model
            loss_val = local_loss + segmentation_loss

            loss_val.backward()
            self.optimizer.step()

            moment_update(self.encoder, self.encoder_ema, 0.999)
            moment_update(self.projector, self.projector_ema, 0.999)
        
            model_end = time.time()

            if local_rank == 0:
                model_t = model_end - model_start
                if idx == 0: ## first batch is always slow and will throw off the averaging
                    batch_t = 0
                else:
                    batch_t = load_end - load_start

            load_start = time.time()
            ## progress bookkeeping
            loss_val = loss_val.cpu().item()
            local_loss = local_loss.cpu().item()
            segmentation_loss = segmentation_loss.cpu().item()
            
            if local_rank == 0: 

                writer.add_scalar("iter/train-loss", loss_val, self.batch_iter)
                writer.add_scalar("iter/train-segmentation-loss", segmentation_loss, self.batch_iter)
                writer.add_scalar("iter/train-local-loss", local_loss, self.batch_iter)
                
                writer.add_scalar("iter/train-pos_l", sim_dct["local_pos_l"], self.batch_iter)
                writer.add_scalar("iter/train-neg_l_2nd_view", sim_dct["2nd_view_neg_l"], self.batch_iter)
                writer.add_scalar("iter/train-neg_l_other_obj", sim_dct["other_obj_neg_l"], self.batch_iter)

                loss_meter.update(loss_val)
                local_loss_meter.update(local_loss, len(batch))
                seg_loss_meter.update(segmentation_loss, len(batch))

                pos_l_meter.update(sim_dct["local_pos_l"], len(batch))
                neg_l_2nd_view_meter.update(sim_dct["2nd_view_neg_l"], len(batch))
                neg_l_other_obj_meter.update(sim_dct["other_obj_neg_l"], len(batch))
                model_time_meter.update(model_t)
                batch_time_meter.update(batch_t)

                if idx % BATCH_LOG_STEP == 0:
                    print(
                        f"E:{epoch:05d}|"
                        f"B:{idx:03d}|{len(loader):03d} "
                        f"Data:{batch_time_meter.avg:.5f}s Model:{model_time_meter.avg:.3f}s "
                        f"Train Loss:{loss_meter.val:.3f}|{loss_meter.avg:.3f} "
                        f"Seg Loss:{seg_loss_meter.val:.3f}|{seg_loss_meter.avg:.3f} "
                        f"Local Loss:{local_loss_meter.val:.3f}|{local_loss_meter.avg:.3f} "
                        f"Pos L {pos_l_meter.val:3f}|{pos_l_meter.avg:3f} "
                        f"2nd view Neg L {neg_l_2nd_view_meter.val:3f}|{neg_l_2nd_view_meter.avg:3f} "
                        f"Other Obj Neg L {neg_l_other_obj_meter.val:3f}|{neg_l_other_obj_meter.avg:3f} "
                    )
            
            self.batch_iter += 1

        epoch_end = time.time()
        self.lr_scheduler.step()

        print("Time per epoch: {:.2f}".format((epoch_end-epoch_start)/60))
    
    @torch.no_grad()
    def eval(self, loader, writer, epoch, tag):
        pck_meter = AverageMeter()
        pixel_dist_meter = AverageMeter()
        seg_loss_meter = AverageMeter()

        pos_l_meter = AverageMeter()
        neg_l_2nd_view_meter  = AverageMeter() 
        neg_l_other_obj_meter = AverageMeter() 

        self.encoder.eval()

        for idx, batch in enumerate(loader):

            if len(batch['image1']) != loader.batch_size:
                continue
            
            image1 = batch['image1'].cuda()
            image2 = batch['image2'].cuda()
            B, _, H, W = image1.shape

            seg1 = batch['seg1'].cuda()

            uv_c1 = batch['uv_c1']
            uv_c2 = batch['uv_c2']
            
            segmentation_loss = torch.zeros(1).cuda()
            
            output_q = self.encoder.module(image1)
            output_k = self.encoder.module(image2)
            
            local_feat_grid_q = output_q['local_feat_pre_proj']
            local_feat_grid_k = output_k['local_feat_pre_proj']

            mask_q = output_q['mask']
            mask_k = output_k['mask']

            _, C, fH, fW = local_feat_grid_q.shape
            
            local_feat_grid_q = self.projector(local_feat_grid_q)
            local_feat_grid_k = self.projector(local_feat_grid_k)
            
            local_feat_grid_q = mask_q.unsqueeze(1) * local_feat_grid_q
            local_feat_grid_k = mask_k.unsqueeze(1) * local_feat_grid_k

            assert H == W

            ratio = H/fH
            uv_c1 = torch.floor(uv_c1/ratio).long()
            uv_c2 = torch.floor(uv_c2/ratio).long()

            pck_output = [
                compute_PCK(f1, f2, uv1.numpy(), uv2.numpy(),H) for 
                (f1, f2, uv1, uv2) in zip(local_feat_grid_q, local_feat_grid_k, uv_c1, uv_c2)]
            
            pck = [x[0] for x in pck_output]
            pck = np.mean(pck)
            
            pixel_dist = [x[1] for x in pck_output]
            pixel_dist = np.mean(pixel_dist)

            local_feat_q, local_feat_k = (
                self.local_contrast.extract_local_features(
                    local_feat_grid_q, 
                    local_feat_grid_k,
                    uv_c1,
                    uv_c2)
                )
            
            _, n_pts, C = local_feat_q.shape
            
            feature_similarity_matrix = torch.mm(
                local_feat_q.view(B*n_pts, C), local_feat_k.view(B*n_pts, C).T)

            l_pos, l_neg_2nd_view, l_neg_other_obj = self.local_contrast.get_pos_neg_l(
                    feature_similarity_matrix)

            segmentation_loss = torch.nn.functional.binary_cross_entropy(
                output_q['mask'], seg1
                    )

            segmentation_loss = segmentation_loss.cpu().item()

            pck_meter.update(pck, len(batch))
            pixel_dist_meter.update(pixel_dist, len(batch))
            seg_loss_meter.update(segmentation_loss, len(batch))

            pos_l_meter.update(l_pos.detach().mean().item(), len(batch))
            neg_l_2nd_view_meter.update(l_neg_2nd_view.detach().mean().item(), len(batch))
            neg_l_other_obj_meter.update(l_neg_other_obj.detach().mean().item(), len(batch))

            if idx % BATCH_LOG_STEP == 0:
                print(
                    f"E:{epoch:05d}|"
                    f"B: {idx:03d}|{len(loader):03d} "
                    f"Seg Loss:{seg_loss_meter.val:.3f}|{seg_loss_meter.avg:.3f} "
                    f"PCK {tag}:{pck_meter.val:.3f}|{pck_meter.avg:.3f} "
                    f"Pos L {tag}:{pos_l_meter.val:3f}|{pos_l_meter.avg:3f} "
                    f"2nd view Neg L {tag}:{neg_l_2nd_view_meter.val:3f}|{neg_l_2nd_view_meter.avg:3f} "
                    f"Other Obj Neg L {tag}:{neg_l_other_obj_meter.val:3f}|{neg_l_other_obj_meter.avg:3f} "
                )

        writer.add_scalar(f"epoch/{tag}-pck", pck_meter.avg, epoch)
        writer.add_scalar(f"epoch/{tag}-pixel-dist", pixel_dist_meter.avg, epoch)
        writer.add_scalar(f"epoch/{tag}-pos_l", pos_l_meter.avg, epoch)
        writer.add_scalar(f"epoch/{tag}-neg_l_2nd_view", neg_l_2nd_view_meter.avg, epoch)
        writer.add_scalar(f"epoch/{tag}-neg_l_other_obj", neg_l_other_obj_meter.avg, epoch)
        writer.add_scalar(f"epoch/{tag}-seg-loss", seg_loss_meter.avg, epoch)

        epoch_end = time.time()
        
        return pck_meter.avg
    
    @torch.no_grad()
    def test_viz(self, loader, epoch, output_dir, tag):
        self.encoder.eval()
        viz_dir = os.path.join(output_dir, "viz", "viz_e{:05d}".format(epoch))

        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        local_feat_list_1 = [None] * len(loader)
        local_feat_list_2 = [None] * len(loader)

        image_list_1 = [None] * len(loader)
        image_list_2 = [None] * len(loader)

        uv_c_list_1 = [None] * len(loader)
        uv_c_list_2 = [None] * len(loader)

        mask_gt_list = [None] * len(loader)
        mask_pred_list = [None] * len(loader)

        for idx, batch in enumerate(loader):

            image1 = batch["image1"].cuda()
            image2 = batch["image2"].cuda()

            seg1 = batch["seg1"]

            uv_c1 = batch["uv_c1"]
            uv_c2 = batch["uv_c2"]

            uv_c1 = uv_c1[:, [0], :]
            uv_c2 = uv_c2[:, [0], :]

            image_list_1[idx] = image1
            image_list_2[idx] = image2

            uv_c_list_1[idx] = uv_c1
            uv_c_list_2[idx] = uv_c2

            feats1 = self.encoder.module(image1)
            feats2 = self.encoder.module(image2)

            local_feat_grid_1 = feats1['local_feat_pre_proj']
            local_feat_grid_2 = feats2['local_feat_pre_proj']
            
            mask_1 = feats1['mask']
            mask_2 = feats2['mask']

            local_feat_grid_1 = self.projector(local_feat_grid_1)
            local_feat_grid_2 = self.projector(local_feat_grid_2)
            
            local_feat_grid_1 = mask_1.unsqueeze(0).unsqueeze(0) * local_feat_grid_1
            local_feat_grid_2 = mask_2.unsqueeze(0).unsqueeze(0) * local_feat_grid_2
            
            local_feat_list_1[idx] = local_feat_grid_1
            local_feat_list_2[idx] = local_feat_grid_2
            
            mask_gt_list[idx] = seg1
            mask_pred_list[idx] = feats1["mask"]

        idx = 0
        for (
            mask_gt,
            mask_pred,
            local_feats1,
            local_feats2,
            image1,
            image2,
            uv_c1,
            uv_c2,
        ) in zip(
            mask_gt_list,
            mask_pred_list,
            local_feat_list_1,
            local_feat_list_2,
            image_list_1,
            image_list_2,
            uv_c_list_1,
            uv_c_list_2,
        ):

            B, C, H, W = image1.shape

            image1 = image1.cpu().squeeze().numpy().transpose(1, 2, 0)
            image2 = image2.cpu().squeeze().numpy().transpose(1, 2, 0)

            # Positive local feature similarity visualization
            assert H == W

            im_uv_c1 = uv_c1.squeeze()
            im_uv_c2 = uv_c2.squeeze()

            viz_local_heatmaps(
                tag,
                viz_dir,
                image1,
                image2,
                im_uv_c1,
                im_uv_c2,
                local_feats1,
                local_feats2,
                epoch,
                idx,
            )

            viz_segmentations(tag, viz_dir, mask_gt, mask_pred, epoch, idx)

            idx += 1
    
    @torch.no_grad()
    def LS_eval_local(self, loader, writer, epoch):
        self.encoder.eval()

        n_shot = 1
        n_way = 5

        per_episode_accuracy = []

        with torch.no_grad():
            for idx, batch in enumerate(loader):

                images, labels = batch

                encoder_output = self.encoder.module(images.cuda())
                embeds = encoder_output["local_feat_pre_proj"]
                masks = encoder_output["mask"]

                embeds = self.projector(embeds)
                embeds = masks.unsqueeze(1) * embeds

                shot_labels = labels[: n_shot * n_way]
                query_labels = labels[n_shot * n_way :]

                shot_embeds = embeds[: n_shot * n_way]
                query_embeds = embeds[n_shot * n_way :]

                _, C, H, W = shot_embeds.shape

                shot_embeds = shot_embeds.reshape(n_way, n_shot, C, H, W)
                shot_labels = shot_labels.reshape(n_way, n_shot)

                accuracy = compute_episode_accuracy_local_knn(
                    shot_embeds, query_embeds, shot_labels, query_labels
                )

                accuracy = accuracy.item()
                per_episode_accuracy.append(accuracy)

                m = np.mean(per_episode_accuracy)
                p_str = (f"{idx:04d}/{len(loader):04d} - "
                         f"curr epi:{accuracy:.4f}  avg:{m:.4f}")
                print(p_str, end="\r")

        m = np.mean(per_episode_accuracy)

        print(f"E:{epoch:05d} Low Shot Validation {m:.3f}")

        writer.add_scalar("epoch/LS-val-acc", m, epoch)

        return m

    def save(self, p):
        save_dct = dict(
            encoder_dict=self.encoder.state_dict(),
            encoder_ema_dict=self.encoder_ema.state_dict(),
            projector_dict=self.projector.state_dict(),
            projector_ema_dict=self.projector_ema.state_dict(),
            optim_dict=self.optimizer.state_dict(),
            scheduler_dict=self.lr_scheduler.state_dict(),
            batch_iter=self.batch_iter,
        )

        torch.save(save_dct, p)

    def load(self, p, extract=False):
        dct = torch.load(p, map_location="cpu")

        encoder_dict = dct["encoder_dict"]
        encoder_ema_dict = dct["encoder_ema_dict"]

        projector_dict = dct["projector_dict"]
        projector_ema_dict= dct["projector_ema_dict"]

        optim_dict = dct["optim_dict"]
        scheduler_dict = dct["scheduler_dict"]

        if extract:
            encoder_dict = {k.replace("module.", ""):v for k, v in encoder_dict.items()}
            projector_dict = {k.replace("module.", ""):v for k, v in projector_dict.items()}

        self.encoder.load_state_dict(encoder_dict)
        self.encoder_ema.load_state_dict(encoder_ema_dict)
        self.projector.load_state_dict(projector_dict)
        self.projector_ema.load_state_dict(projector_ema_dict)
        self.optimizer.load_state_dict(optim_dict)
        self.lr_scheduler.load_state_dict(scheduler_dict)
        self.batch_iter = dct["batch_iter"]


    def train_full_ddp(self, loaders, writer, epoch_start, epoch_end, val_freq):

        train_loader = loaders["train_loader"]
        train_loader_eval = loaders["train_loader_eval"]
        train_loader_eval_viz = loaders["train_loader_eval_viz"]
        val_loader = loaders["val_loader"]
        val_loader_viz = loaders["val_loader_viz"]
        ls_val_loader = loaders["ls_val_loader"]

        local_rank = int(os.environ["LOCAL_RANK"])

        if local_rank == 0:
            ckpt_log_dir = os.path.join(writer.log_dir, "ckpts")

            if not os.path.exists(ckpt_log_dir):
                os.makedirs(ckpt_log_dir)

        best_metric_train = 1e-5
        best_metric_val = 1e-5
        best_ls_val_acc = 1e-5

        eval_metric_train = 1e-5
        eval_metric_val = 1e-5
        ls_val_acc = 1e-5

        for epoch in range(epoch_start, epoch_end):

            self.train_epoch(train_loader, writer, epoch)

            if local_rank == 0:

                if epoch % val_freq == 0:

                    ls_val_acc = self.LS_eval_local(ls_val_loader, writer, epoch)
                    eval_metric_train = self.eval(
                        train_loader_eval, writer, epoch, "eval_train"
                    )
                    eval_metric_val = self.eval(val_loader, writer, epoch, "eval_val")

                    print(
                        f"Epoch {epoch:05d}\t"
                        f"Current val seen {eval_metric_train:.3f}\t"
                        f"Current val unseen {eval_metric_val:.3f}\t"
                        f"Current LS val acc {ls_val_acc:.3f}"
                    )

                    self.test_viz(train_loader_eval_viz, epoch, writer.log_dir, "train")
                    self.test_viz(val_loader_viz, epoch, writer.log_dir, "val")

                    if eval_metric_train > best_metric_train:
                        best_metric_train = eval_metric_train
                        p = os.path.join(writer.log_dir, "best_eval_train_ckpt.pt")
                        self.save(p)

                    if eval_metric_val > best_metric_val:
                        best_metric_val = eval_metric_val
                        p = os.path.join(writer.log_dir, "best_eval_val_ckpt.pt")
                        self.save(p)

                    if ls_val_acc > best_ls_val_acc:
                        best_ls_val_acc = ls_val_acc
                        p = os.path.join(writer.log_dir, "best_ls_val_ckpt.pt")
                        self.save(p)

                if epoch % MODEL_LOG_STEP == 0:
                    self.save(os.path.join(ckpt_log_dir, "{:04d}.pt".format(epoch)))

                p = os.path.join(writer.log_dir, "last.pt")
                self.save(p)

                print(
                    f"Epoch {epoch:05d}\t"
                    f"Best val seen {best_metric_train:.3f}\t"
                    f"Best val unseen {best_metric_val:.3f}\t"
                    f"Best LS val acc {best_ls_val_acc:.3f}"
                )

            print("{} PRE BARRIER".format(local_rank))
            dist.barrier()
            print("{} POST BARRIER".format(local_rank))
