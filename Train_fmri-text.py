#!/usr/bin/env python
# coding: utf-8

# Import packages & functions

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom models and functions #
from src import utils
from src.models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, \
    VersatileDiffusionPriorNetwork

# Multi-GPU config #
from accelerate import Accelerator

accelerator = Accelerator(split_batches=False, mixed_precision='fp16')
print("PID of this process =", os.getpid())
print = accelerator.print  # only print if local_rank=0
device = accelerator.device
print("device:", device)
num_devices = torch.cuda.device_count()
if num_devices == 0: num_devices = 1
num_workers = num_devices
print(accelerator.state)
local_rank = accelerator.state.local_process_index
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
print("distributed =", distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =",
      world_size)

# # Configurations

parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="train_proj_coco_text",
    help="name of model, used for ckpt saving and wandb logging (if enabled)",
)
parser.add_argument(
    "--data_path", type=str, default="/mnt/sda/songzengyu/code/brain_imager/other_subject/subj07/data",
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--subj", type=int, default=7, choices=[1, 2, 5, 7],
)
parser.add_argument(
    "--batch_size", type=int, default=128,
    help="Batch size can be increased by 10x if only training v2c and not diffusion prior",
)
parser.add_argument(
    "--hidden", action=argparse.BooleanOptionalAction, default=True,
    help="if True, CLIP embeddings will come from last hidden layer (e.g., 257x768 - Versatile Diffusion), rather than final layer",
)
parser.add_argument(
    "--clip_variant", type=str, default="ViT-L/14", choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
    help='OpenAI clip variant',
)
parser.add_argument(
    "--wandb_log", action=argparse.BooleanOptionalAction, default=False,
    help="whether to log to wandb",
)
parser.add_argument(
    "--resume_from_ckpt", action=argparse.BooleanOptionalAction, default=False,
    help="if not using wandb and want to resume from a ckpt",
)
parser.add_argument(
    "--wandb_project", type=str, default="stability",
    help="wandb project name",
)
parser.add_argument(
    "--mixup_pct", type=float, default=.33,
    help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
)
parser.add_argument(
    "--norm_embs", action=argparse.BooleanOptionalAction, default=True,
    help="Do l2-norming of CLIP embeddings",
)
parser.add_argument(
    "--use_image_aug", action=argparse.BooleanOptionalAction, default=False,
    help="whether to use image augmentation",
)
parser.add_argument(
    "--num_epochs", type=int, default=240,
    help="number of epochs of training",
)
parser.add_argument(
    "--prior", action=argparse.BooleanOptionalAction, default=True,
    help="if False, will only use CLIP loss and ignore diffusion prior",
)
parser.add_argument(
    "--v2c", action=argparse.BooleanOptionalAction, default=False,
    help="if False, will only use diffusion prior loss",
)
parser.add_argument(
    "--plot_umap", action=argparse.BooleanOptionalAction, default=False,
    help="Plot UMAP plots alongside reconstructions",
)
parser.add_argument(
    "--lr_scheduler_type", type=str, default='cycle', choices=['cycle', 'linear'],
)
parser.add_argument(
    "--ckpt_saving", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--ckpt_interval", type=int, default=30,
    help="save backup ckpt and reconstruct every x epochs",
)
parser.add_argument(
    "--save_at_end", action=argparse.BooleanOptionalAction, default=False,
    help="if True, saves best.ckpt at end of training. if False and ckpt_saving==True, will save best.ckpt whenever epoch shows best validation score",
)
parser.add_argument(
    "--seed", type=int, default=42,
)
parser.add_argument(
    "--max_lr", type=float, default=3e-4,
)
parser.add_argument(
    "--n_samples_save", type=int, default=0, choices=[0, 1],
    help="Number of reconstructions for monitoring progress, 0 will speed up training",
)
parser.add_argument(
    "--use_projector", action=argparse.BooleanOptionalAction, default=True,
    help="Additional MLP after the main MLP so model can separately learn a way to minimize NCE from prior loss (BYOL)",
)
parser.add_argument(
    "--vd_cache_dir", type=str,
    default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)
parser.add_argument(
    "--coco_feature_path", type=str,
    default="/mnt/sda/songzengyu/code/brain_imager/other_subject/subj07/expand_text/feature/expand_coco_annots_features_beam_5_77x768_subj07_all.npy",
)
args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(args.seed, cudnn_deterministic=False)

# change learning rate based on number of devices
args.max_lr *= accelerator.num_processes

# change batch size based on number of devices if using multi-gpu
# args.batch_size *= accelerator.num_processes

# change num_epochs based on number of devices if using multi-gpu
args.num_epochs *= accelerator.num_processes

# In[5]:


outdir = os.path.abspath(f'../other_subject/subj07/trained_model/fmri-text')

# # Prep models and data loaders
# In[6]:
print('Pulling NSD webdataset data...')
train_url = "{" + f"{args.data_path}/train/train_subj0{args.subj}_" + "{0..17}.tar," + f"{args.data_path}/val/val_subj0{args.subj}_0.tar" + "}"
val_url = f"{args.data_path}/test/test_subj0{args.subj}_" + "{0..1}.tar"
print(train_url, "\n", val_url)
meta_url = f"{args.data_path}/metadata_subj0{args.subj}.json"
num_train = (8559 + 300)*3
num_val = 982

print('Prepping train and validation dataloaders...')
train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    args.batch_size, 'images',
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=300,
    cache_dir=args.data_path,  # "/tmp/wds-cache",
    seed=args.seed,
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images", "coco"],
    local_rank=local_rank,
    world_size=world_size,
)

# In[7]:
print('Creating Clipper...')
clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
clip_size = clip_sizes[args.clip_variant]
if args.hidden:
    clip_extractor = Clipper(args.clip_variant, device=device, hidden_state=True, norm_embs=args.norm_embs)
    out_dim = 77 * clip_size
print("out_dim:", out_dim)

print('Creating voxel2clip...')
if args.subj == 1:
    num_voxels = 15724
elif args.subj == 2:
    num_voxels = 14278
elif args.subj == 3:
    num_voxels = 15226
elif args.subj == 4:
    num_voxels = 13153
elif args.subj == 5:
    num_voxels = 13039
elif args.subj == 6:
    num_voxels = 17907
elif args.subj == 7:
    num_voxels = 12682
elif args.subj == 8:
    num_voxels = 14386
voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, clip_size=clip_size, use_projector=args.use_projector)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)

# setup prior network
out_dim = clip_size
depth = 6
dim_head = 64
heads = clip_size // 64  # heads * dim_head = 12 * 64 = 768
if args.hidden:
    guidance_scale = 3.5
    timesteps = 100
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=77,
        learned_query_mode="pos_emb"
    ).to(device)
    print("prior_network loaded")

    # custom version that can fix seeds
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    ).to(device)

# optimizer
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in diffusion_prior.net.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 1e-2},
    {'params': [p for n, p in diffusion_prior.net.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0},
    {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 1e-2},
    {'params': [p for n, p in diffusion_prior.voxel2clip.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)

global_batch_size = args.batch_size * num_devices

# lr_scheduler
if args.lr_scheduler_type == 'cycle':
    total_steps = int(args.num_epochs * (num_train // global_batch_size))
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2 / args.num_epochs
    )

def save_ckpt(tag):
    ckpt_path = outdir + f'/{tag}.pth'
    print(f'saving {ckpt_path}', flush=True)
    unwrapped_model = accelerator.unwrap_model(diffusion_prior)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'val_losses': val_losses,
            'lrs': lrs,
        }, ckpt_path)
    except:
        print("Couldn't save... moving on to prevent crashing.")
    del unwrapped_model
print("Done with model preparations!")

# # Main

# In[9]:
epoch = 0
losses, val_losses, lrs = [], [], []
best_val_loss = 1e9
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, args.num_epochs - int(args.mixup_pct * args.num_epochs))
if args.hidden:
    prior_mult = 0.3

torch.cuda.empty_cache()

# In[10]:
diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
    diffusion_prior, optimizer, train_dl, val_dl, lr_scheduler
)

# In[11]:
coco_features = np.load(args.coco_feature_path)
print(f"{args.model_name} starting with epoch {epoch} / {args.num_epochs}")
progress_bar = tqdm(range(epoch, args.num_epochs), ncols=1200, disable=(local_rank != 0))


for epoch in progress_bar:
    diffusion_prior.train()
    loss_prior_sum = 0.
    val_loss_prior_sum = 0.
    loss_nce_sum = 0.
    val_loss_nce_sum = 0.

    for train_i, (voxel, image, coco) in enumerate(train_dl):
        with torch.cuda.amp.autocast():
            optimizer.zero_grad()

            repeat_index = train_i % 3
            voxel = voxel[:, repeat_index].float()

            if epoch < int(args.mixup_pct * args.num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            # clip_target = clip_extractor.embed_image(image).float()
            # self add
            clip_target = coco_features[coco.squeeze().cpu()]
            clip_target = torch.tensor(clip_target).float().to('cuda:0')
            clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)

            if args.hidden:
                clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)

            if args.prior:
                loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)

            clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
            clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if epoch < int(args.mixup_pct * args.num_epochs):
                loss_nce = utils.mixco_nce(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=.006,
                    perm=perm, betas=betas, select=select)
            else:
                epoch_temp = soft_loss_temps[epoch-int(args.mixup_pct * args.num_epochs)]
                loss_nce = utils.soft_clip_loss(
                    clip_voxels_norm,
                    clip_target_norm,
                    temp=epoch_temp)

            if args.prior:
                loss_nce_sum += loss_nce.item()
                loss_prior_sum += loss_prior.item()
                loss = loss_nce + (prior_mult * loss_prior)

            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()
            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if args.lr_scheduler_type is not None:
                lr_scheduler.step()

    diffusion_prior.eval()
    for val_i, (voxel, image, coco) in enumerate(val_dl):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # repeat_index = val_i % 3

                # voxel = voxel[:,repeat_index].float()
                voxel = torch.mean(voxel, axis=1).float()

                clip_target = coco_features[coco.squeeze().cpu()]
                clip_target = torch.tensor(clip_target).float().to('cuda:0')
                clip_voxels, clip_voxels_proj = diffusion_prior.module.voxel2clip(voxel) if distributed else diffusion_prior.voxel2clip(voxel)

                if args.hidden:
                    clip_voxels = clip_voxels.view(len(voxel), -1, clip_size)

                if args.prior:
                    val_loss_prior, aligned_clip_voxels = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target)

                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

                if epoch < int(args.mixup_pct * args.num_epochs):
                    val_loss_nce = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=None, betas=None, select=None)
                else:
                    val_loss_nce = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                if args.prior:
                    val_loss_nce_sum += val_loss_nce.item()
                    val_loss_prior_sum += val_loss_prior.item()
                    val_loss = val_loss_nce + (prior_mult * val_loss_prior)
                utils.check_loss(val_loss)
                val_losses.append(val_loss.item())


    if local_rank == 0:
        if (not args.save_at_end and args.ckpt_saving) or (args.save_at_end and epoch == args.num_epochs - 1):
            # save best model
            val_loss = np.mean(val_losses[-(val_i + 1):])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt(f'best')

        logs = {"train/loss": np.mean(losses[-(train_i + 1):]),
                "val/loss": np.mean(val_losses[-(val_i + 1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "val/num_steps": len(val_losses),
                "train/loss_nce": loss_nce_sum / (train_i + 1),
                "train/loss_prior": loss_prior_sum / (train_i + 1),
                "val/loss_nce": val_loss_nce_sum / (val_i + 1),
                "val/loss_prior": val_loss_prior_sum / (val_i + 1)}

        progress_bar.set_postfix(**logs)

        # Save model checkpoint and reconstruct
        if epoch == args.num_epochs - 1:
            save_ckpt(f'last')
            print(f'best_val_loss: {best_val_loss:.3f}')
    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()

print("\n===Finished!===\n")
if not utils.is_interactive():
    sys.exit(0)

