# # Import packages & functions

import os
import shutil
import sys
import json
import traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import OrderedDict
import kornia
from kornia.augmentation.container import AugmentationSequential

import utils
from models import Voxel2StableDiffusionModel
from convnext import ConvnextXL

local_rank = 0
device = torch.device("cuda:1")
num_devices = 1
num_workers = 1
from diffusers.models import AutoencoderKL

autoenc = AutoencoderKL(
    down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
    up_block_types=['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
    block_out_channels=[128, 256, 512, 512],
    layers_per_block=2,
    sample_size=256
)
autoenc.load_state_dict(torch.load('/mnt/sda/songzengyu/code/brain_imager/data/sd_image_var_autoenc.pth'))
autoenc.requires_grad_(False)
autoenc.eval()

# # Configurations
voxel_dims = 1
batch_size = 16
num_epochs = 120
lr_scheduler = 'cycle'
initial_lr = 1e-3
max_lr = 5e-4
ckpt_saving = True
save_at_end = False
use_mp = False
mixup_pct = -1
use_cont = True
use_blurred_training = False
subj_id = "02"
seed = 0
ckpt_path = None
cont_model = 'cnx'
ups_mode = '4x'

# need non-deterministic CuDNN for conv3D to work
utils.seed_everything(seed + local_rank, cudnn_deterministic=False)
torch.backends.cuda.matmul.allow_tf32 = True

# if running command line, read in args or config file values and override above params
try:
    config_keys = [k for k, v in globals().items() if not k.startswith('_') \
                   and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
except:
    pass


autoenc.to(device)

if use_cont:
    mixup_pct = -1
    if cont_model == 'cnx':
        cnx = ConvnextXL('/mnt/sda/songzengyu/code/brain_imager/data/convnext_xlarge_alpha0.75_fullckpt.pth')
        cnx.requires_grad_(False)
        cnx.eval()
        cnx.to(device)
    train_augs = AugmentationSequential(
        # kornia.augmentation.RandomCrop((480, 480), p=0.3),
        # kornia.augmentation.Resize((512, 512)),
        kornia.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8),
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.RandomSolarize(p=0.2),
        kornia.augmentation.RandomGaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0), p=0.1),
        kornia.augmentation.RandomResizedCrop((512, 512), scale=(0.5, 1.0)),
        data_keys=["input"],
    )

outdir = f'/mnt/sda/songzengyu/code/brain_imager/other_subject/subj02/trained_model/fmri-panoramic'

# # Prep models and data loaders
if local_rank == 0: print('Creating voxel2sd...')
in_dims = {'01': 15724, '02': 14278, '05': 13039, '07': 12682}
if voxel_dims == 1:  # 1D data
    voxel2sd = Voxel2StableDiffusionModel(use_cont=use_cont, in_dim=in_dims[subj_id], ups_mode=ups_mode)

voxel2sd.to(device)
voxel2sd = torch.nn.SyncBatchNorm.convert_sync_batchnorm(voxel2sd)

if local_rank == 0: print('Pulling NSD webdataset data...')
train_url = f"{{/mnt/sda/songzengyu/code/brain_imager/other_subject/subj02/data/train/train_subj{subj_id}_{{0..17}}.tar,/mnt/sda/songzengyu/code/brain_imager/other_subject/subj02/data/val/val_subj{subj_id}_0.tar}}"
val_url = f"/mnt/sda/songzengyu/code/brain_imager/other_subject/subj02/data/test/test_subj{subj_id}_{{0..1}}.tar"
meta_url = f"/mnt/sda/songzengyu/code/brain_imager/other_subject/subj02/data/metadata_subj{subj_id}.json"

if local_rank == 0: print('Prepping train and validation dataloaders...')
num_train = (8559 + 300)*3
num_val = 982

train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    batch_size, 'images',
    num_devices=num_devices,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=16,
    cache_dir='/mnt/sda/songzengyu/code/brain_imager/data',  # "/tmp/wds-cache",
    seed=seed,
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images", "coco"],
    local_rank=local_rank,
    world_size=1,
)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
opt_grouped_parameters = [
    {'params': [p for n, p in voxel2sd.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 1e-2},
    {'params': [p for n, p in voxel2sd.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=1e-3)
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                   total_steps=num_epochs * ((num_train // batch_size) // num_devices),
                                                   final_div_factor=1000,
                                                   last_epoch=-1, pct_start=2 / num_epochs)


def save_ckpt(tag):
    ckpt_path = os.path.join(outdir, f'{tag}.pth')
    print(f'saving {ckpt_path}')
    if local_rank == 0:
        state_dict = voxel2sd.state_dict()
        for key in list(state_dict.keys()):
            if 'module.' in key:
                state_dict[key.replace('module.', '')] = state_dict[key]
                del state_dict[key]
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': losses,
                'val_losses': val_losses,
                'lrs': lrs,
            }, ckpt_path)
        except:
            print('Failed to save weights')
            print(traceback.format_exc())


if local_rank == 0: print("Done with model preparations!")
epoch = 0
progress_bar = tqdm(range(epoch, num_epochs), ncols=150, disable=(local_rank != 0))
losses = []
val_losses = []
lrs = []
best_val_loss = 1e10
mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(1, 3, 1, 1)
std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(1, 3, 1, 1)

panoramic_imgs = np.load('/mnt/sda/songzengyu/code/brain_imager/other_subject/subj02/panoramic_data/subj02_divided_images_all.npy')

for epoch in progress_bar:
    voxel2sd.train()

    loss_mse_sum = 0
    loss_cont_sum = 0
    val_loss_mse_sum = 0

    for train_i, (voxel, image, coco) in enumerate(train_dl):
        optimizer.zero_grad()

        panoramic_img = panoramic_imgs[coco.squeeze().cpu()]
        panoramic_img_tensor = torch.tensor(panoramic_img).to(device).float()

        # image = image.to(device).float()
        image_512 = F.interpolate(panoramic_img_tensor, (512, 512), mode='bilinear', align_corners=False, antialias=True)

        voxel = voxel.to(device).float()
        voxel = utils.voxel_select(voxel)
        if epoch <= mixup_pct * num_epochs:
            voxel, perm, betas, select = utils.mixco(voxel)
        else:
            select = None

        with torch.cuda.amp.autocast(enabled=use_mp):
            autoenc_image = kornia.filters.median_blur(image_512, (15, 15)) if use_blurred_training else image_512
            image_enc = autoenc.encode(2 * autoenc_image - 1).latent_dist.mode() * 0.18215

            if use_cont:
                image_enc_pred, transformer_feats = voxel2sd(voxel, return_transformer_feats=True)
            else:
                image_enc_pred = voxel2sd(voxel)

            if epoch <= mixup_pct * num_epochs:
                image_enc_shuf = image_enc[perm]
                betas_shape = [-1] + [1] * (len(image_enc.shape) - 1)
                image_enc[select] = image_enc[select] * betas[select].reshape(*betas_shape) + \
                                    image_enc_shuf[select] * (1 - betas[select]).reshape(*betas_shape)

            if use_cont:
                image_norm = (image_512 - mean) / std
                image_aug = (train_augs(image_512) - mean) / std
                _, cnx_embeds = cnx(image_norm)
                _, cnx_aug_embeds = cnx(image_aug)

                cont_loss = utils.soft_cont_loss(
                    F.normalize(transformer_feats.reshape(-1, transformer_feats.shape[-1]), dim=-1),
                    F.normalize(cnx_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    F.normalize(cnx_aug_embeds.reshape(-1, cnx_embeds.shape[-1]), dim=-1),
                    temp=0.075,
                    distributed=False
                )
                del image_aug, cnx_embeds, transformer_feats
            else:
                cont_loss = torch.tensor(0)

            # mse_loss = F.mse_loss(image_enc_pred, image_enc)/0.18215
            mse_loss = F.l1_loss(image_enc_pred, image_enc)
            del image_512, voxel
            loss = mse_loss / 0.18215 + 0.1 * cont_loss
            loss_mse_sum += mse_loss.item()
            loss_cont_sum += cont_loss.item()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

        loss.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    if local_rank == 0:
        voxel2sd.eval()
        for val_i, (voxel, image, coco) in enumerate(val_dl):
            with torch.inference_mode():
                panoramic_img = panoramic_imgs[coco.squeeze().cpu()]
                panoramic_img_tensor = torch.tensor(panoramic_img).to(device).float()

                # image = image.to(device).float()
                image_512 = F.interpolate(panoramic_img_tensor, (512, 512), mode='bilinear', align_corners=False, antialias=True)
                voxel = voxel.to(device).float()
                voxel = voxel.mean(1)

                with torch.cuda.amp.autocast(enabled=use_mp):
                    image_enc = autoenc.encode(2 * image_512 - 1).latent_dist.mode() * 0.18215
                    if hasattr(voxel2sd, 'module'):
                        image_enc_pred = voxel2sd.module(voxel)
                    else:
                        image_enc_pred = voxel2sd(voxel)

                    mse_loss = F.mse_loss(image_enc_pred, image_enc)

                    val_loss_mse_sum += mse_loss.item()
                    val_losses.append(mse_loss.item())

        if not save_at_end and ckpt_saving:
            # save best model
            val_loss = np.mean(val_losses[-(val_i + 1):])
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_ckpt('best')

        if epoch == num_epochs - 1:
            save_ckpt('last')

        logs = {
            "train/loss": np.mean(losses[-(train_i + 1):]),
            "val/loss": np.mean(val_losses[-(val_i + 1):]),
            "train/lr": lrs[-1],
            "train/num_steps": len(losses),
            "train/loss_mse": loss_mse_sum / (train_i + 1),
            "train/loss_cont": loss_cont_sum / (train_i + 1),
            "val/loss_mse": val_loss_mse_sum / (val_i + 1),
        }
        progress_bar.set_postfix(**logs)










