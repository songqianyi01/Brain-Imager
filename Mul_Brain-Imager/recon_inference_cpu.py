#!/usr/bin/env python
# coding: utf-8

import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json
import argparse
import numpy as np
import math
from einops import rearrange
import time
import random
import string
import h5py
from tqdm import tqdm
import webdataset as wds

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from accelerate import Accelerator

# SDXL unCLIP requires code from https://github.com/Stability-AI/generative-models/tree/main
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder, FrozenOpenCLIPEmbedder2
from generative_models.sgm.models.diffusion import DiffusionEngine
from generative_models.sgm.util import append_dims
from omegaconf import OmegaConf

# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True

# custom functions #
import utils
from models import *

# 指定设备
device_cpu = "cpu"
device_cuda= "cuda"

# ====================== ARGS =======================
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--model_name", type=str, default="final_subj01_pretrained_1sess_24bs",
    help="will load ckpt for model found in ../train_logs/model_name",
)
parser.add_argument(
    "--data_path", type=str, default='/media/data/songzengyu/Mul_Brain-Imager/MindEyeV2-main/src/mindeyev2_dataset',
    help="Path to where NSD data is stored / where to download it to",
)
parser.add_argument(
    "--cache_dir", type=str, default='/media/data/songzengyu/Mul_Brain-Imager/MindEyeV2-main/src/cache_dir',
    help="Path to where misc. files downloaded from huggingface are stored. Defaults to current src directory.",
)
parser.add_argument(
    "--subj", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8],
    help="Validate on which subject?",
)
parser.add_argument(
    "--blurry_recon", action=argparse.BooleanOptionalAction, default=False,
)
parser.add_argument(
    "--n_blocks", type=int, default=4,
)
parser.add_argument(
    "--hidden_dim", type=int, default=4096,
)
parser.add_argument(
    "--new_test", action=argparse.BooleanOptionalAction, default=True,
)
parser.add_argument(
    "--seed", type=int, default=42,
)

args = parser.parse_args()

# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)

# seed all random functions
utils.seed_everything(seed)

# make output directory
os.makedirs("evals", exist_ok=True)
os.makedirs(f"evals/{model_name}", exist_ok=True)

# ====================== DATA LOAD =======================
voxels = {}
f = h5py.File(f'{data_path}/betas_all_subj0{subj}_fp32_renorm.hdf5', 'r')
betas = f['betas'][:]
betas = torch.Tensor(betas).to(device_cpu)
num_voxels = betas[0].shape[-1]
voxels[f'subj0{subj}'] = betas
print(f"num_voxels for subj0{subj}: {num_voxels}")

if not new_test:
    if subj == 3:
        num_test = 2113
    elif subj == 4:
        num_test = 1985
    elif subj == 6:
        num_test = 2113
    elif subj == 8:
        num_test = 1985
    else:
        num_test = 2770
    test_url = f"{data_path}/wds/subj0{subj}/test/" + "0.tar"
else:
    if subj == 3:
        num_test = 2371
    elif subj == 4:
        num_test = 2188
    elif subj == 6:
        num_test = 2371
    elif subj == 8:
        num_test = 2188
    else:
        num_test = 3000
    test_url = f"{data_path}/wds/subj0{subj}/new_test/" + "0.tar"

print(test_url)


def my_split_by_node(urls): return urls


test_data = wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node) \
    .decode("torch") \
    .rename(behav="behav.npy", past_behav="past_behav.npy", future_behav="future_behav.npy",
            olds_behav="olds_behav.npy") \
    .to_tuple(*["behav", "past_behav", "future_behav", "olds_behav"])
test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False, drop_last=True, pin_memory=False)
print(f"Loaded test dl for subj{subj}!\n")

# ====================== IMAGE & VOXEL PREP =======================
f = h5py.File(f'{data_path}/coco_images_224_float16.hdf5', 'r')
images = f['images']

test_images_idx = []
test_voxels_idx = []
for test_i, (behav, past_behav, future_behav, old_behav) in enumerate(test_dl):
    test_voxels = voxels[f'subj0{subj}'][behav[:, 0, 5].cpu().long()]
    test_voxels_idx = np.append(test_images_idx, behav[:, 0, 5].cpu().numpy())
    test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
test_images_idx = test_images_idx.astype(int)
test_voxels_idx = test_voxels_idx.astype(int)
assert (test_i + 1) * num_test == len(test_voxels) == len(test_images_idx)



# ====================== CLIP EMBEDDER (LOCAL PATH) =======================
clip_img_embedder = FrozenOpenCLIPImageEmbedder(
    arch="ViT-bigG-14",
    version="/media/data/songzengyu/Mul_Brain-Imager/MindEyeV2-main/src/cache_dir/open_clip_pytorch_model.bin",
    output_tokens=True,
    only_tokens=True,
)
clip_img_embedder.to(device_cpu)
clip_seq_dim = 256
clip_emb_dim = 1664

# ====================== MODEL =======================
class MindEyeModule(nn.Module):
    def __init__(self):
        super(MindEyeModule, self).__init__()

    def forward(self, x):
        return x


model = MindEyeModule()


class RidgeRegression(torch.nn.Module):
    def __init__(self, input_sizes, out_features):
        super(RidgeRegression, self).__init__()
        self.out_features = out_features
        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        out = self.linears[subj_idx](x[:, 0]).unsqueeze(1)
        return out


model.ridge = RidgeRegression([num_voxels], out_features=hidden_dim)

from diffusers.models.vae import Decoder
from models import BrainNetwork

model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, seq_len=1,
                              clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim)

out_dim = clip_emb_dim
depth = 6
dim_head = 52
heads = clip_emb_dim // 52
timesteps = 100

prior_network = PriorNetwork(
    dim=out_dim,
    depth=depth,
    dim_head=dim_head,
    heads=heads,
    causal=False,
    num_tokens=clip_seq_dim,
    learned_query_mode="pos_emb"
)

model.diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
)
model.to(device_cpu)

# ====================== LOAD CKPT =======================
tag = 'last'
outdir = os.path.abspath(f'/media/data/songzengyu/Mul_Brain-Imager/MindEyeV2-main/src/train_logs/{model_name}')
print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
try:
    checkpoint = torch.load(outdir + f'/{tag}.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict, strict=True)
    del checkpoint
except:
    import deepspeed

    state_dict = deepspeed.utils.zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(checkpoint_dir=outdir, tag=tag)
    model.load_state_dict(state_dict, strict=False)
    del state_dict
print("ckpt loaded!")

# ====================== DIFFUSION UNCLIP =======================
config = OmegaConf.load("generative_models/configs/unclip6.yaml")
config = OmegaConf.to_container(config, resolve=True)
unclip_params = config["model"]["params"]
network_config = unclip_params["network_config"]
denoiser_config = unclip_params["denoiser_config"]
first_stage_config = unclip_params["first_stage_config"]
conditioner_config = unclip_params["conditioner_config"]
sampler_config = unclip_params["sampler_config"]
scale_factor = unclip_params["scale_factor"]
disable_first_stage_autocast = unclip_params["disable_first_stage_autocast"]
first_stage_config['target'] = 'sgm.models.autoencoder.AutoencoderKL'
sampler_config['params']['num_steps'] = 38

diffusion_engine = DiffusionEngine(network_config=network_config,
                                   denoiser_config=denoiser_config,
                                   first_stage_config=first_stage_config,
                                   conditioner_config=conditioner_config,
                                   sampler_config=sampler_config,
                                   scale_factor=scale_factor,
                                   disable_first_stage_autocast=disable_first_stage_autocast)
diffusion_engine.eval().requires_grad_(False)
diffusion_engine.to(device_cuda)

ckpt_path = f'{cache_dir}/unclip6_epoch0_step110000.ckpt'
ckpt = torch.load(ckpt_path, map_location='cpu')
diffusion_engine.load_state_dict(ckpt['state_dict'])

batch = {"jpg": torch.randn(1, 3, 1, 1).to(device_cuda),
         "original_size_as_tuple": torch.ones(1, 2).to(device_cuda) * 768,
         "crop_coords_top_left": torch.zeros(1, 2).to(device_cuda)}
out = diffusion_engine.conditioner(batch)
vector_suffix = out["vector"].to(device_cuda)
print("unclip loaded!")

# ====================== RECONSTRUCTION (NO CAPTION !) =======================

model.eval().requires_grad_(False)

all_blurryrecons = None
all_recons = None
all_clipvoxels = None

minibatch_size = 1
num_samples_per_image = 1

with torch.no_grad():
    for batch in tqdm(range(0, len(np.unique(test_images_idx)), minibatch_size)):
        uniq_imgs = np.unique(test_images_idx)[batch:batch + minibatch_size]
        voxel = None
        for uniq_img in uniq_imgs:
            locs = np.where(test_images_idx == uniq_img)[0]
            if len(locs) == 1:
                locs = locs.repeat(3)
            elif len(locs) == 2:
                locs = locs.repeat(2)[:3]
            assert len(locs) == 3
            if voxel is None:
                voxel = test_voxels[None, locs]
            else:
                voxel = torch.vstack((voxel, test_voxels[None, locs]))
        voxel = voxel.to(device_cpu)

        for rep in range(3):
            voxel_ridge = model.ridge(voxel[:, [rep]], 0)
            backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
            if rep == 0:
                clip_voxels = clip_voxels0
                backbone = backbone0
                blurry_image_enc = blurry_image_enc0[0]
            else:
                clip_voxels += clip_voxels0
                backbone += backbone0
                blurry_image_enc += blurry_image_enc0[0]
        clip_voxels /= 3
        backbone /= 3
        blurry_image_enc /= 3

        if all_clipvoxels is None:
            all_clipvoxels = clip_voxels.cpu()
        else:
            all_clipvoxels = torch.vstack((all_clipvoxels, clip_voxels.cpu()))


        prior_out = model.diffusion_prior.p_sample_loop(backbone.shape,
                                                        text_cond=dict(text_embed=backbone),
                                                        cond_scale=1., timesteps=20)

        # 图像重建
        for i in range(len(voxel)):
            samples = utils.unclip_recon(prior_out[[i]].to(device_cuda),
                                         diffusion_engine,
                                         vector_suffix,
                                         num_samples=num_samples_per_image)
            if all_recons is None:
                all_recons = samples.cpu()
            else:
                all_recons = torch.vstack((all_recons, samples.cpu()))
            torch.cuda.empty_cache()



# ====================== SAVE =======================
imsize = 256
all_recons = transforms.Resize((imsize, imsize))(all_recons).float()
print(all_recons.shape)
torch.save(all_recons, f"evals/{model_name}/{model_name}_all_recons.pt")
torch.save(all_clipvoxels, f"evals/{model_name}/{model_name}_all_clipvoxels.pt")
print(f"saved {model_name} outputs!")

if not utils.is_interactive():
    sys.exit(0)
