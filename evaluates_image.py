#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from datetime import datetime
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
device = torch.device('cuda:0')
print("device:",device)
import utils

seed=42
utils.seed_everything(seed=seed)

# # Configurations
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--recon_path", type=str, default="/media/data/songzengyu/Brain-Imager/code/run_server_github/result/ours_test_images.pt",
    help="reconstruct",
)
parser.add_argument(
    "--all_images_path", type=str, default="/media/data/songzengyu/Brain-Imager/code/run_server_github/result/gt_test_images.pt",
    help="ground_true",
)
args = parser.parse_args()

# ====================== 加载数据 ======================
all_brain_recons = torch.load(f'{args.recon_path}')
all_images = torch.load(f'{args.all_images_path}')

print(all_images.shape)
print(all_brain_recons.shape)

all_images = all_images.to(device)
all_brain_recons = all_brain_recons.to(device).to(all_images.dtype).clamp(0,1)

# # Display reconstructions next to ground truth images
imsize = 256
all_images = transforms.Resize((imsize,imsize),antialias=True)(all_images)
all_brain_recons = transforms.Resize((imsize,imsize),antialias=True)(all_brain_recons)

np.random.seed(0)
ind = np.flip(np.array([112,119,101,44,159,22,173,174,175,189,981,243,249,255,265]))
all_interleaved = torch.zeros(len(ind)*2,3,imsize,imsize)
icount = 0
for t in ind:
    all_interleaved[icount] = all_images[t]
    all_interleaved[icount+1] = all_brain_recons[t]
    icount += 2

plt.rcParams["savefig.bbox"] = 'tight'
def save_image(imgs, path, figsize):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=figsize)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = transforms.ToPILImage()(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.savefig(path, dpi=300)
    plt.close()

grid = make_grid(all_interleaved, nrow=10, padding=2)
save_image(grid, "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/comparison.png", figsize=(20,16))
print("✅ 对比图已保存为：/media/data/songzengyu/Brain-Imager/code/run_server_github/result/comparison.png")


# # 2-Way Identification
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

@torch.no_grad()
def two_way_identification(all_brain_recons, all_images, model, preprocess, feature_layer=None, return_avg=True):
    preds = model(torch.stack([preprocess(recon) for recon in all_brain_recons], dim=0).to(device))
    reals = model(torch.stack([preprocess(indiv) for indiv in all_images], dim=0).to(device))
    if feature_layer is None:
        preds = preds.float().flatten(1).cpu().numpy()
        reals = reals.float().flatten(1).cpu().numpy()
    else:
        preds = preds[feature_layer].float().flatten(1).cpu().numpy()
        reals = reals[feature_layer].float().flatten(1).cpu().numpy()

    r = np.corrcoef(reals, preds)
    r = r[:len(all_images), len(all_images):]
    congruents = np.diag(r)

    success = r < congruents
    success_cnt = np.sum(success, 0)

    if return_avg:
        perf = np.mean(success_cnt) / (len(all_images)-1)
        return perf
    else:
        return success_cnt, len(all_images)-1

# ====================== 清空显存函数 ======================
def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

clean_gpu()

# ## PixCorr （无模型，无需清理）
preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
])
all_images_flattened = preprocess(all_images).reshape(len(all_images), -1).cpu()
all_brain_recons_flattened = preprocess(all_brain_recons).view(len(all_brain_recons), -1).cpu()

corrsum = 0
for i in tqdm(range(len(all_images))):
    corrsum += np.corrcoef(all_images_flattened[i], all_brain_recons_flattened[i])[0][1]
corrmean = corrsum / len(all_images)
pixcorr = corrmean
print(f"PixCorr: {pixcorr}")
clean_gpu()

# ## SSIM （无模型，无需清理）
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
])
img_gray = rgb2gray(preprocess(all_images).permute((0,2,3,1)).cpu())
recon_gray = rgb2gray(preprocess(all_brain_recons).permute((0,2,3,1)).cpu())

ssim_score=[]
for im,rec in tqdm(zip(img_gray,recon_gray),total=len(all_images)):
    ssim_score.append(ssim(rec, im, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))
ssim = np.mean(ssim_score)
print(f"SSIM: {ssim}")
clean_gpu()

# -------------------------- AlexNet --------------------------
from torchvision.models import alexnet, AlexNet_Weights
alex_weights = AlexNet_Weights.IMAGENET1K_V1
alex_model = create_feature_extractor(alexnet(weights=alex_weights), return_nodes=['features.4','features.11']).to(device)
alex_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, alex_model, preprocess, 'features.4')
alexnet2 = np.mean(all_per_correct)
print(f"early, AlexNet(2): {alexnet2:.4f}")

all_per_correct = two_way_identification(all_brain_recons.to(device).float(), all_images, alex_model, preprocess, 'features.11')
alexnet5 = np.mean(all_per_correct)
print(f"mid, AlexNet(5): {alexnet5:.4f}")

# 清理
del alex_model
clean_gpu()

# -------------------------- InceptionV3 --------------------------
from torchvision.models import inception_v3, Inception_V3_Weights
weights = Inception_V3_Weights.DEFAULT
inception_model = create_feature_extractor(inception_v3(weights=weights), return_nodes=['avgpool']).to(device)
inception_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Resize(342, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images, inception_model, preprocess, 'avgpool')
inception = np.mean(all_per_correct)
print(f"InceptionV3: {inception:.4f}")

# 清理
del inception_model
clean_gpu()

# -------------------------- CLIP --------------------------
import clip
clip_model, preprocess = clip.load("ViT-L/14", device=device)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
])

all_per_correct = two_way_identification(all_brain_recons, all_images, clip_model.encode_image, preprocess, None)
clip_ = np.mean(all_per_correct)
print(f"CLIP: {clip_:.4f}")

# 清理
del clip_model
clean_gpu()

# -------------------------- Efficient Net --------------------------
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
weights = EfficientNet_B1_Weights.DEFAULT
eff_model = create_feature_extractor(efficientnet_b1(weights=weights), return_nodes=['avgpool']).to(device)
eff_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Resize(255, interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

gt = eff_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = eff_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()
effnet = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
print("Efficient Net:",effnet)

# 清理
del eff_model
clean_gpu()

# -------------------------- SwAV --------------------------
swav_model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
swav_model = create_feature_extractor(swav_model, return_nodes=['avgpool']).to(device)
swav_model.eval().requires_grad_(False)

preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR,antialias=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

gt = swav_model(preprocess(all_images))['avgpool']
gt = gt.reshape(len(gt),-1).cpu().numpy()
fake = swav_model(preprocess(all_brain_recons))['avgpool']
fake = fake.reshape(len(fake),-1).cpu().numpy()
swav = np.array([sp.spatial.distance.correlation(gt[i],fake[i]) for i in range(len(gt))]).mean()
print("SwAV:",swav)

# 清理
del swav_model
clean_gpu()

# # Display in table
data = {
    "Metric": ["PixCorr", "SSIM", "AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP", "EffNet-B", "SwAV"],
    "Value": [pixcorr, ssim, alexnet2, alexnet5, inception, clip_, effnet, swav],
}

df = pd.DataFrame(data)
print(df.to_string(index=False))

if not utils.is_interactive():
    df.to_csv("/media/data/songzengyu/Brain-Imager/code/run_server_github/result/image_metrics", sep='\t')

plt.close('all')
clean_gpu()