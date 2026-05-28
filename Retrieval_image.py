#!/usr/bin/env python
# coding: utf-8
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse
import csv

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 导入自定义模块
import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, VersatileDiffusionPriorNetwork

# 固定随机种子
seed = 42
utils.seed_everything(seed=seed)

# 不同被试的体素数量
subj_voxels = {1: 15724, 2: 14278, 5: 13039, 7: 12682}
num_voxels = subj_voxels[1]

# ===================== 保存路径设置 =====================
save_dir = "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_retrieval_image"
os.makedirs(save_dir, exist_ok=True)

# ===================== 加载测试数据 =====================
num_val = 982
batch_size = 300
val_loops = 30
voxels_key = 'nsdgeneral.npy'
val_url = f"/media/data/songzengyu/Brain-Imager/data/data_single/test/subj01/test_subj01_{{0..1}}.tar"
val_data = wds.WebDataset(val_url, resampled=True) \
    .decode("torch") \
    .rename(images="jpg;png", voxels=voxels_key) \
    .to_tuple("voxels", "images") \
    .batched(batch_size, partial=False) \
    .with_epoch(val_loops)

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# ===================== 加载预训练模型 =====================
out_dim = 257 * 768
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

prior_network = VersatileDiffusionPriorNetwork(
    dim=768, depth=6, dim_head=64, heads=12, causal=False,learned_query_mode="pos_emb",
).to(device)

diffusion_prior = BrainDiffusionPrior(
    net=prior_network, image_embed_dim=768, condition_on_text_encodings=False,cond_drop_prob=0.2,image_embed_scale=None,
    timesteps=100, voxel2clip=voxel2clip
).to(device)

ckpt_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/model/CI.pth"
checkpoint = torch.load(ckpt_path, map_location=device)
diffusion_prior.load_state_dict(checkpoint['model_state_dict'], strict=False)
diffusion_prior.eval().to(device)
diffusion_priors = [diffusion_prior]

# ===================== 双向检索（逐个样本计算 + Top1/5/10）=====================
print("\n=== 开始 fMRI ↔ 图像 双向检索 (Top1/5/10) ===")

# 完全和文本检索代码格式一致
percent_correct_fwds_top1 = []
percent_correct_fwds_top5 = []
percent_correct_fwds_top10 = []

percent_correct_bwds_top1 = []
percent_correct_bwds_top5 = []
percent_correct_bwds_top10 = []

# 保存最后一批用于可视化
last_fwd_sim = None
last_bwd_sim = None
last_img_batch = None

with torch.no_grad():
    for val_i, (voxel_batch, img_batch) in enumerate(tqdm(val_dl, total=val_loops)):
        voxel_batch = torch.mean(voxel_batch, axis=1).to(device)
        B = voxel_batch.shape[0]

        img_emb_list = []
        fmri_emb_list = []

        # 逐个样本计算（省显存）
        for i in range(B):
            voxel = voxel_batch[i:i+1]
            img = img_batch[i:i+1].to(device)

            img_emb = clip_extractor.embed_image(img).float()
            _, fmri_emb = diffusion_prior.voxel2clip(voxel.float())

            img_emb = nn.functional.normalize(img_emb.reshape(1, -1), dim=-1)
            fmri_emb = nn.functional.normalize(fmri_emb.reshape(1, -1), dim=-1)

            img_emb_list.append(img_emb.cpu())
            fmri_emb_list.append(fmri_emb.cpu())

            del voxel, img, img_emb, fmri_emb
            torch.cuda.empty_cache()

        # 拼接特征
        img_emb_all = torch.cat(img_emb_list, dim=0).to(device)
        fmri_emb_all = torch.cat(fmri_emb_list, dim=0).to(device)
        labels = torch.arange(len(img_emb_all)).to(device)

        # 相似度
        fwd_sim = utils.batchwise_cosine_similarity(fmri_emb_all, img_emb_all)
        bwd_sim = utils.batchwise_cosine_similarity(img_emb_all, fmri_emb_all)

        # 保存最后一批
        last_fwd_sim = fwd_sim.detach().cpu()
        last_bwd_sim = bwd_sim.detach().cpu()
        last_img_batch = img_batch.detach().cpu()

        # ==== 计算 Top1 / Top5 / Top10 ====
        # fMRI → 图像
        percent_correct_fwds_top1.append(utils.topk(fwd_sim, labels, k=1).item())
        percent_correct_fwds_top5.append(utils.topk(fwd_sim, labels, k=5).item())
        percent_correct_fwds_top10.append(utils.topk(fwd_sim, labels, k=10).item())

        # 图像 → fMRI
        percent_correct_bwds_top1.append(utils.topk(bwd_sim, labels, k=1).item())
        percent_correct_bwds_top5.append(utils.topk(bwd_sim, labels, k=5).item())
        percent_correct_bwds_top10.append(utils.topk(bwd_sim, labels, k=10).item())

        # 清理显存
        del voxel_batch, img_batch, img_emb_all, fmri_emb_all, labels, fwd_sim, bwd_sim
        del img_emb_list, fmri_emb_list
        torch.cuda.empty_cache()

# ===================== 统计结果（和文本检索完全一样的函数）=====================
def calc_metrics(data):
    mean = np.mean(data)
    sem = np.std(data) / np.sqrt(len(data))
    ci = stats.norm.interval(0.95, loc=mean, scale=sem)
    return mean, ci[0], ci[1]

# fMRI → 图像
fwd1_mean, fwd1_low, fwd1_high = calc_metrics(percent_correct_fwds_top1)
fwd5_mean, fwd5_low, fwd5_high = calc_metrics(percent_correct_fwds_top5)
fwd10_mean, fwd10_low, fwd10_high = calc_metrics(percent_correct_fwds_top10)

# 图像 → fMRI
bwd1_mean, bwd1_low, bwd1_high = calc_metrics(percent_correct_bwds_top1)
bwd5_mean, bwd5_low, bwd5_high = calc_metrics(percent_correct_bwds_top5)
bwd10_mean, bwd10_low, bwd10_high = calc_metrics(percent_correct_bwds_top10)

# 打印
print(f"\n=== 最终图像检索结果 ===")
print(f"fMRI→图像 Top-1 准确率: {fwd1_mean:.4f}  95%CI: [{fwd1_low:.4f}, {fwd1_high:.4f}]")
print(f"fMRI→图像 Top-5 准确率: {fwd5_mean:.4f}  95%CI: [{fwd5_low:.4f}, {fwd5_high:.4f}]")
print(f"fMRI→图像 Top-10准确率: {fwd10_mean:.4f} 95%CI: [{fwd10_low:.4f}, {fwd10_high:.4f}]")
print(f"图像→fMRI Top-1 准确率: {bwd1_mean:.4f}  95%CI: [{bwd1_low:.4f}, {bwd1_high:.4f}]")
print(f"图像→fMRI Top-5 准确率: {bwd5_mean:.4f}  95%CI: [{bwd5_low:.4f}, {bwd5_high:.4f}]")
print(f"图像→fMRI Top-10准确率: {bwd10_mean:.4f} 95%CI: [{bwd10_low:.4f}, {bwd10_high:.4f}]")

# ===================== 保存CSV（和文本检索格式完全一致）=====================
csv_path = os.path.join(save_dir, "image_retrieval_accuracy.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["direction", "topk", "acc", "ci_lower", "ci_upper"])
    writer.writerow(["fmri2image", 1, round(fwd1_mean,4), round(fwd1_low,4), round(fwd1_high,4)])
    writer.writerow(["fmri2image", 5, round(fwd5_mean,4), round(fwd5_low,4), round(fwd5_high,4)])
    writer.writerow(["fmri2image",10, round(fwd10_mean,4), round(fwd10_low,4), round(fwd10_high,4)])
    writer.writerow(["image2fmri", 1, round(bwd1_mean,4), round(bwd1_low,4), round(bwd1_high,4)])
    writer.writerow(["image2fmri", 5, round(bwd5_mean,4), round(bwd5_low,4), round(bwd5_high,4)])
    writer.writerow(["image2fmri",10, round(bwd10_mean,4), round(bwd10_low,4), round(bwd10_high,4)])

print(f"\n✅ 图像检索结果已保存到: {csv_path}")

# ===================== 可视化 =====================
plt.switch_backend('Agg')

def visualize_retrieval(sim_matrix, images, title, save_path, n_samples=4, topk=5):
    fig, ax = plt.subplots(n_samples, topk + 1, figsize=(14, 3 * n_samples))
    fig.suptitle(title, fontsize=16)

    for i in range(n_samples):
        ax[i, 0].imshow(utils.torch_to_Image(images[i]))
        ax[i, 0].set_title("Original")
        ax[i, 0].axis("off")

        top_indices = np.flip(np.argsort(sim_matrix[i]))[:topk]
        for j, idx in enumerate(top_indices):
            ax[i, j+1].imshow(utils.torch_to_Image(images[idx]))
            ax[i, j+1].set_title(f"Top {j+1}")
            ax[i, j+1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 可视化已保存: {save_path}")

# 画图
fwd_img_path = os.path.join(save_dir, "fmri_to_image_retrieval.png")
bwd_img_path = os.path.join(save_dir, "image_to_fmri_retrieval.png")

visualize_retrieval(last_fwd_sim.numpy(), last_img_batch, "fMRI → Image Retrieval", fwd_img_path)
visualize_retrieval(last_bwd_sim.numpy(), last_img_batch, "Image → fMRI Retrieval", bwd_img_path)