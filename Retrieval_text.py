#!/usr/bin/env python
# coding: utf-8
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch
import torch.nn as nn
from tqdm import tqdm
import csv
import sys

# ===================== 设备配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# 导入自定义模块
import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork
from omegaconf import OmegaConf

# 加入taming路径（你原文里的）
sys.path.append('/media/data/songzengyu/Brain-Imager/code/run_server_github/taming-transformers')

# 固定随机种子
seed = 42
utils.seed_everything(seed=seed)

# 不同被试的体素数量
subj_voxels = {1: 15724, 2: 14278, 5: 13039, 7: 12682}
num_voxels = subj_voxels[1]

# ===================== 保存路径设置 =====================
save_dir = "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_retrieval_text"
os.makedirs(save_dir, exist_ok=True)

# ===================== Stable Diffusion 加载（用于get_text_features）=====================
stable_diffusion_config_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/v1-inference.yaml'
stable_diffusion_skpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/sd-v1-4.ckpt'

config = OmegaConf.load(stable_diffusion_config_path)
def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location=device)
    sd = pl_sd["state_dict"]
    from ldm.util import instantiate_from_config
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    return model.to(device).eval()

LDM_model = load_model_from_config(config, stable_diffusion_skpt_path)

# 你训练时用的文本编码函数
def get_text_features(text):
    return LDM_model.get_learned_conditioning(text)

# =====================  直接加载 体素 + 文本 .pt 文件 =====================
fmri_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/voxels.pt"  # 你的体素文件
text_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/ours_git_captions.pt"  # 你的文本文件

print(f"加载 fMRI 体素: {fmri_path}")
fmri_all = torch.load(fmri_path).squeeze(1)  # [982, 15724]

print(f"加载 对应文本: {text_path}")
text_all = torch.load(text_path)  # list: 982条文本

print(f"数据加载完成 → fMRI: {fmri_all.shape}, 文本: {len(text_all)}")

# ===================== 加载文本预训练模型 CT.pth =====================
text_out_dim = 77 * 768
voxel2clip_text_kwargs = dict(in_dim=num_voxels, out_dim=text_out_dim, clip_size=768, use_projector=True)
voxel2clip_text = BrainNetwork(**voxel2clip_text_kwargs)
voxel2clip_text.requires_grad_(False)
voxel2clip_text.eval()

prior_network_text = VersatileDiffusionPriorNetwork(
    dim=768, depth=6, dim_head=64, heads=12, causal=False,
    num_tokens=77, learned_query_mode="pos_emb",
).to(device)

diffusion_prior_text = BrainDiffusionPrior(
    net=prior_network_text, image_embed_dim=768, condition_on_text_encodings=False, cond_drop_prob=0.2,
    image_embed_scale=None, timesteps=100, voxel2clip=voxel2clip_text
).to(device)

text_ckpt_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/model/CT.pth"
text_checkpoint = torch.load(text_ckpt_path, map_location=device)
diffusion_prior_text.load_state_dict(text_checkpoint['model_state_dict'], strict=False)
diffusion_prior_text.eval().to(device)

# ===================== 双向检索（直接用加载好的982个样本）=====================
print("\n=== 开始 fMRI ↔ 文本 双向检索 (Top1/5/10) ===")

batch_size = 300
val_loops = 30

# 存储所有 top-k 结果
percent_correct_fwds_top1 = []
percent_correct_fwds_top5 = []
percent_correct_fwds_top10 = []

percent_correct_bwds_top1 = []
percent_correct_bwds_top5 = []
percent_correct_bwds_top10 = []

with torch.no_grad():
    for val_i in tqdm(range(val_loops)):
        # 随机抽取一个batch（和图像检索逻辑完全一致，保证公平对比）
        indices = torch.randperm(982)[:batch_size]
        voxel_batch = fmri_all[indices].to(device)
        text_batch = [text_all[i] for i in indices]

        # 1. 体素预处理
        voxel_batch = torch.mean(voxel_batch, axis=1) if voxel_batch.dim() == 3 else voxel_batch

        # 2. 用SD编码真实文本
        text_emb = get_text_features(text_batch).float()

        # 3. fMRI预测文本嵌入
        _, fmri_text_emb = diffusion_prior_text.voxel2clip(voxel_batch.float())

        # 4. 归一化
        text_emb = nn.functional.normalize(text_emb.reshape(len(text_emb), -1), dim=-1)
        fmri_text_emb = nn.functional.normalize(fmri_text_emb.reshape(len(fmri_text_emb), -1), dim=-1)

        # 5. 相似度 & Top-1/5/10
        labels = torch.arange(len(text_emb)).to(device)
        fwd_sim = utils.batchwise_cosine_similarity(fmri_text_emb, text_emb)
        bwd_sim = utils.batchwise_cosine_similarity(text_emb, fmri_text_emb)

        # fMRI → 文本
        percent_correct_fwds_top1.append(utils.topk(fwd_sim, labels, k=1).item())
        percent_correct_fwds_top5.append(utils.topk(fwd_sim, labels, k=5).item())
        percent_correct_fwds_top10.append(utils.topk(fwd_sim, labels, k=10).item())

        # 文本 → fMRI
        percent_correct_bwds_top1.append(utils.topk(bwd_sim, labels, k=1).item())
        percent_correct_bwds_top5.append(utils.topk(bwd_sim, labels, k=5).item())
        percent_correct_bwds_top10.append(utils.topk(bwd_sim, labels, k=10).item())

# ===================== 统计结果 =====================
def calc_metrics(data):
    mean = np.mean(data)
    sem = np.std(data) / np.sqrt(len(data))
    ci = stats.norm.interval(0.95, loc=mean, scale=sem)
    return mean, ci[0], ci[1]

# fMRI → 文本
fwd1_mean, fwd1_low, fwd1_high = calc_metrics(percent_correct_fwds_top1)
fwd5_mean, fwd5_low, fwd5_high = calc_metrics(percent_correct_fwds_top5)
fwd10_mean, fwd10_low, fwd10_high = calc_metrics(percent_correct_fwds_top10)

# 文本 → fMRI
bwd1_mean, bwd1_low, bwd1_high = calc_metrics(percent_correct_bwds_top1)
bwd5_mean, bwd5_low, bwd5_high = calc_metrics(percent_correct_bwds_top5)
bwd10_mean, bwd10_low, bwd10_high = calc_metrics(percent_correct_bwds_top10)

print(f"\n=== 最终文本检索结果 ===")
print(f"fMRI→文本 Top-1 准确率: {fwd1_mean:.4f}  95%CI: [{fwd1_low:.4f}, {fwd1_high:.4f}]")
print(f"fMRI→文本 Top-5 准确率: {fwd5_mean:.4f}  95%CI: [{fwd5_low:.4f}, {fwd5_high:.4f}]")
print(f"fMRI→文本 Top-10准确率: {fwd10_mean:.4f} 95%CI: [{fwd10_low:.4f}, {fwd10_high:.4f}]")
print(f"文本→fMRI Top-1 准确率: {bwd1_mean:.4f}  95%CI: [{bwd1_low:.4f}, {bwd1_high:.4f}]")
print(f"文本→fMRI Top-5 准确率: {bwd5_mean:.4f}  95%CI: [{bwd5_low:.4f}, {bwd5_high:.4f}]")
print(f"文本→fMRI Top-10准确率: {bwd10_mean:.4f} 95%CI: [{bwd10_low:.4f}, {bwd10_high:.4f}]")

# ===================== 保存CSV =====================
csv_path = os.path.join(save_dir, "text_retrieval_accuracy.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["direction", "topk", "acc", "ci_lower", "ci_upper"])
    writer.writerow(["fmri2text", 1, round(fwd1_mean,4), round(fwd1_low,4), round(fwd1_high,4)])
    writer.writerow(["fmri2text", 5, round(fwd5_mean,4), round(fwd5_low,4), round(fwd5_high,4)])
    writer.writerow(["fmri2text",10, round(fwd10_mean,4), round(fwd10_low,4), round(fwd10_high,4)])
    writer.writerow(["text2fmri", 1, round(bwd1_mean,4), round(bwd1_low,4), round(bwd1_high,4)])
    writer.writerow(["text2fmri", 5, round(bwd5_mean,4), round(bwd5_low,4), round(bwd5_high,4)])
    writer.writerow(["text2fmri",10, round(bwd10_mean,4), round(bwd10_low,4), round(bwd10_high,4)])

print(f"\n 保存到: {csv_path}")

# ===================== 可视化 =====================
plt.switch_backend('Agg')

def visualize_text_retrieval(sim_matrix, text_list, title, save_path, n_samples=2, topk=5):
    fig, ax = plt.subplots(n_samples, 1, figsize=(16, 5))  # 更小高度
    fig.suptitle(title, fontsize=20, fontweight='bold')
    for i in range(n_samples):
        orig = text_list[i]
        top_idx = np.flip(np.argsort(sim_matrix[i]))[:topk]
        top_texts = [text_list[j] for j in top_idx]
        msg = f"[Original Text]:\n{orig}\n\n"
        msg += "\n".join([f"Top {k+1}: {t}" for k, t in enumerate(top_texts)])
        ax[i].text(0.02, 0.95, msg, transform=ax[i].transAxes,
                   verticalalignment='top', fontsize=14)  # 字体更大
        ax[i].axis("off")
    plt.tight_layout(pad=0.5, h_pad=1.0)  # 紧凑布局
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存图片: {save_path}")

# 用最后一个batch做可视化
fwd_text_path = os.path.join(save_dir, "fmri_to_text_retrieval.png")
bwd_text_path = os.path.join(save_dir, "text_to_fmri_retrieval.png")
visualize_text_retrieval(fwd_sim.cpu().numpy(), text_batch, "fMRI → Text Retrieval", fwd_text_path)
visualize_text_retrieval(bwd_sim.cpu().numpy(), text_batch, "Text → fMRI Retrieval", bwd_text_path)