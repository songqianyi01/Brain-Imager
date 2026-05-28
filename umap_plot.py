import warnings
warnings.filterwarnings("ignore")
import os

import matplotlib
matplotlib.use('Agg')

# 环境配置
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['NUMBA_DISABLE_DEPRECATION_WARNINGS'] = '1'

import torch
import umap
import numpy as np
import matplotlib.pyplot as plt

seed = 42

# ===================== 加载提前提取好的特征 =====================
all_clipimgs = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_text_features.pt")
all_backbones = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_backbone.pt")
all_prior_out = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_prior.pt")
all_proj_out = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_proj.pt")

k = np.random.randint(0, 16)
all_prior_out = all_prior_out[:, k, :, :]  # shape → (982,77,768)

# 统一转为 numpy
clipimgs_np = all_clipimgs.flatten(1).detach().cpu().numpy()
prior_out_np = all_prior_out.flatten(1).detach().cpu().numpy()
backbones_np = all_backbones.flatten(1).detach().cpu().numpy()
proj_out_np = all_proj_out.flatten(1).detach().cpu().numpy()

# ===================== UMAP 1: CLIP Text x Retrievals =====================
print("正在绘制第 1 张图...")
reducer = umap.UMAP(random_state=seed)
plt.figure(figsize=(5, 5))
data1 = clipimgs_np
data2 = proj_out_np
color = "orange"

embedding = reducer.fit_transform(np.concatenate([data1, data2], axis=0))
euclidean_dist = np.mean(np.linalg.norm(embedding[:len(data1)] - embedding[len(data1):], axis=1))

# 散点
plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], c='blue', alpha=0.5)
plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], c=color, alpha=0.5)

# ✅ 1. 去掉XY轴刻度（隐藏 -5, 0, 5 等数字）
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# ✅ 2. 画彩色标题（模仿示例图：蓝色 x 黑色 橙色）
# 使用更高的垂直位置，确保错开
ax = plt.gca()
# 左边：蓝色
ax.text(0.355, 1.08, 'CLIP Text', color='blue', fontsize=14, ha='center', transform=ax.transAxes)
# 中间：黑色 x
ax.text(0.5, 1.08, 'x', color='black', fontsize=14, ha='center', transform=ax.transAxes)
# 右边：对应颜色
ax.text(0.638, 1.08, 'Retrieval', color='orange', fontsize=14, ha='center', transform=ax.transAxes)

# ✅ 3. 显示英文距离标签（在标题下方）
ax.text(0.5, 1.03, f'Avg. euclidean distance = {euclidean_dist:.2f}', color='black', fontsize=12, ha='center', transform=ax.transAxes)

# ✅ 4. 设置XY轴标签（UMAP 1/2）
plt.xlabel("UMAP 1", fontsize=12)
plt.ylabel("UMAP 2", fontsize=12)

plt.savefig('/media/data/songzengyu/Brain-Imager/code/run_server_github/result_umap/text_umap/umap_retrieval.png', dpi=300, bbox_inches='tight')
plt.close('all')
print("umap_retrieval done!")

# ===================== UMAP 2: CLIP Text x Prior Out =====================
print("正在绘制第 2 张图...")
reducer = umap.UMAP(random_state=seed)
plt.figure(figsize=(5, 5))
data1 = clipimgs_np
data2 = prior_out_np
color = "red"

embedding = reducer.fit_transform(np.concatenate([data1, data2], axis=0))
euclidean_dist = np.mean(np.linalg.norm(embedding[:len(data1)] - embedding[len(data1):], axis=1))

plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], c='blue', alpha=0.5)
plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], c=color, alpha=0.5)

# 去掉刻度
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# 彩色标题
ax = plt.gca()
ax.text(0.355, 1.08, 'CLIP Text', color='blue', fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.5, 1.08, 'x', color='black', fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.705, 1.08, 'Diffusion Prior', color='red', fontsize=14, ha='center', transform=ax.transAxes)

# 英文距离
ax.text(0.5, 1.03, f'Avg. euclidean distance = {euclidean_dist:.2f}', color='black', fontsize=12, ha='center', transform=ax.transAxes)

plt.xlabel("UMAP 1", fontsize=12)
plt.ylabel("UMAP 2", fontsize=12)

plt.savefig('/media/data/songzengyu/Brain-Imager/code/run_server_github/result_umap/text_umap/umap_prior.png', dpi=300, bbox_inches='tight')
plt.close('all')
print("umap_prior done!")

# ===================== UMAP 3: CLIP Text x Backbones =====================
print("正在绘制第 3 张图...")
reducer = umap.UMAP(random_state=seed)
plt.figure(figsize=(5, 5))
data1 = clipimgs_np
data2 = backbones_np
color = "green"

embedding = reducer.fit_transform(np.concatenate([data1, data2], axis=0))
euclidean_dist = np.mean(np.linalg.norm(embedding[:len(data1)] - embedding[len(data1):], axis=1))

plt.scatter(embedding[:len(data1), 0], embedding[:len(data1), 1], c='blue', alpha=0.5)
plt.scatter(embedding[len(data1):, 0], embedding[len(data1):, 1], c=color, alpha=0.5)

# 去掉刻度
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# 彩色标题
ax = plt.gca()
ax.text(0.355, 1.08, 'CLIP Text', color='blue', fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.5, 1.08, 'x', color='black', fontsize=14, ha='center', transform=ax.transAxes)
ax.text(0.71, 1.08, 'MLP Backbone', color='green', fontsize=14, ha='center', transform=ax.transAxes)

# 英文距离
ax.text(0.5, 1.03, f'Avg. euclidean distance = {euclidean_dist:.2f}', color='black', fontsize=12, ha='center', transform=ax.transAxes)

plt.xlabel("UMAP 1", fontsize=12)
plt.ylabel("UMAP 2", fontsize=12)

plt.savefig('/media/data/songzengyu/Brain-Imager/code/run_server_github/result_umap/text_umap/umap_backbone.png', dpi=300, bbox_inches='tight')
plt.close('all')
print("umap_backbone done!")

print("\n全部三张图绘制完成！")