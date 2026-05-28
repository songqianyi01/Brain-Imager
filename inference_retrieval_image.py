#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch.nn as nn

# ===================== 演示配置 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.switch_backend('Agg')

# 四个文件路径
feat_img_clip_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_image_features.pt"
feat_fmri_proj_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/img_proj.pt"
images_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/gt_test_images.pt"

save_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_retrieval_image/demo_image_retrieval_top5.png"

# ===================== 加载所有预存数据 =====================
print("加载预计算特征")
fmri_proj = torch.load(feat_fmri_proj_path, map_location=device)
img_clip = torch.load(feat_img_clip_path, map_location=device)
images = torch.load(images_path, map_location='cpu')

print("加载完成")
print("fmri_proj shape:", fmri_proj.shape)
print("img_clip shape: ", img_clip.shape)
print("images shape:   ", images.shape)

img_clip = nn.functional.normalize(img_clip.reshape(len(img_clip), -1), dim=-1)


# ========== 单样本检索 ==========
def retrieve_single(sample_idx, topk=5):
    query = fmri_proj[sample_idx:sample_idx + 1]
    query = query.reshape(1, -1)  # [1, 257*768]
    query = nn.functional.normalize(query, dim=-1)

    sim = utils.batchwise_cosine_similarity(query, img_clip)
    sim = sim.cpu().numpy().squeeze()
    topk_idx = np.flip(np.argsort(sim))[:topk]
    return topk_idx, sim[topk_idx]


# 结果可视化
def plot_retrieval(idx, top5_idx):
    real_img = images[idx]
    top_imgs = [images[i] for i in top5_idx]

    fig, axes = plt.subplots(1, 6, figsize=(20, 4))

    axes[0].imshow(utils.torch_to_Image(real_img))
    axes[0].set_title(f"Query {idx}\nGround Truth", fontsize=14, weight='bold')
    axes[0].axis('off')

    for i, img in enumerate(top_imgs):
        axes[i + 1].imshow(utils.torch_to_Image(img))
        axes[i + 1].set_title(f"Top {i + 1}", fontsize=14, weight='bold')
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print("图片已保存至", save_path)

# 主程序
if __name__ == '__main__':
    sample_idx = 100
    top5_idx, scores = retrieve_single(sample_idx)

    print("样本序号:", sample_idx)
    print("Top5 索引:", list(top5_idx))
    print("Top5 相似度:", [round(s, 4) for s in scores])

    plot_retrieval(sample_idx, top5_idx)