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

# ===================== 文本检索对应的4个pt文件 =====================
feat_text_clip_path  = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_text_features.pt"
feat_fmri_proj_path  = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_proj.pt"
texts_path           = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/gt_captions.pt"

save_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_retrieval_text/demo_text_retrieval_top5.png"

# ===================== 加载所有预存数据 =====================
print("加载预计算特征")
fmri_proj = torch.load(feat_fmri_proj_path, map_location=device)
text_clip = torch.load(feat_text_clip_path, map_location=device)
texts     = torch.load(texts_path, map_location='cpu')

print("加载完成")
print("fmri_proj shape:", fmri_proj.shape)
print("text_clip shape: ", text_clip.shape)
print("texts数量:   ", len(texts))

# 预处理：展平+归一化
text_clip = nn.functional.normalize(text_clip.reshape(len(text_clip), -1), dim=-1)

# ========== 单样本检索 ==========
def retrieve_single(sample_idx, topk=5):
    query = fmri_proj[sample_idx:sample_idx + 1]
    query = query.reshape(1, -1)
    query = nn.functional.normalize(query, dim=-1)

    sim = utils.batchwise_cosine_similarity(query, text_clip)
    sim = sim.cpu().numpy().squeeze()
    topk_idx = np.flip(np.argsort(sim))[:topk]
    return topk_idx, sim[topk_idx]

# ========== 文本检索可视化 ==========
def plot_retrieval(idx, top5_idx):
    real_text = texts[idx]
    top_texts = [texts[i] for i in top5_idx]

    fig, axes = plt.subplots(6, 1, figsize=(20, 6))

    # 全部加粗：Query + Ground Truth
    axes[0].text(0.5, 0.5, f"$\\mathbf{{Query {idx}}}$\n$\\mathbf{{Ground Truth}}$\n{real_text}",
                 fontsize=14, ha='center', va='center', wrap=True)
    axes[0].axis("off")

    # Top 1-5 标题加粗
    for i, txt in enumerate(top_texts):
        axes[i+1].text(0.5, 0.5, f"$\\mathbf{{Top {i+1}}}$\n{txt}",
                       fontsize=14, ha='center', va='center', wrap=True)
        axes[i+1].axis("off")

    plt.tight_layout(pad=0.5, h_pad=0.3)
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