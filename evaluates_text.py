import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import sys
sys.path.append("/media/data/songzengyu/Brain-Imager/code/run_server_github/coco-caption-master")
sys.path.append("/media/data/songzengyu/Brain-Imager/code/run_server_github/coco-caption-master/pycocoevalcap")
sys.path.append("/media/data/songzengyu/Brain-Imager/code/run_server_github/coco-caption-master/pycocoevalcap/cider")
import torch
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util
from transformers import CLIPModel, AutoTokenizer, AutoProcessor
import evaluate
import pandas as pd
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# ==============================================================================
all_captions = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/gt_captions.pt", map_location='cpu')
all_predcaptions = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/ours_decoding_texts.pt", map_location='cpu')
all_git_generated_captions = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/ours_git_captions.pt", map_location='cpu')


# ==============================================================================
# 1. 加载所有评估指标
# ==============================================================================
print("加载评估指标中...")
meteor = evaluate.load('/media/data/songzengyu/Brain-Imager/code/run_server_github/metrics/meteor.py')
rouge = evaluate.load('rouge')
cider = evaluate.load('/media/data/songzengyu/Brain-Imager/code/run_server_github/metrics/cider.py')
spice = evaluate.load('/media/data/songzengyu/Brain-Imager/code/run_server_github/metrics/spice.py')


# ==============================================================================
# 2. 指标计算
# ==============================================================================

# -------------------------- METEOR --------------------------
print("计算 METEOR...")
meteor_img_ref = meteor.compute(predictions=all_git_generated_captions, references=all_captions)
meteor_brain_ref = meteor.compute(predictions=all_predcaptions, references=all_captions)
meteor_brain_img = meteor.compute(predictions=all_predcaptions, references=all_git_generated_captions)
relative_brain_image_meteor = meteor_brain_img["meteor"] / meteor_img_ref["meteor"]

# -------------------------- ROUGE --------------------------
print("计算 ROUGE...")
rouge_img_ref = rouge.compute(predictions=all_git_generated_captions, references=all_captions)
rouge_brain_ref = rouge.compute(predictions=all_predcaptions, references=all_captions)
rouge_brain_img = rouge.compute(predictions=all_predcaptions, references=all_git_generated_captions)
relative_brain_image_rouge1 = rouge_brain_img['rouge1'] / rouge_img_ref['rouge1']
relative_brain_image_rougeL = rouge_brain_img['rougeL'] / rouge_img_ref['rougeL']

print("计算 CIDEr...")
cider_img_ref = cider.compute(predictions=all_git_generated_captions, references=all_captions)
cider_brain_ref = cider.compute(predictions=all_predcaptions, references=all_captions)
cider_brain_img = cider.compute(predictions=all_predcaptions, references=all_git_generated_captions)
relative_brain_image_cider = cider_brain_img["cider"] / cider_img_ref["cider"]
print(cider_brain_ref)
print(cider_brain_img)
# # -------------------------- CIDEr (官方 Python3 版) --------------------------
# print("计算 CIDEr...")
#
# # --------------------- 以下是纯 Python3 官方 CIDEr 实现 ---------------------
# import re
# import math
# import collections
# import numpy as np
#
# def preprocess_caption(s):
#     s = s.lower().strip()
#     s = re.sub(r'[^\w\s]', '', s)
#     return s
#
# def get_ngrams(s, n=4):
#     tokens = preprocess_caption(s).split()
#     ngrams = []
#     for i in range(1, n+1):
#         for j in range(len(tokens)-i+1):
#             ngrams.append(tuple(tokens[j:j+i]))
#     return ngrams
#
# def compute_cider_scores(refs, preds):
#     idf = collections.defaultdict(lambda: 0)
#     n_refs = len(refs)
#     for r in refs:
#         grams = set(get_ngrams(r))
#         for g in grams:
#             idf[g] += 1
#
#     for g in idf:
#         idf[g] = math.log((n_refs + 1) / (idf[g] + 1)) + 1
#
#     scores = []
#     for r, p in zip(refs, preds):
#         r_grams = get_ngrams(r)
#         p_grams = get_ngrams(p)
#
#         r_cnt = collections.Counter(r_grams)
#         p_cnt = collections.Counter(p_grams)
#
#         vec_r = []
#         vec_p = []
#         for g in set(list(r_cnt.keys()) + list(p_cnt.keys())):
#             w = idf.get(g, 0)
#             vec_r.append(r_cnt.get(g, 0) * w)
#             vec_p.append(p_cnt.get(g, 0) * w)
#
#         norm_r = np.linalg.norm(vec_r)
#         norm_p = np.linalg.norm(vec_p)
#         if norm_r == 0 or norm_p == 0:
#             scores.append(0.0)
#         else:
#             scores.append(np.dot(vec_r, vec_p) / (norm_r * norm_p))
#     return np.mean(scores)
#
# # 计算三组官方 CIDEr
# cider_img_ref = compute_cider_scores(all_captions, all_git_generated_captions)
# cider_brain_ref = compute_cider_scores(all_captions, all_predcaptions)
# cider_brain_img = compute_cider_scores(all_git_generated_captions, all_predcaptions)
# relative_brain_image_cider = cider_brain_img / cider_img_ref


# -------------------------- Sentence Transformer --------------------------
print("计算 Sentence Similarity...")
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
with torch.no_grad():
    embedding_brain = sentence_model.encode(all_predcaptions, convert_to_tensor=True)
    embedding_captions = sentence_model.encode(all_captions, convert_to_tensor=True)
    embedding_images = sentence_model.encode(all_git_generated_captions, convert_to_tensor=True)

    ss_sim_brain_img = util.pytorch_cos_sim(embedding_brain, embedding_images).cpu()
    ss_sim_brain_cap = util.pytorch_cos_sim(embedding_brain, embedding_captions).cpu()
    ss_sim_img_cap = util.pytorch_cos_sim(embedding_images, embedding_captions).cpu()
relative_brain_image_ss = ss_sim_brain_img.diag().mean() / ss_sim_img_cap.diag().mean()

# -------------------------- SPICE --------------------------
print("计算 SPICE...")
spice_img_ref = spice.compute(predictions=all_git_generated_captions, references=all_captions)
spice_brain_ref = spice.compute(predictions=all_predcaptions, references=all_captions)
spice_brain_img = spice.compute(predictions=all_predcaptions, references=all_git_generated_captions)
relative_brain_image_spice = spice_brain_img["spice"] / spice_img_ref["spice"]

# -------------------------- CLIP-B --------------------------
print("计算 CLIP-B...")
model_clip_b = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
tokenizer_b = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
with torch.no_grad():
    embedding_brain_b = model_clip_b.get_text_features(**tokenizer_b(list(all_predcaptions), return_tensors="pt", padding=True))
    embedding_captions_b = model_clip_b.get_text_features(**tokenizer_b(list(all_captions), return_tensors="pt", padding=True))
    embedding_images_b = model_clip_b.get_text_features(**tokenizer_b(all_git_generated_captions, return_tensors="pt", padding=True))

clip_B_sim_brain_img = util.pytorch_cos_sim(embedding_brain_b, embedding_images_b).cpu()
clip_B_sim_brain_cap = util.pytorch_cos_sim(embedding_brain_b, embedding_captions_b).cpu()
clip_B_sim_img_cap = util.pytorch_cos_sim(embedding_images_b, embedding_captions_b).cpu()
relative_brain_image_clip_B = clip_B_sim_brain_img.diag().mean() / clip_B_sim_img_cap.diag().mean()

# -------------------------- CLIP-L --------------------------
print("计算 CLIP-L...")
model_clip_l = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
tokenizer_l = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
with torch.no_grad():
    embedding_brain_l = model_clip_l.get_text_features(**tokenizer_l(list(all_predcaptions), return_tensors="pt", padding=True))
    embedding_captions_l = model_clip_l.get_text_features(**tokenizer_l(list(all_captions), return_tensors="pt", padding=True))
    embedding_images_l = model_clip_l.get_text_features(**tokenizer_l(all_git_generated_captions, return_tensors="pt", padding=True))

clip_L_sim_brain_img = util.pytorch_cos_sim(embedding_brain_l, embedding_images_l).cpu()
clip_L_sim_brain_cap = util.pytorch_cos_sim(embedding_brain_l, embedding_captions_l).cpu()
clip_L_sim_img_cap = util.pytorch_cos_sim(embedding_images_l, embedding_captions_l).cpu()
relative_brain_image_clip_L = clip_L_sim_brain_img.diag().mean() / clip_L_sim_img_cap.diag().mean()

# ==============================================================================
# 3. 打印所有结果
# ==============================================================================
print("\n" + "="*80)
print("所有文本评估指标结果")
print("="*80)

print(f"METEOR:  {meteor_img_ref['meteor']:.4f} | {meteor_brain_ref['meteor']:.4f} | {meteor_brain_img['meteor']:.4f} | Relative: {relative_brain_image_meteor:.4f}")
print(f"ROUGE-L: {rouge_img_ref['rougeL']:.4f} | {rouge_brain_ref['rougeL']:.4f} | {rouge_brain_img['rougeL']:.4f} | Relative: {relative_brain_image_rougeL:.4f}")
print(f"ROUGE-1: {rouge_img_ref['rouge1']:.4f} | {rouge_brain_ref['rouge1']:.4f} | {rouge_brain_img['rouge1']:.4f} | Relative: {relative_brain_image_rouge1:.4f}")
print(f"CIDEr:   {cider_img_ref['cider']:.4f} | {cider_brain_ref['cider']:.4f} | {cider_brain_img['cider']:.4f} | Relative: {relative_brain_image_cider:.4f}")
print(f"Sentence:{ss_sim_img_cap.diag().mean():.4f} | {ss_sim_brain_cap.diag().mean():.4f} | {ss_sim_brain_img.diag().mean():.4f} | Relative: {relative_brain_image_ss.mean():.4f}")
print(f"SPICE:   {spice_img_ref['spice']:.4f} | {spice_brain_ref['spice']:.4f} | {spice_brain_img['spice']:.4f} | Relative: {relative_brain_image_spice:.4f}")
print(f"CLIP-B:  {clip_B_sim_img_cap.diag().mean():.4f} | {clip_B_sim_brain_cap.diag().mean():.4f} | {clip_B_sim_brain_img.diag().mean():.4f} | Relative: {relative_brain_image_clip_B.mean():.4f}")
print(f"CLIP-L:  {clip_L_sim_img_cap.diag().mean():.4f} | {clip_L_sim_brain_cap.diag().mean():.4f} | {clip_L_sim_brain_img.diag().mean():.4f} | Relative: {relative_brain_image_clip_L.mean():.4f}")

# ==============================================================================
# 4. 保存到 CSV
# ==============================================================================
caption_metrics = {
    "Meteor_img_ref": meteor_img_ref['meteor'],
    "Meteor_brain_ref": meteor_brain_ref['meteor'],
    "Meteor_brain_img": meteor_brain_img['meteor'],
    "Meteor_relative": relative_brain_image_meteor,

    "RougeL_img_ref": rouge_img_ref['rougeL'],
    "RougeL_brain_ref": rouge_brain_ref['rougeL'],
    "RougeL_brain_img": rouge_brain_img['rougeL'],
    "RougeL_relative": relative_brain_image_rougeL,

    "Rouge1_img_ref": rouge_img_ref['rouge1'],
    "Rouge1_brain_ref": rouge_brain_ref['rouge1'],
    "Rouge1_brain_img": rouge_brain_img['rouge1'],
    "Rouge1_relative": relative_brain_image_rouge1,

    "CIDEr_img_ref": cider_img_ref['cider'],
    "CIDEr_brain_ref": cider_brain_ref['cider'],
    "CIDEr_brain_img": cider_brain_img['cider'],
    "CIDEr_relative": relative_brain_image_cider,

    "Sentence_img_ref": ss_sim_img_cap.diag().mean().item(),
    "Sentence_brain_ref": ss_sim_brain_cap.diag().mean().item(),
    "Sentence_brain_img": ss_sim_brain_img.diag().mean().item(),
    "Sentence_relative": relative_brain_image_ss.mean().item(),

    "SPICE_img_ref": spice_img_ref['spice'],
    "SPICE_brain_ref": spice_brain_ref['spice'],
    "SPICE_brain_img": spice_brain_img['spice'],
    "SPICE_relative": relative_brain_image_spice,

    "CLIP-B_img_ref": clip_B_sim_img_cap.diag().mean().item(),
    "CLIP-B_brain_ref": clip_B_sim_brain_cap.diag().mean().item(),
    "CLIP-B_brain_img": clip_B_sim_brain_img.diag().mean().item(),
    "CLIP-B_relative": relative_brain_image_clip_B.mean().item(),

    "CLIP-L_img_ref": clip_L_sim_img_cap.diag().mean().item(),
    "CLIP-L_brain_ref": clip_L_sim_brain_cap.diag().mean().item(),
    "CLIP-L_brain_img": clip_L_sim_brain_img.diag().mean().item(),
    "CLIP-L_relative": relative_brain_image_clip_L.mean().item(),
}


df = pd.DataFrame.from_dict(caption_metrics, orient='index', columns=["Value"])
csv_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/text_metrics.csv'
df.to_csv(csv_path, sep='\t')
print(f"\n指标已保存至: {csv_path}")

# ==============================================================================
# 5. 文本定性对比展示
# ==============================================================================
print("\n生成文本定性对比图...")
plt.rcParams["savefig.bbox"] = 'tight'
np.random.seed(0)

# 只保留 4 个示例
ind = [10, 20, 30, 40]
show_num = min(4, len(ind))
ind = ind[:show_num]

# 只保留 2 列：GT + Brain-Predicted
fig, axes = plt.subplots(nrows=len(ind), ncols=2, figsize=(24, len(ind)*1.6))

# 列标题
axes[0,0].set_title("Ground Truth Caption", fontsize=18, weight='bold', color='blue')
axes[0,1].set_title("Brain-Predicted Caption", fontsize=18, weight='bold', color='green')

for row_idx, sample_idx in enumerate(ind):
    gt = str(all_git_generated_captions[sample_idx]).strip()
    brain_pred = str(all_predcaptions[sample_idx]).strip()

    # 大字体、紧凑行间距
    axes[row_idx, 0].text(0.5, 0.5, gt,
                          ha='center', va='center', wrap=True, fontsize=20, linespacing=0.85)
    axes[row_idx, 1].text(0.5, 0.5, brain_pred,
                          ha='center', va='center', wrap=True, fontsize=20, linespacing=0.85)

    for ax in axes[row_idx]:
        ax.axis('off')

save_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/text_comparison.png"
# 极致紧凑
plt.subplots_adjust(wspace=0.1, hspace=0.01)
plt.savefig(save_path, dpi=350, bbox_inches='tight')
plt.close()
print(f"文本对比图已保存至: {save_path}")
print("\n 所有评估与可视化完成！")