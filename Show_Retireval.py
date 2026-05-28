#!/usr/bin/env python
# coding: utf-8
import os

# 指定显卡，建议在命令行或代码开头执行
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import numpy as np
import gradio as gr
import utils
import torch.nn as nn

# ===================== 1. 全局配置与数据加载 =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("正在加载预计算特征 ...")

# --- 图像检索相关路径 ---
feat_img_clip_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_image_features.pt"
feat_fmri_proj_img_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/img_proj.pt"
images_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/gt_test_images.pt"

# --- 文本检索相关路径 ---
feat_text_clip_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_text_features.pt"
feat_fmri_proj_txt_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_proj.pt"
texts_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/gt_captions.pt"

# --- 图像数据加载 ---
print("  > 加载图像特征...")
fmri_proj_img = torch.load(feat_fmri_proj_img_path, map_location=device)
img_clip = torch.load(feat_img_clip_path, map_location=device)
images = torch.load(images_path, map_location='cpu')
# 归一化图像 CLIP 特征
img_clip = nn.functional.normalize(img_clip.reshape(len(img_clip), -1), dim=-1)

# --- 文本数据加载 ---
print("  > 加载文本特征...")
fmri_proj_txt = torch.load(feat_fmri_proj_txt_path, map_location=device)
text_clip = torch.load(feat_text_clip_path, map_location=device)
texts = torch.load(texts_path, map_location='cpu')
# 归一化文本 CLIP 特征
text_clip = nn.functional.normalize(text_clip.reshape(len(text_clip), -1), dim=-1)

print("所有数据加载完成！正在启动 UI...")


# ===================== 2. 核心检索逻辑 =====================

def retrieve_image(sample_idx):
    """图像检索逻辑"""
    sample_idx = int(sample_idx)
    if sample_idx >= len(fmri_proj_img):
        return None, None

    query = fmri_proj_img[sample_idx:sample_idx + 1]
    query = query.reshape(1, -1)
    query = nn.functional.normalize(query, dim=-1)

    # 计算余弦相似度并获取 Top-5
    sim = utils.batchwise_cosine_similarity(query, img_clip)
    sim = sim.cpu().numpy().squeeze()
    topk_idx = np.flip(np.argsort(sim))[:5]
    scores = sim[topk_idx]

    # 获取 Ground Truth 图片
    gt_image = utils.torch_to_Image(images[sample_idx])

    # 构造 Gradio Gallery 需要的数据格式: [(PIL_Image, "标签"), ...]
    top5_gallery = []
    for rank, (idx, score) in enumerate(zip(topk_idx, scores)):
        img_pil = utils.torch_to_Image(images[idx])
        caption = f"Top {rank + 1} (相似度: {score:.4f})"
        top5_gallery.append((img_pil, caption))

    return gt_image, top5_gallery


def retrieve_text(sample_idx):
    """文本检索逻辑"""
    sample_idx = int(sample_idx)
    if sample_idx >= len(fmri_proj_txt):
        return "警告: 索引超出范围", "无结果"

    query = fmri_proj_txt[sample_idx:sample_idx + 1]
    query = query.reshape(1, -1)
    query = nn.functional.normalize(query, dim=-1)

    # 计算余弦相似度并获取 Top-5
    sim = utils.batchwise_cosine_similarity(query, text_clip)
    sim = sim.cpu().numpy().squeeze()
    topk_idx = np.flip(np.argsort(sim))[:5]
    scores = sim[topk_idx]

    # 获取 Ground Truth 文本
    gt_text = texts[sample_idx]

    # 构造 Markdown 格式的排版结果，视觉上清晰美观
    top5_md = ""
    for rank, (idx, score) in enumerate(zip(topk_idx, scores)):
        retrieved_text = texts[idx]
        top5_md += f"**Top {rank + 1}** (相似度: `{score:.4f}`)\n"
        top5_md += f"> {retrieved_text}\n\n"

    return gt_text, top5_md


# ===================== 3. Gradio UI 界面构建 =====================

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"),title="毕业答辩：脑信号多模态检索演示系统") as demo:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <h1>单 fMRI 信号跨模态检索系统演示</h1>
            <p>输入测试集中的 fMRI 样本索引，系统将基于脑信号特征，分别在图像库和文本库中检索出最匹配的 Top-5 结果。</p>
        </div>
        """
    )

    # ---------- 全局控制区 ----------
    with gr.Row():
        with gr.Column(scale=1):
            input_idx = gr.Number(label="输入 fMRI 样本索引 (如: 100)", value=100, precision=0)

    # ---------- 分页检索区 ----------
    with gr.Tabs():
        # --- 选项卡 1: 图像检索 ---
        with gr.TabItem("图像检索"):
            btn_image = gr.Button("开始图像检索", variant="primary")
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1):
                    gt_img_output = gr.Image(label="Ground Truth (受试者看到的真实图像)", type="pil")
                with gr.Column(scale=4):
                    top5_img_output = gr.Gallery(
                        label="Top-5 检索结果",
                        columns=5,
                        height="auto",
                        show_label=True,
                        object_fit="contain"
                    )

            btn_image.click(
                fn=retrieve_image,
                inputs=input_idx,
                outputs=[gt_img_output, top5_img_output]
            )

        # --- 选项卡 2: 文本检索 ---
        with gr.TabItem("文本检索"):
            btn_text = gr.Button("开始文本检索", variant="primary")
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(scale=1):
                    gt_text_output = gr.Textbox(
                        label="Ground Truth (真实图像对应的描述)",
                        lines=5,
                        interactive=False
                    )
                with gr.Column(scale=4):
                    # 使用 Markdown 渲染带 blockquote 的优美排版
                    top5_text_output = gr.Markdown("### Top-5 检索结果将显示在这里")

            btn_text.click(
                fn=retrieve_text,
                inputs=input_idx,
                outputs=[gt_text_output, top5_text_output]
            )

if __name__ == '__main__':
    # 设置 server_name="0.0.0.0" 允许局域网内其他设备访问
    # 答辩时，你可以在讲台电脑上输入服务器 IP:7860 直接打开该界面
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)