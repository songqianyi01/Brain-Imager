#!/usr/bin/env python
# coding: utf-8
import os
import sys
import torch
import numpy as np
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os

# 强制映射双卡环境
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
# 注意：这里我们不再限制只可见单卡，而是通过 PyTorch 指定 cuda:0 和 cuda:1
os.environ['CUDA_VISIBLE_DEVICES'] = '3,2'  # 请根据你服务器真实的卡号填写，比如使用1卡和2卡

sys.path.append('/media/data/songzengyu/Brain-Imager/code/run_server_github/taming-transformers')

import utils
from models import Clipper, Voxel2StableDiffusionModel
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
from transformers import AutoProcessor, AutoTokenizer, T5ForConditionalGeneration
from modeling_git import GitForCausalLMClipEmb
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
import gradio as gr

# ===================== 1. 设备与路径配置 =====================
device_img = torch.device('cuda:0') # 图像解码卡
device_txt = torch.device('cuda:1') # 文本解码卡
seed = 42
utils.seed_everything(seed=seed)

FEATURE_DIR = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features"
VOXEL_PT_PATH = "/media/data/songzengyu/Brain-Imager/code/run_server_github/voxels.pt"

# 模型权重路径
nature_ckpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/ZN.pth'
panoramic_ckpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/ZP.pth'
vd_cache_dir = '/media/data/songzengyu/Brain-Imager/code/run_server_github/versatile_diffusion/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'

mlp_ckpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/text_generate_mlp.pth'
stable_diffusion_config_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/v1-inference.yaml'
stable_diffusion_skpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/sd-v1-4.ckpt'
gt_images_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/gt_test_images.pt"
gt_texts_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/gt_captions.pt"


# ===================== 2. 全局预加载数据与模型 =====================
print("==========  正在初始化答辩演示系统 (预加载中) ==========")

# 加载预计算特征与脑信号
img_embeds = torch.load(os.path.join(FEATURE_DIR, "img_prior.pt"))
proj_img_embeds = torch.load(os.path.join(FEATURE_DIR, "img_proj.pt"))
txt_embeds = torch.load(os.path.join(FEATURE_DIR, "txt_prior.pt"))
proj_txt_embeds = torch.load(os.path.join(FEATURE_DIR, "txt_proj.pt"))
fmri_all = torch.load(VOXEL_PT_PATH)
gt_images = torch.load(gt_images_path, map_location='cpu')
gt_texts = torch.load(gt_texts_path, map_location='cpu')

# ----------- [卡 1] 加载 图像解码模型 -----------
print(" [Card 1] Loading Image Decoding Models...")
voxel2sd_nature = Voxel2StableDiffusionModel(in_dim=15724)
ckpt_n = torch.load(nature_ckpt_path, map_location=device_img)
voxel2sd_nature.load_state_dict(ckpt_n['model_state_dict'], strict=False)
voxel2sd_nature.to(device_img).eval().requires_grad_(False)

voxel2sd_pano = Voxel2StableDiffusionModel(in_dim=15724)
ckpt_p = torch.load(panoramic_ckpt_path, map_location=device_img)
voxel2sd_pano.load_state_dict(ckpt_p['model_state_dict'], strict=False)
voxel2sd_pano.to(device_img).eval().requires_grad_(False)

vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device_img).to(torch.float16)
vd_pipe.image_unet.eval().requires_grad_(False)
vd_pipe.vae.eval().requires_grad_(False)
noise_scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")

# 配置双 UNet
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = 1.0  # 保持与你代码一致的配置
        module.condition_lengths = [257, 77]
        module.transformer_index_for_condition = [0, 1]
unet1 = vd_pipe.image_unet
unet2 = vd_pipe.image_unet
vae = vd_pipe.vae
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device_img)
to_pil = transforms.ToPILImage()

# ----------- [卡 2] 加载 文本解码模型 -----------
print(" [Card 2] Loading Text Decoding Models...")


class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 1024))

    def forward(self, x): return self.fc(x)


git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
git_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco", torch_dtype=torch.float16).to(
    device_txt).eval()
git_model.requires_grad_(False)

mlp_model = MappingNetwork().to(device_txt).half()
mlp_ckpt = torch.load(mlp_ckpt_path, map_location=device_txt)
mlp_model.load_state_dict(mlp_ckpt['model_state_dict'])
mlp_model.eval().requires_grad_(False)

t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
t5_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", torch_dtype=torch.float16).to(
    device_txt).eval()
t5_model.requires_grad_(False)

# 加载 SD Text Encoder
sd_config = OmegaConf.load(stable_diffusion_config_path)
pl_sd = torch.load(stable_diffusion_skpt_path, map_location=device_txt) # 确保load时就映射到卡1
LDM_model = instantiate_from_config(sd_config.model)
LDM_model.load_state_dict(pl_sd["state_dict"], strict=False)

# ─── 核心修复：强制将所有子模块以及文本编码器完全移至 device_txt ───
LDM_model = LDM_model.to(device_txt)
if hasattr(LDM_model, "cond_stage_model"):
    LDM_model.cond_stage_model = LDM_model.cond_stage_model.to(device_txt)
    # 如果内部包裹了 transformer，再次确保转移
    if hasattr(LDM_model.cond_stage_model, "transformer"):
        LDM_model.cond_stage_model.transformer = LDM_model.cond_stage_model.transformer.to(device_txt)

LDM_model = LDM_model.half().eval()
LDM_model.model.diffusion_model = None  # 卸载 U-Net 省显存

print(" 所有模型预加载完成！系统就绪。")


# ===================== 3. 核心算法工具函数 =====================
def batchwise_cosine_similarity(Z, B):
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    return ((Z @ B) / (Z_norm @ B_norm)).T


def retrieval_text_embedding(text_embedding, proj_embedding):
    v2c_reference_out = nn.functional.normalize(proj_embedding.view(len(proj_embedding), -1), dim=-1)
    sims = []
    for im in range(16):
        currecon = text_embedding[im].unsqueeze(0)
        currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim=-1)
        cursim = batchwise_cosine_similarity(v2c_reference_out, currecon)
        sims.append(cursim.item())
    return int(np.nanargmax(sims))


def robust_inverse_scale_manual(scaled_data, median, iqr):
    return scaled_data * iqr + median


def gaussian_pyramid(img, levels):
    pyramid = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def laplacian_pyramid(gaussian_pyr):
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(laplacian)
    laplacian_pyr.append(gaussian_pyr[-1])
    return laplacian_pyr


def reconstruct_from_laplacian_pyramid(laplacian_pyr):
    img = laplacian_pyr[-1]
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, laplacian_pyr[i])
    return img


def mix_blur_vector(nature_vector, panoramic_vector):
    clear_images = nature_vector.astype('float64')
    struct_images = panoramic_vector.astype('float64')
    fused_images = np.zeros_like(clear_images)
    levels = 6
    lameta = [1, 1, 1, 1, 0.6, 0.8, 0.8]
    for c in range(3):
        img_clear = clear_images[0, c, :, :]
        img_struct = struct_images[0, c, :, :]
        gaussian_pyr_clear = gaussian_pyramid(img_clear, levels)
        gaussian_pyr_struct = gaussian_pyramid(img_struct, levels)
        lap_pyr_clear = laplacian_pyramid(gaussian_pyr_clear)
        lap_pyr_struct = laplacian_pyramid(gaussian_pyr_struct)
        fused_pyr = []
        for level in range(levels + 1):
            fused_layer = cv2.addWeighted(lap_pyr_clear[level], lameta[level], lap_pyr_struct[level],
                                          (1 - lameta[level]), 0)
            fused_pyr.append(fused_layer)
        fused_images[0, c, :, :] = reconstruct_from_laplacian_pyramid(fused_pyr)
    return fused_images


# ===================== 4. UI 触发业务逻辑 =====================
def plot_fmri_signal(idx):
    """根据索引绘制脑电/fMRI信号平铺图"""
    signal = fmri_all[idx].squeeze(0).cpu().numpy()
    plt.figure(figsize=(10, 3))
    plt.plot(signal, color='#1f77b4', linewidth=0.6)
    plt.xlabel("Voxel Index", fontsize=9)
    plt.ylabel("Signal Intensity", fontsize=9)
    plt.title(f"fMRI Voxel Flatten Signal (Sample Index: {idx})", fontsize=11, fontweight='bold')
    plt.tight_layout()

    # 转为 PIL Image 返回给 UI
    canvas = plt.get_current_fig_manager().canvas
    plt.savefig('tmp_fmri.jpg', dpi=200, bbox_inches='tight')
    plt.close()
    return Image.open('tmp_fmri.jpg')


def ui_decode_image(idx):
    """点击图像解码按钮触发"""
    idx = int(idx)
    fmri_img = plot_fmri_signal(idx)
    gt_img_pil = utils.torch_to_Image(gt_images[idx])
    # 切换到卡1进行计算
    with torch.no_grad():
        image_embedding = img_embeds[idx].to(device_img)
        proj_image_embedding = proj_img_embeds[idx:idx + 1].to(device_img)
        text_embedding = txt_embeds[idx].to(device_img)
        proj_text_embedding = proj_txt_embeds[idx:idx + 1].to(device_img)

        best_text_pick = retrieval_text_embedding(text_embedding, proj_text_embedding)
        best_text_embedding = text_embedding[best_text_pick]

        fmri_sample = fmri_all[idx].squeeze(0).to(device_img)

        # 预测模糊向量
        ae_preds_n = voxel2sd_nature(fmri_sample.float().unsqueeze(0))
        nature_vector = (vae.decode(ae_preds_n.half() / 0.18215).sample / 2 + 0.5).cpu().numpy()

        ae_preds_p = voxel2sd_pano(fmri_sample.float().unsqueeze(0))
        panoramic_vector = (vae.decode(ae_preds_p.half() / 0.18215).sample / 2 + 0.5).cpu().numpy()

        # 金字塔融合
        blurry_vector = mix_blur_vector(nature_vector, panoramic_vector)
        blurry_recons = torch.tensor(blurry_vector).to(device_img)

        # 终极重建
        _, brain_recons, laion_best_picks, _ = utils.reconstruction_integrity_noise_assign(
            clip_extractor, unet1, unet2, vae, noise_scheduler,
            voxel2clip_cls=None, diffusion_priors=None,
            text_token=best_text_embedding.unsqueeze(0),
            img_lowlevel=blurry_recons,
            num_inference_steps=20, n_samples_save=1, recons_per_sample=16,
            guidance_scale=3.5, img2img_strength=0.85, timesteps_prior=None,
            seed=seed, retrieve=False, plotting=False, img_variations=False, verbose=False,
            input_embedding=image_embedding, proj_embedding=proj_image_embedding,
        )
        brain_recons = brain_recons[:, laion_best_picks.astype(np.int8)].squeeze(0).squeeze(0)
        decoded_image = to_pil(brain_recons)

    return fmri_img, gt_img_pil, decoded_image


def ui_decode_text(idx):
    """点击文本解码按钮触发"""
    idx = int(idx)
    fmri_img = plot_fmri_signal(idx)
    gt_text_str = gt_texts[idx]
    # ─── 核心修复：显式指定 torch.cuda.device ───
    with torch.cuda.device(device_txt):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            image_embedding = img_embeds[idx].to(device_txt)
            proj_text_embedding = proj_txt_embeds[idx:idx + 1].to(device_txt)

            # GIT 模型生成候选文本
            git_feature = mlp_model(image_embedding)
            git_feature_process = robust_inverse_scale_manual(git_feature, 0.155, 1.25)

            generated_ids = git_model.generate(
                pixel_values=git_feature_process,
                num_beams=10, diversity_penalty=1.0, num_beam_groups=5, max_length=25
            )
            text_git_captioning = git_processor.batch_decode(generated_ids, skip_special_tokens=True)

            # SD 文本特征检索最优
            text_git_embedding = LDM_model.get_learned_conditioning(text_git_captioning).cpu().detach().squeeze()

            # 确保转换成 cpu 张量进行检索，防止两卡冲突
            best_pick = retrieval_text_embedding(torch.tensor(text_git_embedding).cpu(), proj_text_embedding.cpu())
            best_git_text = text_git_captioning[best_pick]

            # T5 润色重写
            input_text = f"Rewrite the following image annotation to make it meaningful and coherent: {best_git_text}"
            inputs = t5_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=40).to(device_txt)
            outs = t5_model.generate(input_ids=inputs["input_ids"], max_length=30, num_beams=10,
                                     attention_mask=inputs["attention_mask"])
            final_caption = t5_tokenizer.decode(outs[0], skip_special_tokens=True)

    return fmri_img, gt_text_str, final_caption


# ===================== 5. Gradio 界面设计 =====================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="毕业答辩：脑信号多模态解码演示系统") as demo:
    gr.Markdown(
        """
        # 毕业答辩演示：基于 fMRI 脑信号的多模态解码系统
        **课题名称**：面向多模态数据的人脑视觉信号重建 &nbsp;&nbsp;|&nbsp;&nbsp; **核心框架**: Brain-Imager
        """
    )
    gr.HTML("<hr style='border:1px solid #dfdfdf;'>")

    with gr.Row():
        with gr.Column(scale=1):
            input_index = gr.Number(
                value=257,
                label="输入测试集 fMRI 样本索引 (范围: 0 - 981)",
                precision=0,
                interactive=True
            )

            with gr.Group():
                gr.Markdown("### 模态解码控制台")
                btn_image = gr.Button("触发：视觉图像解码", variant="primary")
                btn_text = gr.Button("触发：语义脑字幕生成", variant="secondary")

        with gr.Column(scale=2):
            out_fmri = gr.Image(label="fMRI Flatten 信号流", type="pil")

    gr.HTML("<br>")

    # 核心对照看板区
    with gr.Tabs():
        # 图像对照组
        with gr.TabItem("视觉解码结果"):
            with gr.Row():
                with gr.Column():
                    out_gt_image = gr.Image(label="[对照组] Ground Truth (受试者看到的真实图像)", type="pil", width=380)
                with gr.Column():
                    out_image = gr.Image(label="[生成组] Brain-Imager (视觉解码图像)", type="pil", width=380)

        # 文本对照组
        with gr.TabItem("脑字幕生成结果"):
            with gr.Row():
                with gr.Column():
                    out_gt_text = gr.Textbox(label="[对照组] Ground Truth (真实测试集标注描述)", lines=4,
                                             interactive=False)
                with gr.Column():
                    out_text = gr.Textbox(label="[生成组] Brain-Imager (脑字幕生成文本)", lines=4,
                                          interactive=False)

    # ===================== 6. 按钮事件显式绑定 =====================
    # 注意：这里的 outputs 列表分别增加了新接收的对照组组件
    btn_image.click(
        fn=ui_decode_image,
        inputs=[input_index],
        outputs=[out_fmri, out_gt_image, out_image]
    )

    btn_text.click(
        fn=ui_decode_text,
        inputs=[input_index],
        outputs=[out_fmri, out_gt_text, out_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

