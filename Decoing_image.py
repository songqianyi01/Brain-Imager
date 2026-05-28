#!/usr/bin/env python
# coding: utf-8
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import numpy as np
import torch.nn as nn
import cv2
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

# 项目自定义模块
import utils
from models import Clipper, Voxel2StableDiffusionModel
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel

# ===================== 设备与基础配置 =====================
device0 = torch.device('cuda')
seed = 42
utils.seed_everything(seed=seed)

# 显存优化
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# ===================== 核心路径配置 =====================
# 预计算特征路径（你已保存的4个embedding）
FEATURE_DIR = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features"
# 模型权重路径
nature_ckpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/ZN.pth'
panoramic_ckpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/ZP.pth'
vd_cache_dir = '/media/data/songzengyu/Brain-Imager/code/run_server_github/versatile_diffusion/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
# 输出路径
SAVE_IMAGE_DIR = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_images"
os.makedirs(SAVE_IMAGE_DIR, exist_ok=True)


# ===================== 工具函数 =====================
def batchwise_cosine_similarity(Z, B):
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity


def retrieval_text_embedding(text_embedding, proj_embedding):
    v2c_reference_out = nn.functional.normalize(proj_embedding.view(len(proj_embedding), -1), dim=-1)
    sims = []
    for im in range(16):
        currecon = text_embedding[im].unsqueeze(0)
        currecon = nn.functional.normalize(currecon.view(len(currecon), -1), dim=-1)
        cursim = batchwise_cosine_similarity(v2c_reference_out, currecon)
        sims.append(cursim.item())
    best_pick = int(np.nanargmax(sims))
    return best_pick


# 金字塔融合函数
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

        fused_img = reconstruct_from_laplacian_pyramid(fused_pyr)
        fused_images[0, c, :, :] = fused_img
    return fused_images


# ===================== 一次性加载所有模型（核心优化） =====================
print("========== 一次性加载所有模型 ==========")

# 1. 加载自然/全景向量生成模型
# 自然模型
voxel2sd_nature = Voxel2StableDiffusionModel(in_dim=15724)
checkpoint = torch.load(nature_ckpt_path, map_location=device0)
voxel2sd_nature.load_state_dict(checkpoint['model_state_dict'], strict=False)
voxel2sd_nature.to(device0).eval().requires_grad_(False)

# 全景模型
voxel2sd_pano = Voxel2StableDiffusionModel(in_dim=15724)
checkpoint = torch.load(panoramic_ckpt_path, map_location=device0)
voxel2sd_pano.load_state_dict(checkpoint['model_state_dict'], strict=False)
voxel2sd_pano.to(device0).eval().requires_grad_(False)

# 2. 加载VersatileDiffusion管道（双UNet）
vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device0).to(torch.float16)
vd_pipe.image_unet.eval().requires_grad_(False)
vd_pipe.vae.eval().requires_grad_(False)
noise_scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")

# 3. 初始化双UNet配置（对应单个代码的unet1/unet2）
# unet1: 仅图像引导
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = 0.0
        module.condition_lengths = [257, 77]
        module.transformer_index_for_condition = [0, 1]
unet1 = vd_pipe.image_unet

# unet2: 仅文本引导
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = 1.0
        module.condition_lengths = [257, 77]
        module.transformer_index_for_condition = [0, 1]
unet2 = vd_pipe.image_unet

vae = vd_pipe.vae

# 4. 加载CLIP提取器
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device0)

# 5. 图像保存工具
to_pil = transforms.ToPILImage()

# 清理显存
del checkpoint
torch.cuda.empty_cache()
print("所有模型加载完成，开始批量解码...")

# ===================== 加载预计算的4个Embedding =====================
print("加载预计算特征...")
img_embeds = torch.load(os.path.join(FEATURE_DIR, "img_prior.pt"))  # 图像embedding
proj_img_embeds = torch.load(os.path.join(FEATURE_DIR, "img_proj.pt"))  # 图像投影embedding
txt_embeds = torch.load(os.path.join(FEATURE_DIR, "txt_prior.pt"))  # 文本embedding
proj_txt_embeds = torch.load(os.path.join(FEATURE_DIR, "txt_proj.pt"))  # 文本投影embedding
fmri_all = torch.load("/media/data/songzengyu/Brain-Imager/code/run_server_github/voxels.pt")
print(f"特征加载完成，共 {len(img_embeds)} 个样本")

# ===================== 批量图像解码主函数 =====================
if __name__ == "__main__":
    num_samples = 982
    all_images = []  # 用于存储所有生成的图像张量

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="批量解码图像"):
            # ============= 1. 获取当前样本的所有embedding =============
            image_embedding = img_embeds[idx].to(device0)  # [16,257,768]
            proj_image_embedding = proj_img_embeds[idx:idx+1].to(device0)
            text_embedding = txt_embeds[idx].to(device0)  # [16,77,768]
            proj_text_embedding = proj_txt_embeds[idx:idx+1].to(device0)

            # ============= 2. 文本检索 =============
            best_text_pick = retrieval_text_embedding(text_embedding, proj_text_embedding)
            best_text_embedding = text_embedding[best_text_pick]

            # ============= 3. 生成模糊向量（自然+全景融合） =============
            fmri_sample = fmri_all[idx].squeeze(0).to(device0)

            # 自然向量
            ae_preds = voxel2sd_nature(fmri_sample.float().unsqueeze(0))
            nature_vec = vae.decode(ae_preds.half() / 0.18215).sample / 2 + 0.5
            nature_vector = nature_vec.cpu().numpy()

            # 全景向量
            ae_preds = voxel2sd_pano(fmri_sample.float().unsqueeze(0))
            pano_vec = vae.decode(ae_preds.half() / 0.18215).sample / 2 + 0.5
            panoramic_vector = pano_vec.cpu().numpy()

            # 金字塔融合
            blurry_vector = mix_blur_vector(nature_vector, panoramic_vector)
            blurry_recons = torch.tensor(blurry_vector).to(device0)

            # ============= 4. 图像重建（核心） =============
            grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction_integrity_noise_assign(
                clip_extractor, unet1, unet2, vae, noise_scheduler,
                voxel2clip_cls=None,
                diffusion_priors=None,
                text_token=best_text_embedding.unsqueeze(0),
                img_lowlevel=blurry_recons,
                num_inference_steps=20,
                n_samples_save=1,
                recons_per_sample=16,
                guidance_scale=3.5,
                img2img_strength=0.85,
                timesteps_prior=None,
                seed=seed,
                retrieve=False,
                plotting=False,
                img_variations=False,
                verbose=False,
                input_embedding=image_embedding,
                proj_embedding=proj_image_embedding,
            )

            # ============= 5. 保存图像并收集到列表中 =============
            brain_recons = brain_recons[:, laion_best_picks.astype(np.int8)].squeeze(0).squeeze(0)
            # 保存单个JPG文件
            img_pil = to_pil(brain_recons)
            img_pil.save(os.path.join(SAVE_IMAGE_DIR, f"ours_decoding_images/{idx}.jpg"))
            # 将图像张量添加到列表中
            all_images.append(brain_recons.cpu())


    # 将所有图像张量保存为一个.pt文件
    all_images_tensor = torch.stack(all_images)
    torch.save(all_images_tensor, os.path.join(SAVE_IMAGE_DIR, "ours_decoding_images.pt"))
    print(f"\n982张图像解码完成！")
    print(f"单个图像保存路径：{SAVE_IMAGE_DIR}")
    print(f"批量图像数据保存路径：{os.path.join(SAVE_IMAGE_DIR, 'ours_decoding_images.pt')}")