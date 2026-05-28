import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from tqdm import tqdm  # 显示进度条，更直观
from models import Clipper


# 路径配置
image_pt_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/gt_test_images.pt"
save_clip_feature_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_image_features.pt"  # 保存提取好的特征

# 模型配置
device = "cuda"
seed = 42

# 加载模型
print('Creating Clipper...')
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)

# 加载所有图片（只加载到CPU，不进GPU）
print(f"Loading images from: {image_pt_path}")
all_images = torch.load(image_pt_path)  # shape: [N, C, H, W]
num_images = len(all_images)
print(f"Total images: {num_images}")

# 逐张提取特征（不会爆内存！）
clip_features = []
with torch.no_grad():  # 禁用梯度，省显存
    for img in tqdm(all_images, desc="Extracting CLIP features"):
        # 增加 batch 维度 [C, H, W] -> [1, C, H, W]
        img = img.unsqueeze(0).to(device)
        # 提取特征
        feat = clip_extractor.embed_image(img).float().cpu()  # 提取后立刻放回CPU
        clip_features.append(feat)

# 拼接所有特征
clip_features = torch.cat(clip_features, dim=0)
print(f"Final CLIP features shape: {clip_features.shape}")

# 保存
torch.save(clip_features, save_clip_feature_path)
print(f"CLIP features saved to: {save_clip_feature_path}")