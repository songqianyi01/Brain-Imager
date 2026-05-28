import os
import torch
import webdataset as wds
from tqdm import tqdm
from transformers import GitProcessor, GitForCausalLM
from PIL import Image
import utils
import numpy as np

# ===================== 基础配置 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
utils.seed_everything(42)

# ===================== NSD数据集加载 =====================
subj_id = "01"
batch_size = 1  # 生成文本用单张即可
num_workers = 1

# NSD数据路径（完全沿用你的路径）
train_url = f"{{/media/data/songzengyu/Brain-Imager/data/data_single/train/subj{subj_id}/train_subj{subj_id}_{{0..17}}.tar,/media/data/songzengyu/Brain-Imager/data/data_single/val/subj{subj_id}/val_subj{subj_id}_0.tar}}"
val_url = f"/media/data/songzengyu/Brain-Imager/data/data_single/test/subj{subj_id}/test_subj{subj_id}_{{0..1}}.tar"
meta_url = f"/media/data/songzengyu/Brain-Imager/data/data_single/metadata_subj{subj_id}.json"
num_val = 982

# 加载验证集 dataloader（和你原代码完全一致）
_, val_dl, _, _ = utils.get_dataloaders(
    batch_size, 'images',
    num_devices=1,
    num_workers=num_workers,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=0,
    num_val=num_val,
    val_batch_size=1,
    cache_dir='/media/data/songzengyu/Brain-Imager/data/data_single',
    seed=42,
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images", "coco"],
    local_rank=0,
    world_size=1,
)

# ===================== GIT 文本生成模型加载 =====================
# 使用微软官方 GIT 图像文本生成模型（最适合自然场景）
processor = GitProcessor.from_pretrained("microsoft/git-large-coco")
model = GitForCausalLM.from_pretrained("microsoft/git-large-coco")
model.eval()
model.to(device)

# ===================== 生成文本 + 逐行打印 + 保存TXT =====================
save_path = "nsd_test_git_captions.txt"
captions = []

print("\n开始生成图像文本，每生成一条自动打印：\n")
with torch.no_grad(), torch.cuda.amp.autocast():
    for idx, (voxel, image, coco) in enumerate(tqdm(val_dl, desc="生成文本中")):
        # 预处理图像
        image = image.to(device).float()
        pil_image = Image.fromarray(
            (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        )

        # GIT 模型生成文本
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        generated_ids = model.generate(
            **inputs,
            max_length=50,  # 文本长度
            num_beams=4,  # 束搜索，生成更准确
            repetition_penalty=1.2
        )
        pred_caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        # 逐行打印
        print(f"第 {idx + 1} 张图像文本：{pred_caption}")
        captions.append(pred_caption)

# 保存所有文本到 TXT 文件
with open(save_path, "w", encoding="utf-8") as f:
    for i, cap in enumerate(captions):
        f.write(f"{cap}\n")

print(f"\n✅ 全部生成完成！共 {len(captions)} 条文本")
print(f"📄 文本已保存至：{os.path.abspath(save_path)}")