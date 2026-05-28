#!/usr/bin/env python
# coding: utf-8
import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer, T5ForConditionalGeneration
from modeling_git import GitForCausalLMClipEmb
import utils
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import sys

sys.path.append('/media/data/songzengyu/Brain-Imager/code/run_server_github/taming-transformers')

# ===================== 设备配置 =====================
device0 = torch.device('cuda')
seed = 42
utils.seed_everything(seed=seed)

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# ===================== 路径 =====================
FMRI_PT_PATH = "/media/data/songzengyu/Brain-Imager/code/run_server_github/voxels.pt"
SAVE_DIR = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text"
FEATURE_DIR = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features"

mlp_ckpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/text_generate_mlp.pth'
stable_diffusion_config_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/v1-inference.yaml'
stable_diffusion_skpt_path = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/sd-v1-4.ckpt'

# ===================== 加载预计算特征 =====================
print("正在加载图像/文本预计算特征...")
img_prior = torch.load(os.path.join(FEATURE_DIR, "img_prior.pt"))
txt_proj = torch.load(os.path.join(FEATURE_DIR, "txt_proj.pt"))
print(f"✅ 特征加载完成，共 {len(img_prior)} 个样本")

# ===================== 工具函数 =====================
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
    best_pick = int(np.nanargmax(sims))
    return best_pick

class MappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(768, 1024), nn.ReLU(), nn.Linear(1024, 1024))

    def forward(self, x):
        return self.fc(x)

def robust_inverse_scale_manual(scaled_data, median, iqr):
    return scaled_data * iqr + median

# ===================== 主程序 =====================
if __name__ == "__main__":
    print("Loading fMRI...")
    fmri = torch.load(FMRI_PT_PATH).squeeze(1)
    print(f"Shape: {fmri.shape}")

    # ===================== GIT =====================
    print("Loading GIT model...")
    git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    git_model = GitForCausalLMClipEmb.from_pretrained(
        "microsoft/git-large-coco",
        torch_dtype=torch.float16
    ).to(device0).eval()
    git_model.requires_grad_(False)

    # ===================== MLP =====================
    print("Loading MLP...")
    mlp_model = MappingNetwork().to(device0).half()
    mlp = torch.load(mlp_ckpt_path)
    mlp_model.load_state_dict(mlp['model_state_dict'])
    mlp_model.eval()

    # ===================== T5 =====================
    print("Loading T5 model...")
    t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    t5_model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-large",
        torch_dtype=torch.float16
    ).to(device0).eval()
    t5_model.requires_grad_(False)

    # ===================== Stable Diffusion =====================
    print("Loading Stable Diffusion (text encoder only)...")
    config = OmegaConf.load(stable_diffusion_config_path)

    def load_model_from_config(config, ckpt, verbose=False):
        pl_sd = torch.load(ckpt, map_location=device0)
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        model = model.to(device0).half().eval()
        # 直接卸载巨大的 UNet，省超多显存
        model.model.diffusion_model = None
        return model

    LDM_model = load_model_from_config(config, stable_diffusion_skpt_path)

    def get_text_features(text):
        with torch.cuda.amp.autocast(dtype=torch.float16):
            return LDM_model.get_learned_conditioning(text).cpu().detach().squeeze()

    # ===================== 开始解码 =====================
    all_texts = []
    print("Decoding 982 samples...")

    for i in tqdm(range(982)):
        image_embedding = img_prior[i].to(device0, non_blocking=True)
        proj_text_embedding = txt_proj[i:i + 1].to(device0, non_blocking=True)

        # GIT 生成
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            clip_feature = image_embedding
            git_feature = mlp_model(clip_feature)
            git_feature_process = robust_inverse_scale_manual(git_feature, 0.155, 1.25)

            generated_ids = git_model.generate(
                pixel_values=git_feature_process,
                num_beams=10,
                diversity_penalty=1.0,
                num_beam_groups=5,
                max_length=25
            )
            text_git_captioning = git_processor.batch_decode(generated_ids, skip_special_tokens=True)

        # SD 文本特征
        text_git_embedding = get_text_features(text_git_captioning)

        # 检索最优
        best_pick = retrieval_text_embedding(torch.tensor(text_git_embedding), proj_text_embedding.cpu())
        best_git_text = text_git_captioning[best_pick]

        # T5 优化
        input_text = f"Rewrite the following image annotation to make it meaningful and coherent: {best_git_text}"
        inputs = t5_tokenizer(input_text, return_tensors='pt', truncation=True, max_length=40).to(device0)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            outs = t5_model.generate(
                input_ids=inputs["input_ids"],
                max_length=30,
                num_beams=10,
                attention_mask=inputs["attention_mask"],
            )
        T5_text = t5_tokenizer.decode(outs[0], skip_special_tokens=True)
        all_texts.append(T5_text)

        # 强制清理显存
        del (clip_feature, git_feature, git_feature_process,
             generated_ids, text_git_captioning, text_git_embedding,
             best_pick, best_git_text, inputs, outs, T5_text)
        torch.cuda.empty_cache()

    # 保存结果
    os.makedirs(SAVE_DIR, exist_ok=True)
    torch.save(all_texts, os.path.join(SAVE_DIR, "ours_decoding_texts.pt"))
    with open(os.path.join(SAVE_DIR, "ours_decoding_texts.txt"), "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(f"{t}\n")

    print("\n✅ 脑字幕解码完成，已保存到 result_text 文件夹")