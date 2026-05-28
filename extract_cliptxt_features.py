import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
import sys
sys.path.append('/media/data/songzengyu/Brain-Imager/code/run_server_github/taming-transformers')

device0 = "cuda:0"

# ===================== 你的路径 =====================
TEXT_PT_PATH = "/media/data/songzengyu/Brain-Imager/code/run_server_github/result_text/gt_captions.pt"
SAVE_TEXT_FEATURE_PATH = "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/clip_text_features.pt"

STABLE_DIFFUSION_CONFIG_PATH = "/media/data/songzengyu/Brain-Imager/code/run_server_github/model/v1-inference.yaml"
STABLE_DIFFUSION_CKPT_PATH = "/media/data/songzengyu/Brain-Imager/code/run_server_github/model/sd-v1-4.ckpt"

# ===================== 工具函数 =====================
def instantiate_from_config(config):
    from ldm.util import instantiate_from_config
    return instantiate_from_config(config)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda().eval()
    return model

# ===================== 只加载一次模型！！！=====================
print("Loading Stable Diffusion model once...")
config = OmegaConf.load(STABLE_DIFFUSION_CONFIG_PATH)
LDM_model = load_model_from_config(config, STABLE_DIFFUSION_CKPT_PATH)
print("Model loaded successfully!\n")

# ===================== 加载文本 =====================
print(f"Loading captions from: {TEXT_PT_PATH}")
all_captions = torch.load(TEXT_PT_PATH)
print(f"Total captions: {len(all_captions)}")

# ===================== 逐 句 提 取 =====================
all_text_feats = []

print("\nStart extracting ONE-BY-ONE...")
with torch.no_grad():
    for text in tqdm(all_captions, desc="Extracting"):
        # 逐句输入，提取特征
        feat = LDM_model.get_learned_conditioning(text).cpu().detach()
        all_text_feats.append(feat)

# ===================== 拼接保存 =====================
all_text_feats = torch.cat(all_text_feats, dim=0)
print(f"\nFinal feature shape: {all_text_feats.shape}")

torch.save(all_text_feats, SAVE_TEXT_FEATURE_PATH)
print(f"\n✅ Saved to: {SAVE_TEXT_FEATURE_PATH}")

# 最后再释放
del LDM_model
torch.cuda.empty_cache()