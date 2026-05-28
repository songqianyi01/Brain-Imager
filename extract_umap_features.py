import os

os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
import torch
import numpy as np

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.empty_cache()

import utils
from models import Clipper, BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork
import argparse
from tqdm import tqdm

device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 42
utils.seed_everything(seed=seed)


# ===================== 图像支路：批量提取 3 个特征 =====================
@torch.no_grad()
def extract_image_features(batch_voxel, ckpt_path):
    out_dim = 257 * 768
    voxel2clip_kwargs = dict(in_dim=15724, out_dim=out_dim, clip_size=768, use_projector=True)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    voxel2clip.eval().to(device0)

    out_dim = 768
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim, depth=6, dim_head=64, heads=12,
        causal=False, num_tokens=257, learned_query_mode="pos_emb"
    )
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network, image_embed_dim=out_dim, condition_on_text_encodings=False,
        timesteps=100, cond_drop_prob=0.2, image_embed_scale=None, voxel2clip=voxel2clip,
    )
    checkpoint = torch.load(ckpt_path, map_location=device0)
    diffusion_prior.load_state_dict(checkpoint['model_state_dict'], strict=False)
    diffusion_prior.eval().to(device0)

    backbone_list = []
    proj_list = []
    prior_list = []

    for i, voxel in enumerate(tqdm(batch_voxel, desc="图像支路处理中")):
        torch.cuda.empty_cache()
        voxel = voxel.float().unsqueeze(0).to(device0)

        out = voxel2clip(voxel)
        backbone = out[0] if isinstance(out, tuple) else out

        prior_feat, proj_feat = utils.reconstruction_image_embeddings(
            None, voxel, None, None, None, None,
            voxel2clip_cls=None, diffusion_priors=[diffusion_prior],
            text_token=None, img_lowlevel=None, num_inference_steps=None,
            n_samples_save=1, recons_per_sample=16, guidance_scale=None,
            img2img_strength=None, timesteps_prior=100, seed=seed,
            retrieve=None, plotting=None, img_variations=False, verbose=None,
        )

        # ✅ 保存 16 条嵌入，形状 [16, 257, 768]
        prior_list.append(prior_feat.cpu())
        proj_list.append(proj_feat.cpu())
        backbone_list.append(backbone.cpu())

        del out, backbone, prior_feat, proj_feat, voxel

    del voxel2clip, diffusion_prior, prior_network, checkpoint
    torch.cuda.empty_cache()

    # ✅ 拼接成正确形状：
    # prior: [982, 16, 257, 768]
    # proj / backbone: [982, ...] 保持不变
    return torch.stack(prior_list), torch.cat(proj_list), torch.cat(backbone_list)


# ===================== 文本支路：批量提取 3 个特征 =====================
@torch.no_grad()
def extract_text_features(batch_voxel, ckpt_path):
    out_dim = 77 * 768
    voxel2clip_kwargs = dict(in_dim=15724, out_dim=out_dim, clip_size=768, use_projector=True)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    voxel2clip.eval().to(device0)

    out_dim = 768
    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim, depth=6, dim_head=64, heads=12,
        causal=False, num_tokens=77, learned_query_mode="pos_emb"
    )
    diffusion_prior = BrainDiffusionPrior(
        net=prior_network, image_embed_dim=out_dim, condition_on_text_encodings=False,
        timesteps=100, cond_drop_prob=0.2, image_embed_scale=None, voxel2clip=voxel2clip,
    )
    checkpoint = torch.load(ckpt_path, map_location=device0)
    diffusion_prior.load_state_dict(checkpoint['model_state_dict'], strict=False)
    diffusion_prior.eval().to(device0)

    backbone_list = []
    proj_list = []
    prior_list = []

    for i, voxel in enumerate(tqdm(batch_voxel, desc="文本支路处理中")):
        torch.cuda.empty_cache()
        voxel = voxel.float().unsqueeze(0).to(device0)

        out = voxel2clip(voxel)
        backbone = out[0] if isinstance(out, tuple) else out

        prior_feat, proj_feat = utils.reconstruction_image_embeddings(
            None, voxel, None, None, None, None,
            voxel2clip_cls=None, diffusion_priors=[diffusion_prior],
            text_token=None, img_lowlevel=None, num_inference_steps=None,
            n_samples_save=1, recons_per_sample=16, guidance_scale=None,
            img2img_strength=None, timesteps_prior=100, seed=seed,
            retrieve=None, plotting=None, img_variations=False, verbose=None,
        )

        # ✅ 保存 16 条嵌入，形状 [16, 77, 768]
        prior_list.append(prior_feat.cpu())
        proj_list.append(proj_feat.cpu())
        backbone_list.append(backbone.cpu())

        del out, backbone, prior_feat, proj_feat, voxel

    del voxel2clip, diffusion_prior, prior_network, checkpoint
    torch.cuda.empty_cache()

    # ✅ 拼接成正确形状：
    # prior: [982, 16, 77, 768]
    # proj / backbone: [982, ...] 保持不变
    return torch.stack(prior_list), torch.cat(proj_list), torch.cat(backbone_list)


# ===================== 主程序 =====================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmri_pt_path", type=str,
                        default="/media/data/songzengyu/Brain-Imager/code/run_server_github/voxels.pt")
    args = parser.parse_args()

    image_ckpt = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/CI.pth'
    text_ckpt = '/media/data/songzengyu/Brain-Imager/code/run_server_github/model/CT.pth'

    print("加载 fMRI 数据...")
    fmri_data = torch.load(args.fmri_pt_path)
    print(f"成功加载 {len(fmri_data)} 个 fMRI 样本")

    print("\n开始提取文本支路特征...")
    txt_prior, txt_proj, txt_backbone = extract_text_features(fmri_data, text_ckpt)

    print("\n保存文本特征...")
    torch.save(txt_backbone, "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_backbone.pt")
    torch.save(txt_proj, "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_proj.pt")
    torch.save(txt_prior, "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/txt_prior.pt")

    print("\n开始提取图像支路特征...")
    img_prior, img_proj, img_backbone = extract_image_features(fmri_data, image_ckpt)

    print("\n保存图像特征...")
    torch.save(img_backbone, "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/img_backbone.pt")
    torch.save(img_proj, "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/img_proj.pt")
    torch.save(img_prior, "/media/data/songzengyu/Brain-Imager/code/run_server_github/features/img_prior.pt")

    print("\n全部完成！输出 6 个文件：")
    print("img_backbone.pt | img_proj.pt | img_prior.pt  (图像prior: 982,16,257,768)")
    print("txt_backbone.pt | txt_proj.pt | txt_prior.pt  (文本prior: 982,16,77,768)")