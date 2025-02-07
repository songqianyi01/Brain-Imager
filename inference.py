import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
import torch

import numpy as np
import utils
import torch.nn as nn
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from models import Clipper, OpenClipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, \
    Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
import torchvision.transforms as transforms
import argparse
from transformers import AutoProcessor,AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from transformers import AutoTokenizer, T5ForConditionalGeneration
import sys
sys.path.append('E:/songzengyu/temp/taming-transformers')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device0 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 42
utils.seed_everything(seed=seed)


def generate_image_embedding(voxel, ckpt_path):
    out_dim = 257 * 768
    voxel2clip_kwargs = dict(in_dim=15724, out_dim=out_dim, clip_size=768, use_projector=True)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    voxel2clip.requires_grad_(False)
    voxel2clip.eval()

    out_dim = 768
    depth = 6
    dim_head = 64
    heads = 12  # heads * dim_head = 12 * 64 = 768
    timesteps = 100  # 100

    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=257,
        learned_query_mode="pos_emb"
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    )

    checkpoint = torch.load(ckpt_path, map_location=device0)
    state_dict = checkpoint['model_state_dict']
    diffusion_prior.load_state_dict(state_dict, strict=False)
    diffusion_prior.eval().to(device0)

    diffusion_priors = [diffusion_prior]
    pass
    with torch.no_grad():
        image_embedding, proj_embedding = utils.reconstruction_image_embeddings(
            None, voxel.to(device0),
            None, None, None, None,
            voxel2clip_cls=None,  # diffusion_prior_cls.voxel2clip,
            diffusion_priors=diffusion_priors,
            text_token=None,
            img_lowlevel=None,
            num_inference_steps=None,
            n_samples_save=1,
            recons_per_sample=16,
            guidance_scale=None,
            img2img_strength=None,  # 0=fully rely on img_lowlevel, 1=not doing img2img
            timesteps_prior=100,
            seed=seed,
            retrieve=None,
            plotting=None,
            img_variations=False,
            verbose=None,
        )

        print("image embedding has been created")
        del diffusion_priors,checkpoint,prior_network,voxel2clip
        torch.cuda.empty_cache()
    return image_embedding,proj_embedding

def generate_text_embedding(voxel, ckpt_path):
    out_dim = 77 * 768
    voxel2clip_kwargs = dict(in_dim=15724, out_dim=out_dim, clip_size=768, use_projector=True)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)
    voxel2clip.requires_grad_(False)
    voxel2clip.eval()

    out_dim = 768
    depth = 6
    dim_head = 64
    heads = 12  # heads * dim_head = 12 * 64 = 768
    timesteps = 100  # 100

    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=77,
        learned_query_mode="pos_emb"
    )

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    )

    checkpoint = torch.load(ckpt_path, map_location=device0)
    state_dict = checkpoint['model_state_dict']
    diffusion_prior.load_state_dict(state_dict, strict=False)
    diffusion_prior.eval().to(device0)

    diffusion_priors = [diffusion_prior]
    pass
    with torch.no_grad():
        text_embedding, proj_embedding = utils.reconstruction_image_embeddings(
            None, voxel.to(device0),
            None, None, None, None,
            voxel2clip_cls=None,  # diffusion_prior_cls.voxel2clip,
            diffusion_priors=diffusion_priors,
            text_token=None,
            img_lowlevel=None,
            num_inference_steps=None,
            n_samples_save=1,
            recons_per_sample=16,
            guidance_scale=None,
            img2img_strength=None,  # 0=fully rely on img_lowlevel, 1=not doing img2img
            timesteps_prior=100,
            seed=seed,
            retrieve=None,
            plotting=None,
            img_variations=False,
            verbose=None,
        )
        print("text embedding has been created")
        del diffusion_priors,checkpoint,prior_network,voxel2clip
        torch.cuda.empty_cache()
    return text_embedding ,proj_embedding

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
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

def generate_nature_vector(voxel, nature_ckpt_path, vd_cache_dir):
    ckpt_path = nature_ckpt_path
    vd_cache_dir = vd_cache_dir
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device0)
        state_dict = checkpoint['model_state_dict']
        voxel2sd = Voxel2StableDiffusionModel(in_dim=15724)
        voxel2sd.load_state_dict(state_dict, strict=False)

        voxel2sd.to(device0)
        voxel2sd.eval()

    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device0).to(torch.float16)
    vd_pipe.image_unet.eval()
    vd_pipe.vae.eval()
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)
    with torch.no_grad():
        ae_preds = voxel2sd(voxel.float().to(device0))
        blurry_recons = vd_pipe.vae.decode(ae_preds.half() / 0.18215).sample / 2 + 0.5
        blurry_recons = blurry_recons.cpu().numpy()
        print("nature vector has been created")

        del voxel2sd,vd_pipe,checkpoint
        torch.cuda.empty_cache()
    return blurry_recons

def generate_panoramic_vector(voxel, panoramic_ckpt_path, vd_cache_dir):
    ckpt_path = panoramic_ckpt_path
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path,map_location=device0)
        state_dict = checkpoint['model_state_dict']
        voxel2sd = Voxel2StableDiffusionModel(in_dim=15724)
        voxel2sd.load_state_dict(state_dict, strict=False)

        voxel2sd.to(device0)
        voxel2sd.eval()
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device0).to(torch.float16)
    vd_pipe.image_unet.eval()
    vd_pipe.vae.eval()
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)
    with torch.no_grad():
        ae_preds = voxel2sd(voxel.float().to(device0))
        blurry_recons = vd_pipe.vae.decode(ae_preds.half() / 0.18215).sample / 2 + 0.5
        blurry_recons = blurry_recons.cpu().numpy()
        print("panoramic vector has been created")
        del voxel2sd, vd_pipe,checkpoint
        torch.cuda.empty_cache()
    return blurry_recons

def gaussian_pyramid(img, levels):
    """构建高斯金字塔"""
    pyramid = [img]
    for i in range(levels):
        img = cv2.pyrDown(img)
        pyramid.append(img)
    return pyramid


def laplacian_pyramid(gaussian_pyr):
    """构建拉普拉斯金字塔"""
    laplacian_pyr = []
    for i in range(len(gaussian_pyr) - 1):
        size = (gaussian_pyr[i].shape[1], gaussian_pyr[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyr[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyr[i], expanded)
        laplacian_pyr.append(laplacian)
    laplacian_pyr.append(gaussian_pyr[-1])  # 最后一级拉普拉斯金字塔是高斯金字塔的最后一级
    return laplacian_pyr


def reconstruct_from_laplacian_pyramid(laplacian_pyr):
    """从拉普拉斯金字塔重建图像"""
    img = laplacian_pyr[-1]  # 从最小图层开始重建
    for i in range(len(laplacian_pyr) - 2, -1, -1):
        size = (laplacian_pyr[i].shape[1], laplacian_pyr[i].shape[0])
        img = cv2.pyrUp(img, dstsize=size)
        img = cv2.add(img, laplacian_pyr[i])
    return img

def mix_blur_vector(nature_vector,panoramic_vector):
    clear_images = nature_vector.astype('float64')
    struct_images = panoramic_vector.astype('float64')
    fused_images = np.zeros_like(clear_images)

    # 设置金字塔层数
    levels = 6
    lameta = [1, 1, 1, 1, 0.6, 0.8, 0.8]

    # 对每张图片分别进行多尺度融合

    for c in range(3):  # 针对每个通道（R, G, B）
        img_clear = clear_images[0, c, :, :]
        img_struct = struct_images[0, c, :, :]

        # 构建高斯和拉普拉斯金字塔
        gaussian_pyr_clear = gaussian_pyramid(img_clear, levels)
        gaussian_pyr_struct = gaussian_pyramid(img_struct, levels)

        lap_pyr_clear = laplacian_pyramid(gaussian_pyr_clear)
        lap_pyr_struct = laplacian_pyramid(gaussian_pyr_struct)

        fused_pyr = []
        for level in range(levels + 1):
            fused_layer = cv2.addWeighted(lap_pyr_clear[level], lameta[level], lap_pyr_struct[level],
                                          (1 - lameta[level]), 0)
            fused_pyr.append(fused_layer)

        # 从融合后的金字塔重建图像
        fused_img = reconstruct_from_laplacian_pyramid(fused_pyr)

        # 保存融合后的通道
        fused_images[0, c, :, :] = fused_img
    print("lpm mix...")
    return fused_images

def image_reconstruct(blurry_vector,best_text_embedding,image_embedding, proj_image_embedding, image_out_path):

    vd_pipe1 = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device0).to(torch.float16)
    vd_pipe1.image_unet.eval()
    vd_pipe1.vae.eval()
    vd_pipe1.image_unet.requires_grad_(False)
    vd_pipe1.vae.requires_grad_(False)

    noise_scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir, subfolder="scheduler")

    # Set weighting of Dual-Guidance
    text_image_ratio = 0  # .5 means equally weight text and image, 0 means use only image
    for name, module in vd_pipe1.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = text_image_ratio
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    unet1 = vd_pipe1.image_unet
    vae = vd_pipe1.vae


    vd_pipe2 = VersatileDiffusionDualGuidedPipeline.from_pretrained(vd_cache_dir).to(device0).to(torch.float16)
    vd_pipe2.image_unet.eval()
    vd_pipe2.vae.eval()
    vd_pipe2.image_unet.requires_grad_(False)
    vd_pipe2.vae.requires_grad_(False)

    num_inference_steps = 20

    # Set weighting of Dual-Guidance
    text_image_ratio = 1  # .5 means equally weight text and image, 0 means use only image
    for name, module in vd_pipe2.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = text_image_ratio
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

    unet2 = vd_pipe2.image_unet

    del vd_pipe1, vd_pipe2
    torch.cuda.empty_cache()

    img_variations = False

    retrieve = False
    plotting = False
    saving = True
    verbose = False
    imsize = 512

    if img_variations:
        guidance_scale = 7.5
    else:
        guidance_scale = 3.5

    clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device0)

    with torch.no_grad():
        if 1:

            blurry_recons = torch.tensor(blurry_vector).to(device0)
            image_clip_embedding = image_embedding
            proj_embedding = proj_image_embedding
            text_prompt_embedding = best_text_embedding.unsqueeze(0)


            grid, brain_recons, laion_best_picks, recon_img = utils.reconstruction_integrity_noise_assign(
                clip_extractor, unet1.to(device0), unet2.to(device0), vae.to(device0), noise_scheduler,
                voxel2clip_cls=None,  # diffusion_prior_cls.voxel2clip,
                diffusion_priors=None,
                text_token=text_prompt_embedding,
                img_lowlevel=blurry_recons,
                num_inference_steps=num_inference_steps,
                n_samples_save=1,
                recons_per_sample=16,
                guidance_scale=guidance_scale,
                img2img_strength=0.85,  # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps_prior=None,
                seed=seed,
                retrieve=retrieve,
                plotting=plotting,
                img_variations=img_variations,
                verbose=verbose,
                input_embedding=image_clip_embedding,
                proj_embedding=proj_embedding,
            )

            brain_recons = brain_recons[:, laion_best_picks.astype(np.int8)].squeeze(0).squeeze(0)
            print(brain_recons.shape)
            to_pil = transforms.ToPILImage()
            img = brain_recons
            img_pil = to_pil(img)
            img_pil.save(f'{image_out_path}/image.jpg')
            del unet1, unet2, vae, noise_scheduler,clip_extractor
            torch.cuda.empty_cache()
            print("image stimulate has been reconstructed")

class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 1024),  # 从768维映射到1024维
            nn.ReLU(),
            nn.Linear(1024, 1024)  # 最终输出1024维
        )

    def forward(self, x):
        return self.fc(x)

def robust_inverse_scale_manual(scaled_data, median, iqr):
    return scaled_data * iqr + median

def git_captioning(mlp_ckpt_path, image_embedding):
    git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
    git_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
    git_model.to(device0)
    git_model.eval().requires_grad_(False)

    model = MappingNetwork().to(device0)
    mlp = torch.load(mlp_ckpt_path)
    model.load_state_dict(mlp['model_state_dict'])
    model.eval()

    with torch.no_grad():
        clip_feature = image_embedding
        git_feature = model(clip_feature)
        git_feature_process = robust_inverse_scale_manual(git_feature, 0.155, 1.25)
        generated_ids = git_model.generate(pixel_values=git_feature_process, num_beams=10, diversity_penalty=1.0,
                                           num_beam_groups=5, max_length=25)
        generated_caption = git_processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("git caption has been created")
        del git_processor,git_model,model,mlp
        torch.cuda.empty_cache()
    return generated_caption

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cuda:0")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def get_text_features(LDM_model, text):
    cap = LDM_model.get_learned_conditioning(text).cpu().detach().numpy().squeeze()
    return cap

def git_embedding(text_git_captioning, stable_diffusion_config_path, stable_diffusion_skpt_path):
    config = OmegaConf.load(stable_diffusion_config_path)
    LDM_model = load_model_from_config(config, stable_diffusion_skpt_path).to(device0)
    embedding = get_text_features(LDM_model, text_git_captioning)
    del config,LDM_model
    torch.cuda.empty_cache()
    return embedding

def T5_refine(best_git_text,text_out_path):
    model_name = "google/flan-t5-large"  # 或者使用更大的模型，例如 "t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device0)

    input_text = f"Rewrite the following image annotation to make it meaningful and coherent: {best_git_text}"
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding='max_length', max_length=40)
    inputs.to(device0)
    with torch.no_grad():
        outs = model.generate(input_ids=inputs["input_ids"], max_length=30,  # 设定生成文本的最大长度
                              num_beams=10,  # 使用束搜索提高生成质量
                              attention_mask=inputs["attention_mask"], )
    translated_text = tokenizer.decode(outs[0], skip_special_tokens=True)
    with open(f'{text_out_path}/brain_captioning.txt','w') as file:
        file.write(translated_text)

def fmri_flatten_picture(signal,out_path):
    time_series = signal.squeeze(0).numpy()  # 变为 (15724,)
    # 创建时间序列图
    plt.figure(figsize=(12, 4))  # 设置画布大小
    plt.plot(time_series, color='blue', linewidth=0.8)  # 绘制折线图，黑色线条
    plt.xlabel("Voxel")  # x 轴标签，表示时间
    plt.ylabel("Signal Intensity")  # y 轴标签，表示信号强度
    plt.title("fMRI Flatten")  # 图表标题
    plt.tight_layout()
    # 保存为 JPG 文件
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布
    print("fmri flatten image have been created")
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--image_output_path", type=str, default="E:/songzengyu/examples/image_reconstruct",)
parser.add_argument(
    "--text_output_path", type=str, default="E:/songzengyu/examples/brain_captioning",)
parser.add_argument(
    "--fmri_input_path", type=str, default="E:/songzengyu/examples/fmri/fmri_flatten_167.npy",)
parser.add_argument(
    "--fmri_image_output_path", type=str, default="E:/songzengyu/examples/fmri_picture/fmri_flatten.jpg",)
args = parser.parse_args()

image_embedding_ckpt_path='E:/songzengyu/model/CI.pth'
text_embedding_ckpt_path='E:/songzengyu/model/CT.pth'
nature_ckpt_path = 'E:/songzengyu/model/ZN.pth'
panoramic_ckpt_path = 'E:/songzengyu/model/ZP.pth'
mlp_ckpt_path = 'E:/songzengyu/model/text_generate_mlp.pth'
vd_cache_dir = 'E:/songzengyu/versatile_diffusion/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7'
stable_diffusion_config_path = 'E:/songzengyu/model/v1-inference.yaml'
stable_diffusion_skpt_path = 'E:/songzengyu/model/sd-v1-4.ckpt'

voxel= np.load(args.fmri_input_path)
voxel = torch.tensor(voxel)
fmri_flatten_picture(voxel,args.fmri_image_output_path)

image_embedding, proj_image_embedding = generate_image_embedding(voxel,image_embedding_ckpt_path)
text_embedding, proj_text_embedding = generate_text_embedding(voxel,text_embedding_ckpt_path)
best_text_pick = retrieval_text_embedding(text_embedding, proj_text_embedding)
best_text_embedding = text_embedding[best_text_pick]

nature_vector = generate_nature_vector(voxel, nature_ckpt_path, vd_cache_dir)
panoramic_vector = generate_panoramic_vector(voxel, panoramic_ckpt_path, vd_cache_dir)
blurry_vector = mix_blur_vector(nature_vector,panoramic_vector)

image_reconstructed_jpg = image_reconstruct(blurry_vector,best_text_embedding,image_embedding, proj_image_embedding,args.image_output_path)


text_git_captioning = git_captioning(mlp_ckpt_path, image_embedding)
text_git_embedding = git_embedding(text_git_captioning, stable_diffusion_config_path, stable_diffusion_skpt_path)
best_pick = retrieval_text_embedding(torch.tensor(text_git_embedding), proj_text_embedding.cpu())
best_git_text = text_git_captioning[best_pick]
T5_text = T5_refine(best_git_text,args.text_output_path)
print('brain captioning has been created')