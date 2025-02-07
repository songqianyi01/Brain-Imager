import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from src import utils
import numpy as np
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"


train_url = "{" + f"/mnt/sda/songzengyu/code/brain_imager/data/train/train_subj01_" + "{0..17}.tar," + f"/mnt/sda/songzengyu/code/brain_imager/data/val/val_subj01_0.tar" + "}"
val_url = f"/mnt/sda/songzengyu/code/brain_imager/data/test/test_subj01_" + "{0..1}.tar"
print(train_url, "\n", val_url)
meta_url = f"/mnt/sda/songzengyu/code/brain_imager/data/metadata_subj01.json"
num_train = (8559 + 300)*3
num_val = 982

print('Prepping train and validation dataloaders...')
train_dl, val_dl, num_train, num_val = utils.get_dataloaders(
    1, 'images',
    num_devices=1,
    num_workers=1,
    train_url=train_url,
    val_url=val_url,
    meta_url=meta_url,
    num_train=num_train,
    num_val=num_val,
    val_batch_size=1,
    cache_dir='/mnt/sda/songzengyu/code/brain_imager/data',  # "/tmp/wds-cache",
    seed=42,
    voxels_key='nsdgeneral.npy',
    to_tuple=["voxels", "images", "coco"],
    local_rank=0,
    world_size=1,
)




import clip
from models import Clipper
clip_model = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)

# 自定义的缩放函数
def robust_scale_manual(data, median, iqr):
    return (data - median) / iqr

# 自定义的逆缩放函数
def robust_inverse_scale_manual(scaled_data, median, iqr):
    return scaled_data * iqr + median



def extract_clip_features(image_batch):
    with torch.no_grad():
        clip_features = clip_model.embed_image(image_batch)
    return clip_features


from transformers import AutoProcessor,AutoModelForCausalLM
from modeling_git import GitForCausalLMClipEmb
git_processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
git_model = GitForCausalLMClipEmb.from_pretrained("microsoft/git-large-coco")
git_model.to(device)
git_model.eval().requires_grad_(False)
vision_encoder=git_model.git.image_encoder

def extract_git_features(image):
    with torch.no_grad():
        image_npy = image.squeeze(0).cpu().numpy()
        image_tran = np.transpose(image_npy, (1, 2, 0))
        image_jpg = (image_tran * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_jpg)
        pixel_values = git_processor(images=image_pil, return_tensors="pt").pixel_values.to(device)
        git_features = vision_encoder(pixel_values).last_hidden_state
    return git_features


import torch.nn as nn
# 定义映射网络
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


import torch.optim as optim

# 初始化映射网络
mapping_network = MappingNetwork().to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 使用 L2 损失
optimizer = optim.Adam(mapping_network.parameters(), lr=1e-4)
best_val_loss = float('inf')

# 训练函数
def train_model(train_dl, mapping_network, optimizer, criterion):
    mapping_network.train()
    total_loss = 0

    for train_i, (voxel, image, coco) in enumerate(train_dl):
        images = image.to(device)

        # 提取 CLIP 和 GIT 特征
        clip_features = extract_clip_features(images)  # (batch_size, 257, 768)

        git_features = extract_git_features(images)  # (batch_size, 257, 1024)
        git_features_process = robust_scale_manual(git_features, 0.155, 1.25)

        # 映射 CLIP 特征到 GIT 特征空间
        predicted_git_features = mapping_network(clip_features)  # (batch_size, 257, 1024)

        # 计算损失
        loss = criterion(predicted_git_features, git_features_process)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / 26577


# 验证函数
def validate_model(val_dl, mapping_network, criterion):
    mapping_network.eval()
    total_loss = 0

    with torch.no_grad():
        for val_i, (voxel, image, coco) in enumerate(val_dl):
            images = image.to(device)

            # 提取 CLIP 和 GIT 特征
            clip_features = extract_clip_features(images)  # (batch_size, 257, 768)

            git_features = extract_git_features(images)  # (batch_size, 257, 1024)
            git_features_process = robust_scale_manual(git_features,0.155, 1.25)

            # 映射 CLIP 特征到 GIT 特征空间
            predicted_git_features = mapping_network(clip_features)  # (batch_size, 257, 1024)

            # 计算损失
            loss = criterion(predicted_git_features, git_features_process)
            total_loss += loss.item()

    return total_loss / 982


def save_model(model, optimizer, epoch, val_loss, best_val_loss, filepath='best_model_new.pth'):
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # 保存模型状态字典
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }, filepath)
        print(f"保存模型，验证损失: {best_val_loss:.4f}")
    return best_val_loss

def save_model_last(model, optimizer, epoch, train_loss, filepath='last_model_new.pth'):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': train_loss,
        }, filepath)
        print(f"保存模型，验证损失: {train_loss:.4f}")


num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train_model(train_dl, mapping_network, optimizer, criterion)
    val_loss = validate_model(val_dl, mapping_network, criterion)
    best_val_loss = save_model(mapping_network, optimizer, epoch, val_loss, best_val_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

save_model_last(mapping_network, optimizer, epoch, val_loss)
