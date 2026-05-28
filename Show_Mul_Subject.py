import numpy as np
import matplotlib.pyplot as plt
import torch

pt_file_paths = [
    "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/gt_test_images.pt",
    "/media/data/songzengyu/Brain-Imager/code/run_server_github/result/ours_test_images.pt",
    "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_Mul-Subject/final_subj01_pretrained_1sess_24bs_all_recons.pt",
    "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_Mul-Subject/final_subj01_pretrained_40sess_24bs_all_recons.pt"
]

indices = [104, 298, 278, 188, 136]
labels = ["Ground\nTruth", "Brain-\nImager", "Mul-\n1hours", "Mul-\n40hours"]
save_path = "/media/data/songzengyu/Brain-Imager/code/run_server_github/Result_Mul-Subject/decoding_comparison.png"


def load(path, indices):
    data = torch.load(path, map_location='cpu')
    imgs = []
    for i in indices:
        img = data[i].detach().cpu().numpy()
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        img = img.squeeze()
        imgs.append(img)
    return imgs


all_imgs = [load(p, indices) for p in pt_file_paths]

fig, axes = plt.subplots(4, 5, figsize=(12, 9), dpi=150)
plt.subplots_adjust(wspace=0.05, hspace=0.15, left=0.12, right=0.98, top=0.9)

# 标题往下移，y 调小即可
fig.suptitle('Reconstruction Comparison of Different Methods', fontsize=18, fontweight='bold', y=0.96)

for row in range(4):
    for col in range(5):
        ax = axes[row, col]
        ax.imshow(all_imgs[row][col])
        ax.axis('off')

        if col == 0:
            ax.text(-0.3, 0.5, labels[row],
                    fontsize=14, fontweight='bold',
                    ha='center', va='center',
                    transform=ax.transAxes)

plt.savefig(save_path, bbox_inches='tight')
print("图片已保存：", save_path)