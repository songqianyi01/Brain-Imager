import torch
import numpy as np
voxels= torch.load('/media/data/songzengyu/Brain-Imager/code/run_server_github/voxels.pt')
for i in [31]:
    voxel = voxels[i].numpy()
    np.save(f"/media/data/songzengyu/Brain-Imager/code/run_server_github/examples/fmri/fmri_flatten_{i}", voxel)