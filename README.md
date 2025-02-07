# Brain-Imager:A Multimodal Framework for Image Reconstruction and Captioning from Human Brain Activity

# Method

# Data
This experiment used the same data set as previous studies. https://huggingface.co/datasets/pscotti/naturalscenesdataset/tree/main/webdataset_avg_split.

# Model
Download files from this link and put it in folder versatile_diffusion. https://huggingface.co/shi-labs/versatile-diffusion/tree/main.
Download files from this link and put it in folder taming-transformers. https://github.com/CompVis/taming-transformers.
You need to download the checkpoint file :sd-v1-4.ckpt and the config file :v1-inference.yaml for Stable Diffusion v1-4 from Hugging Face. Store them in the folders model/

# Train

# Inference
python inference.py
