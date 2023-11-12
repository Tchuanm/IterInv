# from diffusers import StableDiffusionXLPipeline
from pipelines.SDXL_pipeline import StableDiffusionXLPipeline
# from pipelines.SDUP_pipeline import StableDiffusionUpscalePipeline
# from pipelines.ddim_pipeline import StableDiffusionDDIMInvPipeline
from diffusers.image_processor import VaeImageProcessor

from diffusers.utils import pt_to_pil, numpy_to_pil
import torch
from IPython.display import display
import numpy as np
from PIL import Image
import PIL
image_processor=VaeImageProcessor()
import os


# save_path = 'output/SDXL_rec/cat_hq.png'
# img_pth='images/cat_hq.jpg'


pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float32, variant="fp32", use_safetensors=True
    # "stabilityai/stable-diffusion-xl-base-0.9", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
# pipe=pipe.to("cuda")
pipe.enable_model_cpu_offload()
os.system(f"CUDA_VISIBLE_DEVICES=7")

with open('prompt_all_imgs.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 使用分号分隔每一行 
        img_pth, prompt = line.strip().split(': ')
        save_path = 'output/SDXL_rec/' + img_pth.split('/')[-1].split('.')[0] + '.png'


        _inv_raw_image = Image.open(img_pth).convert("RGB").resize((1024,1024))
        inv_raw_image = image_processor.preprocess(_inv_raw_image)

        """1st stage compress and decompress """
        generator = torch.manual_seed(0)
        with torch.no_grad():
            latent = pipe.prepare_image_latents(inv_raw_image.cuda(), 1, pipe.vae.dtype, 'cuda',generator=generator)

        with torch.no_grad():
            # image_pt = pipe.decode_latents_pt(latent)
            image = pipe.decode_latents(latent)

        numpy_to_pil(image)[0].save(save_path)


