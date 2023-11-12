import PIL
import requests
import torch
from io import BytesIO

from diffusers import StableDiffusionDiffEditPipeline, DDIMScheduler, DDIMInverseScheduler
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import numpy as np
from diffusers.utils import pt_to_pil, numpy_to_pil


# def download_image(url):
#     # response = requests.get(url)
#     return PIL.Image.open(BytesIO(response.content)).convert("RGB")
image_processor=VaeImageProcessor()

# img_url = "images/pix2pix-zero/cat/cat_3.png"

# init_image = download_image(img_url).resize((768, 768))
img_pth = 'images/pix2pix-zero/cat/cat_3.png'
_inv_raw_image_0 = Image.open(img_pth).convert("RGB").resize((768,768))
inv_raw_image_0 = image_processor.preprocess(_inv_raw_image_0)

pipe = StableDiffusionDiffEditPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

mask_prompt = "A cat"
prompt = "A dog"

mask_image = pipe.generate_mask(image=inv_raw_image_0, source_prompt=prompt, target_prompt=mask_prompt)
_mask_image = np.repeat(mask_image, 3, axis=0)
_mask_image = torch.from_numpy(_mask_image[np.newaxis, :])
pt_to_pil(_mask_image)[0].save(f"diffedit_SD_mask.png")
image_latents = pipe.invert(image=inv_raw_image_0, prompt=mask_prompt).latents
image = pipe(prompt=prompt, mask_image=mask_image, image_latents=image_latents).images[0]
image.save(f"diffedit_SD.png")

