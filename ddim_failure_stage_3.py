from pipelines.deepfloyd_pipeline import IFPipeline
from pipelines.deepfloyd_inv_pipeline import IFInvPipeline

from pipelines.deepfloyd_SR_pipeline import IFSuperResolutionPipeline
from pipelines.deepfloyd_SR_inv_pipeline import IFSuperResolutionInvPipeline

from pipelines.SDUP_pipeline import StableDiffusionUpscalePipeline

from pipelines.scheduler_ddim import DDIMScheduler
from pipelines.scheduler_ddpm import DDPMScheduler
from pipelines.scheduler_inv import DDIMInverseScheduler

from diffusers.utils import pt_to_pil, numpy_to_pil
import torch
from IPython.display import display
import numpy as np
from PIL import Image

# torch.cuda.set_device(3)
device = torch.device('cuda')
from diffusers.image_processor import VaeImageProcessor
import argparse
import os

### NOTE: image_processor can help to make the batch size ready
image_processor=VaeImageProcessor()

def pil_to_numpy_torch(pil_img: Image):
    image_org = np.array(pil_img).astype(np.float32).transpose(2,0,1)/255.0
    image_org = 2.0 * image_org - 1.0
    image_org = torch.from_numpy(image_org)
    return image_org

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='images/cat_hq.jpg')
    parser.add_argument('--prompt_str', type=str, default='a cat in an astronaut suit')
    parser.add_argument('--output_fold', type=str, default='')

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arguments()
    print(args)
    saving_path = f'DDIM_output/{args.output_fold}/{args.prompt_str}'
    # os.makedirs(saving_path, exist_ok=True)
    generator = torch.manual_seed(0)

    ### NOTE: upscaler can only be float32
    stage_3 = StableDiffusionUpscalePipeline.from_pretrained(
                        "stabilityai/stable-diffusion-x4-upscaler", 
                        torch_dtype=torch.float32)
    # stage_3=stage_3.to(device)
    stage_3.enable_model_cpu_offload()

    _inv_raw_image_2 = Image.open(args.input_image).convert("RGB").resize((1024,1024))
    inv_raw_image_2 = pil_to_numpy_torch(_inv_raw_image_2)

    _inv_raw_image_1 = Image.open(args.input_image).convert("RGB").resize((256,256))
    inv_raw_image_1 = pil_to_numpy_torch(_inv_raw_image_1)

    generator = torch.manual_seed(0)
    with torch.no_grad():
        latent = stage_3.prepare_image_latents(inv_raw_image_2.unsqueeze(0).to(device), 1, stage_3.vae.dtype, device,generator=generator)

    torch.cuda.empty_cache()


    # latent=torch.load('post_submit/latent_1024_stage3_cat_hq.pt')

    stage_3.scheduler = DDIMInverseScheduler.from_config(stage_3.scheduler.config)
    noise_level_3=100


    image_tuple_3_inv, _ = stage_3(
                    prompt=args.prompt_str,
                    image=[inv_raw_image_1], 
                    noise_level=noise_level_3, 
                    generator=generator,
                    output_type="latent",
                    guidance_scale=1.0,
                    latents=latent,
                    num_inference_steps=100,
                    )

    # image_2_inv=image_tuple_3_inv.images
    # torch.save(image_2_inv.cpu().detach(), 'post_submit/image_2_inv.pt')

    image_3_inv=image_tuple_3_inv.images


    stage_3.scheduler = DDIMScheduler.from_config(stage_3.scheduler.config)
    noise_level_3=100

    image_tuple_3_rec, _  = stage_3(
                    prompt=args.prompt_str,
                    image=[inv_raw_image_1], 
                    noise_level=noise_level_3, 
                    generator=generator,
                    output_type="pil",
                    guidance_scale=1.0,
                    num_inference_steps=100,
                    latents=image_3_inv
                    )

    image_3_rec=image_tuple_3_rec.images

    image_3_rec[0].save(f"{saving_path}_if_stage_III_rec.png")

