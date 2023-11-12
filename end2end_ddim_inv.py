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

    # stages 1 
    stage_1 = IFInvPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", variant="fp32", torch_dtype=torch.float32)
    stage_1.enable_model_cpu_offload()
    
    # stage 2 
    # stage_2 = IFSuperResolutionPipeline.from_pretrained("DeepFloyd/IF-II-M-v1.0", 
    #                                                        text_encoder=None, variant="fp32", torch_dtype=torch.float32)
    # stage_2.enable_model_cpu_offload()

    # stages 3for2
    ### NOTE: upscaler can only be float32
    stage_2 = StableDiffusionUpscalePipeline.from_pretrained(
                        "stabilityai/stable-diffusion-x4-upscaler", 
                        torch_dtype=torch.float32)
    stage_2.enable_model_cpu_offload()

    # stages 3
    ### NOTE: upscaler can only be float32
    stage_3 = StableDiffusionUpscalePipeline.from_pretrained(
                        "stabilityai/stable-diffusion-x4-upscaler", 
                        torch_dtype=torch.float32)
    stage_3.enable_model_cpu_offload()

    _inv_raw_image_2 = Image.open(args.input_image).convert("RGB").resize((1024,1024))
    inv_raw_image_2 = pil_to_numpy_torch(_inv_raw_image_2)

    _inv_raw_image_1 = Image.open(args.input_image).convert("RGB").resize((256,256))
    inv_raw_image_1 = pil_to_numpy_torch(_inv_raw_image_1)

    _inv_raw_image_0 = Image.open(args.input_image).convert("RGB").resize((64,64))
    inv_raw_image_0 = pil_to_numpy_torch(_inv_raw_image_0)

    


    # stage 1 inversion 
    # text embeds
    stage_1.scheduler = DDIMInverseScheduler.from_config(stage_1.scheduler.config)

    num_inference_steps = 50 
    prompt_embeds, negative_embeds = stage_1.encode_prompt(args.prompt_str)

    output_inv_1, inter_img_list, uncond_embeddings_list = stage_1(
                        prompt_embeds=prompt_embeds, 
                        negative_prompt_embeds=negative_embeds, 
                        generator=generator, 
                        num_inference_steps=num_inference_steps,
                        output_type="pt",
                        image_init=inv_raw_image_0,
                        guidance_scale=1.0,
                        )
    # torch.save(inter_img_list, f'{saving_path}/inter_img_list_1.pt')
    inv_noise_1 = output_inv_1.images
    torch.cuda.empty_cache()
    stage_1.to('cpu')

    # stage 1 reconstraction
    stage_1.scheduler = DDIMScheduler.from_config(stage_1.scheduler.config)
    output_rec_1, inter_img_list, uncond_embeddings_list = stage_1(
                    prompt_embeds=prompt_embeds, 
                    negative_prompt_embeds=negative_embeds, 
                    generator=generator, 
                    num_inference_steps=num_inference_steps,
                    output_type="pt",
                    image_init=inv_noise_1,
                    guidance_scale=7.5,
                    all_latents=inter_img_list,
                    null_inner_steps = 11, 
                    is_NPI=False,
                    )
    image_rec_1 = output_rec_1.images
    torch.cuda.empty_cache()
    stage_1.to('cpu')

    # stage 2 upscale 
    # stage_2.scheduler = DDIMScheduler.from_config(stage_2.scheduler.config)
    # noise_level=250
    ### NOTE: stage 2 is without text_encoder
    # image_tuple_2_sr, _ = stage_2(
    #                 image=image_rec_1, 
    #                 # image=[inv_raw_image_1], 
    #                 generator=generator,
    #                 prompt_embeds=prompt_embeds, 
    #                 negative_prompt_embeds=negative_embeds, 
    #                 output_type="pt",
    #                 guidance_scale=1.0,
    #                 noise_level=noise_level
    #                 )

    # image_2_sr = image_tuple_2_sr.images
    # # pil_image_2_sr = pt_to_pil(image_2_sr)
    # stage_2.to('cpu')
    # torch.cuda.empty_cache()
    # image_2_rec = image_2_sr


    # stage 2 inversion  
    with torch.no_grad():
        latent = stage_2.prepare_image_latents(inv_raw_image_1.unsqueeze(0).to(device), 1, stage_2.vae.dtype, device, generator=generator)
    torch.cuda.empty_cache()
    stage_2.scheduler = DDIMInverseScheduler.from_config(stage_2.scheduler.config)
    noise_level_2=100

    image_tuple_2_inv, _ = stage_2(
                    prompt=args.prompt_str,
                    image=image_rec_1, 
                    noise_level=noise_level_2, 
                    generator=generator,
                    output_type="latent",
                    guidance_scale=1.0,
                    latents=latent,
                    num_inference_steps=100,
                    )
    image_2_inv=image_tuple_2_inv.images
    torch.cuda.empty_cache()
    stage_2.to('cpu')

    # stage 2 reconstraction
    stage_2.scheduler = DDIMScheduler.from_config(stage_2.scheduler.config)

    image_tuple_2_rec, _  = stage_2(
                    prompt=args.prompt_str,
                    image=image_rec_1, 
                    noise_level=noise_level_2, 
                    generator=generator,
                    output_type="pil",
                    guidance_scale=1.0,
                    num_inference_steps=100,
                    latents=image_2_inv
                    )

    image_2_rec=image_tuple_2_rec.images
    torch.cuda.empty_cache()
    stage_2.to('cpu')


    # stage 3 inversion  
    with torch.no_grad():
        latent = stage_3.prepare_image_latents(inv_raw_image_2.unsqueeze(0).to(device), 1, stage_3.vae.dtype, device,generator=generator)
    torch.cuda.empty_cache()

    stage_3.scheduler = DDIMInverseScheduler.from_config(stage_3.scheduler.config)
    noise_level_3=100

    image_tuple_3_inv, _ = stage_3(
                    prompt=args.prompt_str,
                    image=image_2_rec, 
                    noise_level=noise_level_3, 
                    generator=generator,
                    output_type="latent",
                    guidance_scale=1.0,
                    latents=latent,
                    num_inference_steps=100,
                    )
    image_3_inv=image_tuple_3_inv.images
    torch.cuda.empty_cache()
    stage_3.to('cpu')

    # stage 3 reconstraction
    stage_3.scheduler = DDIMScheduler.from_config(stage_3.scheduler.config)
    noise_level_3=100

    image_tuple_3_rec, _  = stage_3(
                    prompt=args.prompt_str,
                    image=image_2_rec, 
                    noise_level=noise_level_3, 
                    generator=generator,
                    output_type="pil",
                    guidance_scale=1.0,
                    num_inference_steps=100,
                    latents=image_3_inv
                    )

    image_3_rec=image_tuple_3_rec.images
    torch.cuda.empty_cache()
    stage_3.to('cpu')
    image_3_rec[0].save(f"{saving_path}_ddim_stage123_III_rec.png")

