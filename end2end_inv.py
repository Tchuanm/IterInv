from pipelines.SDUP_inv_pipeline import StableDiffusionUpscaleInvPipeline
from pipelines.SDUP_pipeline import StableDiffusionUpscalePipeline

from pipelines.deepfloyd_pipeline import IFPipeline
from pipelines.deepfloyd_inv_pipeline import IFInvPipeline

from pipelines.deepfloyd_SR_pipeline import IFSuperResolutionPipeline
from pipelines.deepfloyd_SR_inv_pipeline import IFSuperResolutionInvPipeline
# from pipelines.deepfloyd_SR_nti_pipeline import IFSuperResolutionInvPipeline

from pipelines.scheduler_ddim import DDIMScheduler
from pipelines.scheduler_inv import DDIMInverseScheduler

from diffusers.utils import pt_to_pil, numpy_to_pil
import torch
from IPython.display import display
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import argparse
import os

### NOTE: image_processor can help to make the batch size ready
image_processor=VaeImageProcessor()

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='images/cat_hq.jpg')
    parser.add_argument('--results_folder', type=str, default='output/')
    # parser.add_argument('--seg_dirs', type=str, default='seg_dirs/catdog')
    parser.add_argument('--prompt_str', type=str, default='a cat in an astronaut suit')
    parser.add_argument('--prompt_file', type=str, default=None)

    ### NOTE: noise level are also hyperparameters
    parser.add_argument('--noise_level_3', type=int, default=100)
    parser.add_argument('--noise_level_2', type=int, default=250)

    parser.add_argument('--num_inference_steps_3', type=int, default=50)
    parser.add_argument('--num_inference_steps_2', type=int, default=50)
    parser.add_argument('--num_inference_steps_1', type=int, default=50)

    parser.add_argument('--inner_steps_3', type=int, default=21)
    parser.add_argument('--inner_steps_2', type=int, default=21)
    parser.add_argument('--inner_steps_1', type=int, default=51)

    parser.add_argument('--scale_factor_2', type=float, default=0.3) 
    parser.add_argument('--scale_factor_3', type=float, default=0.3) 

    # parser.add_argument('--lr_3', type=float, default=51)
    # parser.add_argument('--lr_2', type=float, default=1e-1) 
    parser.add_argument('--lr_2', type=float, default=5e-3) 
    ### NOTE: lr_2: 0.005<= lr_2 <= 0.05 with 101 steps and 100 inference time steps ATTN: not NTI case
    parser.add_argument('--lr_1', type=float, default=1e-3)
    
    parser.add_argument('--guidance_1', type=float, default=3.0) 
    ### NOTE: 3.5 is good >=4.0 starts getting far away
    parser.add_argument('--guidance_2', type=float, default=1.0) 
    parser.add_argument('--guidance_3', type=float, default=1.0) 

    parser.add_argument('--model_path_1', type=str, default="DeepFloyd/IF-I-M-v1.0")
    parser.add_argument('--model_path_2', type=str, default="DeepFloyd/IF-II-M-v1.0")
    parser.add_argument('--model_path_3', type=str, default="stabilityai/stable-diffusion-x4-upscaler")

    parser.add_argument('--enable_1', action='store_true')
    parser.add_argument('--no_enable_1', dest='enable_1', action='store_false')
    parser.set_defaults(enable_1=False)

    parser.add_argument('--enable_3', action='store_true')
    parser.add_argument('--no_enable_3', dest='enable_3', action='store_false')
    parser.set_defaults(enable_3=False)
    # parser.set_defaults(enable_3=True)

    parser.add_argument('--enable_3for2', action='store_true')
    parser.add_argument('--no_enable_3for2', dest='enable_3for2', action='store_false')
    parser.set_defaults(enable_3for2=False)
    # parser.set_defaults(enable_3for2=True)

    parser.add_argument('--enable_3_float16', action='store_true')
    parser.add_argument('--no_enable_3_float16', dest='enable_3_float16', action='store_false')
    parser.set_defaults(enable_3_float16=False)
    # parser.set_defaults(enable_3_float16=True)

    parser.add_argument('--is_NPI', action='store_true')
    parser.add_argument('--no_is_NPI', dest='is_NPI', action='store_false')
    # parser.set_defaults(is_NPI=True)
    parser.set_defaults(is_NPI=False)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arguments()
    print(args)

    ### NOTE: upscaler can only be float 32 to avoid overflow 
    if  args.enable_3 or args.enable_3for2:
        stage_3 = StableDiffusionUpscaleInvPipeline.from_pretrained(
                            args.model_path_3,
                            torch_dtype=torch.float32)
        stage_3.scheduler = DDIMScheduler.from_config(stage_3.scheduler.config)

        ### NOTE: support float 16 for guidance_3>1.0 or modular them to see if help or move to 48GB GPUs
        # if not args.enable_3_float16:
        stage_3_rec = StableDiffusionUpscalePipeline.from_pretrained(
                            args.model_path_3,
                            torch_dtype=torch.float32)
            
        stage_3_rec.scheduler = DDIMScheduler.from_config(stage_3_rec.scheduler.config)

        stage_3.enable_model_cpu_offload()
        stage_3_rec.enable_model_cpu_offload()

    if args.enable_1:
        stage_1 = IFInvPipeline.from_pretrained(args.model_path_1, variant="fp32", torch_dtype=torch.float32)
        stage_1.enable_model_cpu_offload()

    ### NOTE: Get the value from the arguments
    img_pth=args.input_image

    noise_level_3=args.noise_level_3
    noise_level_2=args.noise_level_2

    bname = os.path.basename(args.input_image).split(".")[0]
    
    ### NOTE: include prompt files if provides
    if args.prompt_file is None:
        args.prompt_file=os.path.join(args.results_folder, f"{bname}", f"prompt.txt")
        
    if os.path.isfile(args.prompt_file):
        prompt = open(args.prompt_file).read().strip()
        print(f'get prompt from file: {args.prompt_file} \n')
    else:
        prompt = args.prompt_str
        print(f'get prompt from arguments \n')

    print(prompt)

    num_inference_steps_3=args.num_inference_steps_3
    num_inference_steps_2=args.num_inference_steps_2
    num_inference_steps_1=args.num_inference_steps_1

    inner_steps_3=args.inner_steps_3
    inner_steps_2=args.inner_steps_2
    inner_steps_1=args.inner_steps_1
    
    scale_factor_2=args.scale_factor_2
    scale_factor_3=args.scale_factor_3

    lr_2=args.lr_2
    lr_1=args.lr_1

    guidance_1=args.guidance_1
    guidance_2=args.guidance_2
    guidance_3=args.guidance_3

    ### NOTE: negative prompt inversion
    is_NPI=args.is_NPI

    ### NOTE: creating saving path
    saving_path=os.path.join(args.results_folder, f"{bname}", 
                             f"CFG1_{guidance_1}_CFG3_{guidance_3}_noise3_{noise_level_3}_lr1_{lr_1}_scale3_{scale_factor_3}_NPI_{is_NPI}", 
                             '_'.join(prompt.split(' ')))

    print(f'the saving path is {saving_path}')
    os.makedirs(saving_path, exist_ok=True)
    
    ### NOTE: read the images
    _inv_raw_image_2 = Image.open(img_pth).convert("RGB").resize((1024,1024))
    inv_raw_image_2 = image_processor.preprocess(_inv_raw_image_2)

    _inv_raw_image_1 = Image.open(img_pth).convert("RGB").resize((256,256))
    inv_raw_image_1 = image_processor.preprocess(_inv_raw_image_1)
    
    _inv_raw_image_0 = Image.open(img_pth).convert("RGB").resize((64,64))
    inv_raw_image_0 = image_processor.preprocess(_inv_raw_image_0)
    
    generator = torch.manual_seed(0)

    ### NOTE: =================================================== 1st stage inversion start:
    if args.enable_1:
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
        generator = torch.manual_seed(0)
        stage_1.scheduler = DDIMInverseScheduler.from_config(stage_1.scheduler.config)

        output_inv_1, inter_img_list_1_inv, _ = stage_1(
                            prompt_embeds=prompt_embeds, 
                            negative_prompt_embeds=negative_embeds, 
                            generator=generator, 
                            num_inference_steps=num_inference_steps_1,
                            output_type="pt",
                            image_init=inv_raw_image_0,
                            guidance_scale=1.0,
                            )

        inv_noise_1 = output_inv_1.images
        torch.save(inv_noise_1.cpu(),f"{saving_path}/inv_noise_1.pt")
        torch.save(inter_img_list_1_inv, f"{saving_path}/inter_img_list_1_inv.pt")
        torch.cuda.empty_cache()

        #################### Recon =========================
        stage_1.scheduler = DDIMScheduler.from_config(stage_1.scheduler.config)

        generator = torch.manual_seed(0)
        output_rec_1, _, uncond_embeddings_list = stage_1(
                        prompt_embeds=prompt_embeds, 
                        negative_prompt_embeds=negative_embeds, 
                        generator=generator, 
                        num_inference_steps=num_inference_steps_1,
                        output_type="pt",
                        image_init=inv_noise_1,
                        guidance_scale=guidance_1,
                        all_latents=inter_img_list_1_inv,
                        null_inner_steps = inner_steps_1, 
                        is_NPI=is_NPI,
                        learning_rate=lr_1,
                        )
        image_rec_1 = output_rec_1.images
        stage_1.to('cpu')

        pt_to_pil(image_rec_1)[0].save(f"{saving_path}/if_stage_I_rec.png")
        torch.save(image_rec_1.detach().to('cpu'), f"{saving_path}/image_rec_1.pt")
        torch.save(uncond_embeddings_list, f"{saving_path}/uncond_embeddings_list.pt")
        # torch.save(inter_img_list_1_rec, f"{saving_path}/inter_img_list_1_rec.pt")

        del stage_1
        torch.cuda.empty_cache()

    ### NOTE: =================================================== 1st stage inversion END:

    # ### NOTE: =================================================== 2nd stage inversion start:  upscaler for stage 2
    if args.enable_3for2:
        with torch.no_grad():
            latent = stage_3.prepare_image_latents(inv_raw_image_1.cuda(), 1, stage_3.vae.dtype, 'cuda', generator=generator)
            
        torch.cuda.empty_cache()

        stage_3.scheduler = DDIMScheduler.from_config(stage_3.scheduler.config)
        image_rec_1 = torch.load(f"{saving_path}/image_rec_1.pt")

        generator = torch.manual_seed(0)
        latent_tuple_2_inv, _ = stage_3(
                        prompt=prompt,
                        image=image_rec_1, 
                        noise_level=noise_level_3, 
                        generator=generator,
                        output_type="latent",
                        guidance_scale=guidance_3,
                        latents=latent.float(),
                        scale_factor=scale_factor_3,
                        inner_steps=inner_steps_3,
                        num_inference_steps=num_inference_steps_3,
                        )
        # ### NOTE: 0.3 <= loss scale_factor <= 0.5
        stage_3.to('cpu')
        torch.cuda.empty_cache()
        latent_2_inv=latent_tuple_2_inv.images

        torch.save(latent_2_inv.to('cpu').detach(), f'{saving_path}/latent_2_inv.pt')
        # torch.save(latent_2_inv_list, f'{saving_path}/latent_2_inv_list.pt')

        generator = torch.manual_seed(0)
        # image_tuple_2_rec, latent_2_rec  = stage_3_rec(
        image_tuple_2_rec, _  = stage_3_rec(
                        prompt=prompt,
                        image=image_rec_1, 
                        noise_level=noise_level_3, 
                        generator=generator,
                        output_type="pt",
                        guidance_scale=guidance_3,
                        latents=latent_2_inv,
                        num_inference_steps=num_inference_steps_3,
                        )
        
        stage_3_rec.to('cpu')
        torch.cuda.empty_cache()
        
        image_2_rec=image_tuple_2_rec.images
        torch.save(image_2_rec.detach().to('cpu'), f"{saving_path}/image_2_rec.pt")
        pil_image_2_rec = pt_to_pil(image_2_rec)
        # pil_image_2_rec = image_processor.pt_to_numpy(image_2_rec)
        # pil_image_2_rec = image_processor.numpy_to_pil(pil_image_2_rec)
        pil_image_2_rec[0].save(f"{saving_path}/if_stage_III4II_rec.png")

    ### NOTE: =================================================== 2nd stage inversion END:

    # ### NOTE: =================================================== 3rd stage inversion start:
    if args.enable_3:
        image_2_rec = torch.load(f"{saving_path}/image_2_rec.pt")

        with torch.no_grad():
            latent = stage_3.prepare_image_latents(inv_raw_image_2.cuda(), 1, stage_3.vae.dtype, 'cuda', generator=generator)
        
        torch.cuda.empty_cache()

        stage_3.scheduler = DDIMScheduler.from_config(stage_3.scheduler.config)

        generator = torch.manual_seed(0)
        latent_tuple_3_inv, _ = stage_3(
                        prompt=prompt,
                        image=image_2_rec, 
                        noise_level=noise_level_3, 
                        generator=generator,
                        output_type="latent",
                        guidance_scale=guidance_3,
                        latents=latent.float(),
                        scale_factor=scale_factor_3,
                        inner_steps=inner_steps_3,
                        num_inference_steps=num_inference_steps_3,
                        )
        # ### NOTE: 0.3 <= loss scale_factor <= 0.5. >=0.7 not good
        stage_3.to('cpu')
        latent_3_inv=latent_tuple_3_inv.images

        torch.save(latent_3_inv.to('cpu').detach(), f'{saving_path}/latent_3_inv.pt')
        # torch.save(latent_list, f'{saving_path}/latent_3_inv_list.pt')
        torch.cuda.empty_cache()

        generator = torch.manual_seed(0)
        image_tuple_3_rec, latent_3_rec  = stage_3_rec(
                        prompt=prompt,
                        image=image_2_rec, 
                        noise_level=noise_level_3, 
                        generator=generator,
                        output_type="pil",
                        guidance_scale=guidance_3,
                        latents=latent_3_inv,
                        num_inference_steps=num_inference_steps_3,
                        )
        
        stage_3_rec.to('cpu')
        image_3_rec=image_tuple_3_rec.images
        image_3_rec[0].save(f"{saving_path}/if_stage_III_rec.png")
    ### NOTE: =================================================== 3rd stage inversion END:
    ### TODO: modulizing them into functions and save something (latents) for comparison.