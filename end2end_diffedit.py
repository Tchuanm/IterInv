
from pipelines.SDUP_inv_pipeline import StableDiffusionUpscaleInvPipeline
from pipelines.SDUP_pipeline import StableDiffusionUpscalePipeline

from pipelines.deepfloyd_pipeline import IFPipeline
from pipelines.deepfloyd_inv_pipeline import IFInvPipeline
from pipelines.deepfloyd_edit_pipeline import IFEditPipeline

from diffusers import StableDiffusionDiffEditPipeline
from pipelines.deepfloyd_edit_diffedit_pipeline import IFDiffEditPipeline

from pipelines.deepfloyd_SR_pipeline import IFSuperResolutionPipeline
from pipelines.deepfloyd_SR_inv_pipeline import IFSuperResolutionInvPipeline

from pipelines.scheduler_ddim import DDIMScheduler
from pipelines.scheduler_inv import DDIMInverseScheduler

from pipelines.pipeline_utils import get_mapper, get_replacement_mapper_
from diffusers.utils import pt_to_pil, numpy_to_pil
import torch
from IPython.display import display
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
import argparse
import os
import copy


### NOTE: image_processor can help to make the batch size ready
image_processor=VaeImageProcessor()

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='images/Plug-and-Play-datasets/imagenetr-ti2i/jeep/sculpture_0.jpg')
    parser.add_argument('--results_folder', type=str, default='output/all_imgs_inversion_in_prompt_file')
    # parser.add_argument('--prompt_file', type=str, default=None) 
    parser.add_argument('--prompt_str', type=str, default='a jeep')       # a painting of a kitten in a field of flowers
    parser.add_argument('--edit_prompt_str', type=str, default='a porsche sports car')
    parser.add_argument('--noise_level_3', type=int, default=100)
    parser.add_argument('--noise_level_2', type=int, default=250)

    parser.add_argument('--num_inference_steps_3', type=int, default=50)
    parser.add_argument('--num_inference_steps_2', type=int, default=50)
    parser.add_argument('--num_inference_steps_1', type=int, default=30)

    parser.add_argument('--guidance_1', type=float, default=3.0) 
    ### NOTE: 3.5 is good >=4.0 starts getting far away
    parser.add_argument('--guidance_2', type=float, default=1.0) 
    parser.add_argument('--guidance_3', type=float, default=1.0) 

    parser.add_argument('--lr_2', type=float, default=5e-3) 
    parser.add_argument('--lr_1', type=float, default=1e-3)
    parser.add_argument('--scale_factor_2', type=float, default=0.3) 
    parser.add_argument('--scale_factor_3', type=float, default=0.3) 

    parser.add_argument('--enable_1', action='store_true')
    parser.add_argument('--no_enable_1', dest='enable_1', action='store_false')
    # parser.set_defaults(enable_1=False)
    parser.set_defaults(enable_1=True)

    parser.add_argument('--enable_2', action='store_true')
    parser.add_argument('--no_enable_2', dest='enable_2', action='store_false')
    parser.set_defaults(enable_2=False)
    # parser.set_defaults(enable_2=True)

    parser.add_argument('--enable_3for2', action='store_true')
    parser.add_argument('--no_enable_3for2', dest='enable_3for2', action='store_false')
    # parser.set_defaults(enable_3for2=False)
    parser.set_defaults(enable_3for2=True)

    parser.add_argument('--enable_3', action='store_true')
    parser.add_argument('--no_enable_3', dest='enable_3', action='store_false')
    parser.set_defaults(enable_3=False)
    # parser.set_defaults(enable_3=True)

    parser.add_argument('--model_path_1', type=str, default="DeepFloyd/IF-I-M-v1.0")
    parser.add_argument('--model_path_2', type=str, default="DeepFloyd/IF-II-M-v1.0")
    parser.add_argument('--model_path_3', type=str, default="stabilityai/stable-diffusion-x4-upscaler")

    parser.add_argument('--is_NPI', action='store_true')
    parser.add_argument('--no_is_NPI', dest='is_NPI', action='store_false')
    # parser.set_defaults(is_NPI=True)
    parser.set_defaults(is_NPI=False)
    ## DiffEdit parameters 
    parser.add_argument('--inpaint_strength', type=float, default=0.8)
    parser.add_argument('--guidance_1_edit', type=float, default=3.0) 
    parser.add_argument('--mask_encode_strength', type=float, default=0.1) 
    parser.add_argument('--mask_thresholding_ratio', type=float, default=1) 

    # dog_1 0.4 3.0 0.1 1 
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arguments()
    print(args)
    if  args.enable_3 or  args.enable_3for2:
        stage_3_rec = StableDiffusionUpscalePipeline.from_pretrained(
                            args.model_path_3,
                            torch_dtype=torch.float32)
        stage_3_rec.scheduler = DDIMScheduler.from_config(stage_3_rec.scheduler.config)
        stage_3_rec.enable_model_cpu_offload()

    if args.enable_2:
        stage_2_rec = IFSuperResolutionPipeline.from_pretrained(
            args.model_path_2, text_encoder=None, variant="fp32", torch_dtype=torch.float32
        )
        stage_2_rec.scheduler = DDIMScheduler.from_config(stage_2_rec.scheduler.config)
        stage_2_rec.enable_model_cpu_offload()

    if args.enable_1:
        stage_1_edit = IFDiffEditPipeline.from_pretrained(args.model_path_1, variant="fp32", torch_dtype=torch.float32)
        stage_1_edit.scheduler = DDIMScheduler.from_config(stage_1_edit.scheduler.config)
        stage_1_edit.inverse_scheduler = DDIMInverseScheduler.from_config(stage_1_edit.scheduler.config)
        stage_1_edit.enable_model_cpu_offload()

    ### NOTE: Get the value from the arguments
    img_pth=args.input_image

    noise_level_3=args.noise_level_3
    noise_level_2=args.noise_level_2

    prompt = args.prompt_str
    edit_prompt = args.edit_prompt_str

    num_inference_steps_3=args.num_inference_steps_3
    num_inference_steps_2=args.num_inference_steps_2
    num_inference_steps_1=args.num_inference_steps_1
    
    scale_factor_2=args.scale_factor_2
    scale_factor_3=args.scale_factor_3

    lr_2=args.lr_2
    lr_1=args.lr_1

    guidance_1_edit = args.guidance_1_edit
    guidance_1=args.guidance_1
    guidance_2=args.guidance_2
    guidance_3=args.guidance_3

    is_NPI=args.is_NPI

    inpaint_strength = args.inpaint_strength

    ### NOTE: creating saving path
    bname = os.path.basename(args.input_image).split(".")[0]
    
    # ### NOTE: include prompt files if provides
    # if args.prompt_file is None:
    #     args.prompt_file=os.path.join(args.results_folder, f"{bname}", f"prompt.txt")
        
    # if os.path.isfile(args.prompt_file):
    #     prompt = open(args.prompt_file).read().strip()
    #     print(f'get prompt from file: {args.prompt_file} \n')
    # else:
    #     prompt = args.prompt_str
    #     print(f'get prompt from arguments \n')

    print(prompt)
    print(edit_prompt)
    print('\n')

    saving_path=os.path.join(args.results_folder, f"{bname}", 
                             f"CFG1_{guidance_1}_CFG3_{guidance_3}_noise3_{noise_level_3}_lr1_{lr_1}_scale3_{scale_factor_3}_NPI_{is_NPI}", 
                             '_'.join(prompt.split(' ')))
    
    print(f'the saving path is {saving_path}')
    os.makedirs(saving_path, exist_ok=True)
    os.makedirs(saving_path+'/DiffEdit', exist_ok=True)
    
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
        generator = torch.manual_seed(0)

        prompt_embeds, negative_embeds = stage_1_edit._encode_prompt(prompt).chunk(2)
        edit_prompt_embeds, _ = stage_1_edit._encode_prompt(edit_prompt).chunk(2)
        torch.save(negative_embeds, f"{saving_path}/diffedit_negative_embed.pt")
        torch.save(edit_prompt_embeds, f"{saving_path}/diffedit_edit_embed.pt")

        # inv_noise_1 = torch.load(f"{saving_path}/inv_noise_1.pt")

        guidance_scale = guidance_1_edit   #  guidance_1
        mask_image = stage_1_edit.generate_mask(image=inv_raw_image_0, 
                                                source_prompt=edit_prompt, 
                                                target_prompt=prompt,
                                                generator=generator,
                                                guidance_scale=guidance_scale,
                                                mask_encode_strength=args.mask_encode_strength,
                                                mask_thresholding_ratio=args.mask_thresholding_ratio,
                                                num_inference_steps=num_inference_steps_1,
                                                )
        image_inv_noise = stage_1_edit.invert(image_init=inv_raw_image_0, prompt=prompt, 
                                              inpaint_strength=inpaint_strength,
                                              num_inference_steps=num_inference_steps_1,
                                              generator = generator,
                                              guidance_scale=guidance_scale,
                                              ).images
        _, edit_image = stage_1_edit(
                            prompt=edit_prompt, 
                            mask_image=mask_image,
                            image_invs=image_inv_noise,
                            inpaint_strength=inpaint_strength,
                            num_inference_steps=num_inference_steps_1,
                            generator=generator,
                            output_type="pt",
                            guidance_scale=guidance_scale,
                            )

        stage_1_edit.to('cpu')
        torch.cuda.empty_cache()
        mask_image = np.repeat(mask_image, 3, axis=0)
        mask_image = torch.from_numpy(mask_image[np.newaxis, :])
        pt_to_pil(mask_image)[0].save(f"{saving_path}/diffedit_edit_mask_img.png")
        # pt_to_pil(image_inv_noise[0][0].unsqueeze(0))[0].save(f"{saving_path}/diffedit_edit_if_stage_last_noise.png")
        # pt_to_pil(image_inv_noise[0][-1].unsqueeze(0))[0].save(f"{saving_path}/diffedit_edit_if_stage_I_first_noise.png")
        pt_to_pil(edit_image)[0].save(f"{saving_path}/DiffEdit/{prompt}_2_{edit_prompt}_stage_I_inp_{inpaint_strength}_CFG_{guidance_1_edit}.png")

        torch.save(edit_image, f"{saving_path}/diffedit_edit_image.pt")
        stage_1_edit.to('cpu')
        del stage_1_edit 
        torch.cuda.empty_cache()

    ### NOTE: =================================================== 1st stage inversion END:

    if not args.enable_1:
        negative_embeds = torch.load(f'{saving_path}/diffedit_negative_embed.pt')
        edit_prompt_embeds = torch.load(f'{saving_path}/diffedit_edit_embed.pt')
    
    ### NOTE: =================================================== 2nd stage inversion start:
    if args.enable_2:
        image_2_inv = torch.load(f'{saving_path}/image_2_inv_scale2_{scale_factor_2}.pt')
        edit_image = torch.load(f'{saving_path}/diffedit_edit_image.pt')
        
        generator = torch.manual_seed(0)
        image_tuple_2_rec, _ = stage_2_rec(
                    image=edit_image, 
                    generator=generator,
                    # prompt=edit_prompt,
                    prompt_embeds=edit_prompt_embeds, 
                    negative_prompt_embeds=negative_embeds, 
                    output_type="pt",
                    noise_level=noise_level_2,
                    guidance_scale=1.0,
                    num_inference_steps=num_inference_steps_2,
                    image_init=image_2_inv
                    )
        stage_2_rec.to('cpu')
        torch.cuda.empty_cache()
        image_2_rec=image_tuple_2_rec.images
        pil_image_2_rec = pt_to_pil(image_2_rec)
        torch.save(image_2_rec.to('cpu').detach(), f'{saving_path}/diffedit_edit_image_2_rec_{scale_factor_2}.pt')
        pil_image_2_rec[0].save(f"{saving_path}/DiffEdit/{prompt}_2_{edit_prompt}_stage_II_edit_rec_{scale_factor_2}_CFG_{guidance_1_edit}.png")
        torch.cuda.empty_cache()


    if args.enable_3for2:
        generator = torch.manual_seed(0)
        with torch.no_grad():
            latent = stage_3_rec.prepare_image_latents(inv_raw_image_1.cuda(), 1, stage_3_rec.vae.dtype, 'cuda', generator=generator)
        latent_2_inv=torch.load(f'{saving_path}/latent_2_inv.pt')

        ### NOTE: just for reconstruction here. We will want the edit_image super resolution later.
        image_rec_1=torch.load(f"{saving_path}/diffedit_edit_image.pt")

        generator = torch.manual_seed(0)
        image_tuple_2_rec, _  = stage_3_rec(
                        prompt=edit_prompt,
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
        torch.save(image_2_rec.detach().to('cpu'), f"{saving_path}/diffedit_edit_image_2_rec.pt")
        pil_image_2_rec = pt_to_pil(image_2_rec)
        # pil_image_2_rec = image_processor.pt_to_numpy(image_2_rec)
        # pil_image_2_rec = image_processor.numpy_to_pil(pil_image_2_rec)
        pil_image_2_rec[0].save(f"{saving_path}/DiffEdit/{prompt}_2_{edit_prompt}_stage_III4II_inp_{inpaint_strength}_CFG_{guidance_1_edit}.png")
        torch.cuda.empty_cache()

    ### NOTE: =================================================== 2nd stage inversion END:

    ### NOTE: =================================================== 3rd stage inversion start:
    if args.enable_3:
        with torch.no_grad():
            latent = stage_3_rec.prepare_image_latents(inv_raw_image_2.cuda(), 1, stage_3_rec.vae.dtype, 'cuda', generator=generator)
        latent_3_inv=torch.load(f'{saving_path}/latent_3_inv.pt')

        if args.enable_3for2:
            image_2_rec=torch.load(f'{saving_path}/diffedit_edit_image_2_rec.pt')
        else:
            image_2_rec=torch.load(f'{saving_path}/diffedit_edit_image_2_rec_{scale_factor_2}.pt')

        generator = torch.manual_seed(0)
        image_tuple_3_rec, _  = stage_3_rec(
                        prompt=edit_prompt,
                        image=image_2_rec, 
                        noise_level=noise_level_3, 
                        generator=generator,
                        output_type="pil",
                        guidance_scale=1.0,
                        latents=latent_3_inv,
                        num_inference_steps=num_inference_steps_3,
                        )
        torch.cuda.empty_cache()
        stage_3_rec.to('cpu')
        image_2_rec=image_tuple_3_rec.images
        image_2_rec[0].save(f"{saving_path}/DiffEdit/{prompt}_2_{edit_prompt}_stage_III_inp_{inpaint_strength}_CFG_{guidance_1_edit}.png")
    ### NOTE: =================================================== 3rd stage inversion END:

