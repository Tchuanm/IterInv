from pipelines.SDUP_inv_pipeline import StableDiffusionUpscaleInvPipeline
from pipelines.SDUP_pipeline import StableDiffusionUpscalePipeline

from pipelines.deepfloyd_pipeline import IFPipeline
from pipelines.deepfloyd_inv_pipeline import IFInvPipeline
from pipelines.deepfloyd_edit_pipeline import IFEditPipeline
from pipelines.deepfloyd_edit_pix2pix_zero_pipeline import IFEditZeroPipeline

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



# Examples:
# >>> construct_direction("cat2dog")
def construct_direction(task_name):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    (src, dst) = task_name.split("2")
    emb_dir = f"assets/"             # f"assets/embeddings_sd_1.4"
    embs_a = torch.load(os.path.join(emb_dir, f"{src}.pt"), map_location=device)
    embs_b = torch.load(os.path.join(emb_dir, f"{dst}.pt"), map_location=device)
    return (embs_b.mean(0)-embs_a.mean(0)).unsqueeze(0)


### NOTE: image_processor can help to make the batch size ready
image_processor=VaeImageProcessor()

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default='images/pix2pix-zero/cat/cat_7.png')
    parser.add_argument('--results_folder', type=str, default='output/75imgs_inversion_in_prompt_file')
    parser.add_argument('--prompt_file', type=str, default=None) 
    parser.add_argument('--prompt_str', type=str, default='a painting of a kitten in a field of flowers')       # a painting of a kitten in a field of flowers
    parser.add_argument('--task_name', type=str, default='cat2dog')
    parser.add_argument('--edit_prompt_str', type=str, default='a painting of a puppy in a field of flowers')
    parser.add_argument('--noise_level_3', type=int, default=100)
    parser.add_argument('--noise_level_2', type=int, default=250)

    parser.add_argument('--num_inference_steps_3', type=int, default=50)
    parser.add_argument('--num_inference_steps_2', type=int, default=50)
    parser.add_argument('--num_inference_steps_1', type=int, default=50)

    parser.add_argument('--guidance_1', type=float, default=5.0) 
    ### NOTE: 3.5 is good >=4.0 starts getting far away
    parser.add_argument('--guidance_2', type=float, default=1.0) 
    parser.add_argument('--guidance_3', type=float, default=1.0) 

    parser.add_argument('--lr_2', type=float, default=3e-2) ### 0.01 < 0.05
    parser.add_argument('--lr_1', type=float, default=1e-3)
    parser.add_argument('--scale_factor_2', type=float, default=0.3) 
    parser.add_argument('--scale_factor_3', type=float, default=0.3) 

    parser.add_argument('--enable_1', action='store_true')
    parser.add_argument('--no_enable_1', dest='enable_1', action='store_false')
    # parser.set_defaults(enable_1=False)
    parser.set_defaults(enable_1=True)

    parser.add_argument('--enable_2', action='store_true')
    parser.add_argument('--no_enable_2', dest='enable_2', action='store_false')
    # parser.set_defaults(enable_2=False)
    parser.set_defaults(enable_2=True)

    parser.add_argument('--enable_3for2', action='store_true')
    parser.add_argument('--no_enable_3for2', dest='enable_3for2', action='store_false')
    parser.set_defaults(enable_3for2=False)
    # parser.set_defaults(enable_3for2=True)

    parser.add_argument('--enable_3', action='store_true')
    parser.add_argument('--no_enable_3', dest='enable_3', action='store_false')
    # parser.set_defaults(enable_3=False)
    parser.set_defaults(enable_3=True)

    parser.add_argument('--model_path_1', type=str, default="DeepFloyd/IF-I-M-v1.0")
    parser.add_argument('--model_path_2', type=str, default="DeepFloyd/IF-II-M-v1.0")
    parser.add_argument('--model_path_3', type=str, default="stabilityai/stable-diffusion-x4-upscaler")

    parser.add_argument('--is_NPI', action='store_true')
    parser.add_argument('--no_is_NPI', dest='is_NPI', action='store_false')
    # parser.set_defaults(is_NPI=True)
    parser.set_defaults(is_NPI=False)

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
        stage_1_edit = IFEditZeroPipeline.from_pretrained(args.model_path_1, variant="fp32", torch_dtype=torch.float32)
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

    guidance_1=args.guidance_1
    guidance_2=args.guidance_2
    guidance_3=args.guidance_3

    is_NPI=args.is_NPI

    ### NOTE: creating saving path
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
    print(edit_prompt)
    print('\n')

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
        prompt_embeds, negative_embeds = stage_1_edit.encode_prompt(prompt, negative_prompt=prompt)
        edit_prompt_embeds, _ = stage_1_edit.encode_prompt(edit_prompt)

        generator = torch.manual_seed(0)
        inv_noise_1 = torch.load(f"{saving_path}/inv_noise_1.pt")

        stage_1_edit.scheduler = DDIMScheduler.from_config(stage_1_edit.scheduler.config)

        generator = torch.manual_seed(0)
        output_rec_1, edit_image = stage_1_edit(
                        # prompt=prompt,
                        generator=generator, 
                        num_inference_steps=num_inference_steps_1,
                        output_type="pt",
                        image_init=inv_noise_1,
                        guidance_scale=guidance_1,     # guidance_1
                        #new params
                        # negative_prompt=prompt,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_embeds,
                        edit_dir = construct_direction(args.task_name),
                        )
        image_rec_1 = output_rec_1.images
        stage_1_edit.to('cpu')
        torch.cuda.empty_cache()
        pt_to_pil(image_rec_1)[0].save(f"{saving_path}/p2p_zero_edit_if_stage_I_rec.png")
        pt_to_pil(edit_image)[0].save(f"{saving_path}/p2p_zero_edit_if_stage_I_output.png")

        torch.save(image_rec_1, f"{saving_path}/p2p_zero_edit_image_rec_1.pt")
        torch.save(edit_image, f"{saving_path}/p2p_zero_edit_image.pt")
        torch.cuda.empty_cache()
    ### NOTE: =================================================== 1st stage inversion END:


    ### NOTE: =================================================== 2nd stage inversion start:
    if args.enable_2:

        image_2_inv = torch.load(f'{saving_path}/image_2_inv_scale2_{scale_factor_2}.pt')
        
        edit_image = torch.load(f'{saving_path}/p2p_zero_edit_image.pt')

        generator = torch.manual_seed(0)
        image_tuple_2_rec, _ = stage_2_rec(
                    image=edit_image, 
                    generator=generator,
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
        torch.save(image_2_rec.to('cpu').detach(), f'{saving_path}/p2p_zero_edit_image_2_rec_{scale_factor_2}.pt')
        pil_image_2_rec[0].save(f"{saving_path}/p2p_zero_if_stage_II_edit_rec_{scale_factor_2}.png")
        torch.cuda.empty_cache()

    ### NOTE: =================================================== 2nd stage inversion END:

    ### NOTE: =================================================== 3rd stage inversion start:
    if args.enable_3:
        with torch.no_grad():
            latent = stage_3_rec.prepare_image_latents(inv_raw_image_2.cuda(), 1, stage_3_rec.vae.dtype, 'cuda', generator=generator)
        latent_3_inv=torch.load(f'{saving_path}/latent_3_inv.pt')

        if args.enable_3for2:
            image_2_rec=torch.load(f'{saving_path}/p2p_zero_edit_image_2_rec.pt')
        else:
            image_2_rec=torch.load(f'{saving_path}/p2p_zero_edit_image_2_rec_{scale_factor_2}.pt')

        generator = torch.manual_seed(0)
        image_tuple_3_rec, latent_3_rec  = stage_3_rec(
                        # prompt=prompt,
                        prompt=edit_prompt,
                        # image=inv_raw_image_1, 
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
        image_2_rec[0].save(f"{saving_path}/p2p_zero_edit_if_stage_III_rec.png")
    ### NOTE: =================================================== 3rd stage inversion END:

