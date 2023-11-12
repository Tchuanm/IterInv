from pipelines.SDUP_inv_pipeline import StableDiffusionUpscaleInvPipeline
from pipelines.SDUP_pipeline import StableDiffusionUpscalePipeline

from pipelines.deepfloyd_pipeline import IFPipeline
from pipelines.deepfloyd_inv_pipeline import IFInvPipeline
from pipelines.deepfloyd_edit_pipeline import IFEditPipeline

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
    parser.add_argument('--input_image', type=str, default='images/pix2pix-zero/cat/cat_5.png')
    parser.add_argument('--results_folder', type=str, default='output/75imgs_inversion_in_prompt_file')
    parser.add_argument('--prompt_file', type=str, default=None)
    parser.add_argument('--prompt_str', type=str, default='a white cat with a red bow tie')
    parser.add_argument('--edit_prompt_str', type=str, default='a white dog with a red bow tie')

    parser.add_argument('--noise_level_3', type=int, default=100)
    parser.add_argument('--noise_level_2', type=int, default=250)

    parser.add_argument('--num_inference_steps_3', type=int, default=50)
    parser.add_argument('--num_inference_steps_2', type=int, default=50)
    parser.add_argument('--num_inference_steps_1', type=int, default=50)

    parser.add_argument('--inner_steps_3', type=int, default=21)
    parser.add_argument('--inner_steps_2', type=int, default=51)
    parser.add_argument('--inner_steps_1', type=int, default=51)

    parser.add_argument('--guidance_1', type=float, default=3.0) 
    ### NOTE: 3.5 is good >=4.0 starts getting far away
    parser.add_argument('--guidance_2', type=float, default=1.0) 
    parser.add_argument('--guidance_3', type=float, default=1.0) 

    parser.add_argument('--lr_2', type=float, default=3e-2) ### 0.01 < 0.05
    parser.add_argument('--lr_1', type=float, default=1e-3)
    parser.add_argument('--scale_factor_2', type=float, default=0.4) 
    parser.add_argument('--scale_factor_3', type=float, default=0.3) 

    parser.add_argument('--manual_prompt', action='store_true')
    parser.add_argument('--no-manual_prompt', dest='manual_prompt', action='store_false')
    # parser.set_defaults(manual_prompt=False)
    parser.set_defaults(manual_prompt=True)

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

    ### NOTE: =============================== image editing hyperparameters ===============================
    parser.add_argument('--cross_replace_steps', type=int, default=40)
    parser.add_argument('--self_replace_steps', type=int, default=40)

    ### NOTE: change this part into parameters
    parser.add_argument('--indices_to_alter', nargs='+', type=int, default=[2,7])

    ### ATTN: 1st: p2p hyperparameters REFINE: global edit ; 
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--no-refine', dest='refine', action='store_false')
    parser.set_defaults(refine=False)
    # parser.set_defaults(refine=True)

    ### ATTN: 2nd: REPLACE: word swap
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--no-replace', dest='replace', action='store_false')
    parser.set_defaults(replace=True)
    # parser.set_defaults(replace=False)
    ### NOTE: find the original word and replace by the new word
    parser.add_argument('--original_words', nargs='+', type=str, default=['cat'])
    # parser.add_argument('--original_words', nargs='+', type=str, default=None)
    parser.add_argument('--replace_words', nargs='+', type=str, default=['leopard'])
    
    ### ATTN: 3rd: local edit
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--no-local', dest='local', action='store_false')
    # parser.set_defaults(local=False)
    parser.set_defaults(local=True)
    parser.add_argument('--indices_local', nargs='+', type=int, default=[2,])
    # parser.add_argument('--indices_local', nargs='+', type=int, default=None)
    
    ### ATTN: 4th: reweight
    parser.add_argument('--indices_to_amplify', nargs='+', type=int, default=[2,])
    # parser.add_argument('--indices_to_amplify', nargs='+', type=int, default=None)
    parser.add_argument('--amplify_scale', nargs='+', type=float, default=[1.0,1.0])

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arguments()
    print(args)
    # gpuid = "7"
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

    if  args.enable_3 or args.enable_3for2:
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

    if args.enable_1 or args.enable_2:
        stage_1_edit = IFEditPipeline.from_pretrained(args.model_path_1, variant="fp32", torch_dtype=torch.float32)
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

    is_NPI=args.is_NPI

    ### NOTE: creating saving path
    bname = os.path.basename(args.input_image).split(".")[0]
    
    ### NOTE: include prompt files if provides
    if args.prompt_file is None:
        args.prompt_file=os.path.join(args.results_folder, f"{bname}", f"prompt.txt")
        
    if os.path.isfile(args.prompt_file) and not args.manual_prompt:
        prompt = open(args.prompt_file).read().strip()
        print(f'get prompt from file: {args.prompt_file} \n')
    else:
        prompt = args.prompt_str
        print(f'get prompt from arguments \n')

    ### NOTE: get the edit caption            
    if args.replace and (args.original_words is not None):
        caption_list=prompt.split(' ')
        edit_caption_list=copy.deepcopy(caption_list)
        print('replace the old word with new word \n')
        for replace_id in range(len(args.original_words)):
            org_word = args.original_words[replace_id]
            if org_word in caption_list: 
                org_word_id = caption_list.index(org_word)
            else:
                continue
            edit_caption_list[org_word_id] = args.replace_words[replace_id]
        edit_prompt=' '.join(edit_caption_list)

    ### NOTE: ==============================================================================
    print(prompt)
    print(edit_prompt)
    print('\n')

    if args.refine:
        tokenizer=stage_1_edit.tokenizer
        mapper, alphas = get_mapper(prompt, edit_prompt, tokenizer)
        mapper, alphas = mapper.cuda(), alphas.cuda()
    else:
        mapper, alphas = None, None

    saving_path=os.path.join(args.results_folder, f"{bname}", 
                             f"CFG1_{guidance_1}_CFG3_{guidance_3}_noise3_{noise_level_3}_lr1_{lr_1}_scale3_{scale_factor_3}_NPI_{is_NPI}", 
                             '_'.join(prompt.split(' ')))
    
    print(f'the saving path is {saving_path}')
    os.makedirs(saving_path, exist_ok=True)
    edit_saving_path = os.path.join(saving_path, 'edit', '_'.join(edit_prompt.split(' ')))
    os.makedirs(edit_saving_path, exist_ok=True)
    
    ### NOTE: read the images
    _inv_raw_image_2 = Image.open(img_pth).convert("RGB").resize((1024,1024))
    inv_raw_image_2 = image_processor.preprocess(_inv_raw_image_2)

    _inv_raw_image_1 = Image.open(img_pth).convert("RGB").resize((256,256))
    inv_raw_image_1 = image_processor.preprocess(_inv_raw_image_1)
    
    _inv_raw_image_0 = Image.open(img_pth).convert("RGB").resize((64,64))
    inv_raw_image_0 = image_processor.preprocess(_inv_raw_image_0)
    
    generator = torch.manual_seed(0)

    if args.enable_1:
        prompt_embeds, negative_embeds = stage_1_edit.encode_prompt(prompt)
        edit_prompt_embeds, _ = stage_1_edit.encode_prompt(edit_prompt)
        generator = torch.manual_seed(0)
        
        inv_noise_1 = torch.load(f"{saving_path}/inv_noise_1.pt")
        uncond_embeddings_list = torch.load(f"{saving_path}/uncond_embeddings_list.pt")

        stage_1_edit.scheduler = DDIMScheduler.from_config(stage_1_edit.scheduler.config)

        generator = torch.manual_seed(0)
        output_rec_1, edit_image, inter_img_list_1_edit_rec = stage_1_edit(
                        prompt_embeds=prompt_embeds, 
                        edit_prompt_embeds=edit_prompt_embeds,  ########## New parameters
                        negative_prompt_embeds=negative_embeds, 
                        generator=generator, 
                        num_inference_steps=num_inference_steps_1,
                        output_type="pt",
                        image_init=inv_noise_1,
                        guidance_scale=guidance_1,
                        uncond_embeddings_list=uncond_embeddings_list,
                        cross_replace_steps=args.cross_replace_steps,
                        self_replace_steps=args.self_replace_steps,
                        local=args.local,
                        indices_local=args.indices_local,
                        amplify_scale=args.amplify_scale,
                        indices_to_amplify=args.indices_to_amplify,
                        mapper = mapper, 
                        alphas = alphas,
                        refine=args.refine,
                        replace=args.replace,
                        )
        image_rec_1 = output_rec_1.images
        stage_1_edit.to('cpu')

        pt_to_pil(image_rec_1)[0].save(f"{edit_saving_path}/edit_if_stage_I_rec.png")
        pt_to_pil(edit_image)[0].save(f"{edit_saving_path}/edit_if_stage_I_output.png")

        torch.save(image_rec_1, f"{edit_saving_path}/edit_image_rec_1.pt")
        torch.save(edit_image, f"{edit_saving_path}/edit_image.pt")

    ### NOTE: =================================================== 1st stage inversion END:


    ### NOTE: =================================================== 2nd stage inversion start:
    if args.enable_2:

        image_2_inv = torch.load(f'{saving_path}/image_2_inv_scale2_{scale_factor_2}.pt')
        edit_image = torch.load(f'{edit_saving_path}/edit_image.pt')

        generator = torch.manual_seed(0)
        image_tuple_2_rec, _ = stage_2_rec(
                    image=edit_image, 
                    generator=generator,
                    prompt_embeds=prompt_embeds, 
                    negative_prompt_embeds=negative_embeds, 
                    output_type="pt",
                    noise_level=noise_level_2,
                    guidance_scale=1.0,
                    num_inference_steps=num_inference_steps_2,
                    image_init=image_2_inv
                    )
        stage_2_rec.to('cpu')

        image_2_rec=image_tuple_2_rec.images
        pil_image_2_rec = pt_to_pil(image_2_rec)
        torch.save(image_2_rec.to('cpu').detach(), f'{edit_saving_path}/edit_image_2_rec_{scale_factor_2}.pt')
        pil_image_2_rec[0].save(f"{edit_saving_path}/if_stage_II_edit_rec_{scale_factor_2}.png")
        torch.cuda.empty_cache()

    ### NOTE: =================================================== 2nd stage inversion END:

    ### NOTE: =================================================== 2nd stage inversion start:

    if args.enable_3for2:
        with torch.no_grad():
            latent = stage_3_rec.prepare_image_latents(inv_raw_image_1.cuda(), 1, stage_3_rec.vae.dtype, 'cuda', generator=generator)

        latent_2_inv=torch.load(f'{saving_path}/latent_2_inv.pt')

        ### NOTE: just for reconstruction here. We will want the edit_image super resolution later.
        # if not args.enable_1:
        #     # image_rec_1=torch.load(f"{saving_path}/image_rec_1.pt")
        #     image_rec_1=torch.load(f"{saving_path}/edit_image.pt")
        # else:
        #     # image_rec_1=torch.load(f"{saving_path}/edit_image_rec_1.pt")
        image_rec_1=torch.load(f"{saving_path}/edit_image.pt")

        generator = torch.manual_seed(0)
        image_tuple_2_rec, latent_2_rec  = stage_3_rec(
                        prompt=prompt,
                        # prompt=edit_prompt,
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
        torch.save(image_2_rec.detach().to('cpu'), f"{edit_saving_path}/edit_image_2_rec.pt")
        pil_image_2_rec = pt_to_pil(image_2_rec)
        # pil_image_2_rec = image_processor.pt_to_numpy(image_2_rec)
        # pil_image_2_rec = image_processor.numpy_to_pil(pil_image_2_rec)
        pil_image_2_rec[0].save(f"{edit_saving_path}/edit_if_stage_III4II_rec.png")
        torch.cuda.empty_cache()

    ### NOTE: =================================================== 2nd stage inversion END:

    ### NOTE: =================================================== 3rd stage inversion start:
    if args.enable_3:
        with torch.no_grad():
            latent = stage_3_rec.prepare_image_latents(inv_raw_image_2.cuda(), 1, stage_3_rec.vae.dtype, 'cuda', generator=generator)

        latent_3_inv=torch.load(f'{saving_path}/latent_3_inv.pt')

        if args.enable_3for2:
            image_2_rec=torch.load(f'{edit_saving_path}/edit_image_2_rec.pt')
        else:
            image_2_rec=torch.load(f'{edit_saving_path}/edit_image_2_rec_{scale_factor_2}.pt')

        generator = torch.manual_seed(0)
        image_tuple_3_rec, latent_3_rec  = stage_3_rec(
                        prompt=prompt,
                        # prompt=edit_prompt,
                        # image=inv_raw_image_1, 
                        image=image_2_rec, 
                        noise_level=noise_level_3, 
                        generator=generator,
                        output_type="pil",
                        guidance_scale=1.0,
                        latents=latent_3_inv,
                        num_inference_steps=num_inference_steps_3,
                        )
        
        stage_3_rec.to('cpu')
        image_2_rec=image_tuple_3_rec.images
        image_2_rec[0].save(f"{edit_saving_path}/edit_if_stage_III_rec.png")
    ### NOTE: =================================================== 3rd stage inversion END:

