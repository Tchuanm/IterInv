from PIL import Image
import torchvision.transforms as transforms
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.multimodal import CLIPScore
from torchmetrics import PeakSignalNoiseRatio
# from _utils.extractor import VitExtractor
import numpy as np

# torch.set_default_device(7)

dino_preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
# vit_extractor = VitExtractor('dino_vitb8', 'cuda')

psnr = PeakSignalNoiseRatio()
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)



from PIL import Image
import torch
from torchvision import transforms

org_transform = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize(1024), 
    transforms.ToTensor(),
])


### NOTE: set the img paths of each from the prompt txt. 
root_pth = '/disk1/users/ctang/deepfloyd/'
folder = 'all_imgs_inversion_in_prompt_file'
# choose the paramter settings of ours. 
ddim_stages = 'stages23'
# our_setting = 'CFG1_3.0_CFG3_1.0_noise3_100_lr1_0.001_scale3_0.3_NPI_False' 
our_setting = 'CFG1_3.0_CFG3_1.0_noise3_100_lr1_0.001_scale3_0.3_NPI_True'


ddim_eval_dict={'lpips':0, 'ssim':0, 'clip_score':0, 'psnr':0,  'mse':0 }
ours_eval_dict={'lpips':0, 'ssim':0, 'clip_score':0, 'psnr':0, 'mse':0 }

count_imgs = 0 
with open('prompt_all_imgs.txt', 'r') as file:
    for line in file: 
        ori_img_pth, prompt = line.strip().split(': ')

        image_name = ori_img_pth.split('/')[-1].split('.')[0]  # cat_6       
        _prompt = '_'.join(prompt.split(' '))
        
        ori_img_pth = root_pth + ori_img_pth
        ours_img_path = root_pth+ f'output/{folder}/{image_name}/{our_setting}/{_prompt}/if_stage_III_rec.png'
        # ddim_img_path = root_pth+ f'DDIM_output/only_stage3/{prompt}_if_stage_III_rec.png'
        # ddim_img_path = root_pth+ f'DDIM_output/{ddim_stages}/{prompt}_ddim_stage123_III_rec.png'
        
        # ours_img_path = root_pth+ f'output/stage2_inv/stage123_a_object_scale_0.3/{image_name}.png'
        ddim_img_path = root_pth+ f'output/SDXL_rec/{image_name}.png'

        ddim_image = Image.open(ddim_img_path)
        our_image = Image.open(ours_img_path)
        ori_img = Image.open(ori_img_pth)

        ddim_image_tensor = transforms.ToTensor()(ddim_image)
        our_image_tensor = transforms.ToTensor()(our_image)
        ori_img_tensor = org_transform(ori_img)

        # evaluate_score_for_one_img(ori_img_tensor, prompt, our_image_tensor, ddim_image_tensor, ours_eval_dict, ddim_eval_dict)
        count_imgs += 1  

        # 1. LPIPS scores, lower is better. 
        ori_ours_lpips_score = lpips(ori_img_tensor.unsqueeze(0)*2-1, our_image_tensor.unsqueeze(0)*2-1)
        ori_ddim_lpips_score = lpips(ori_img_tensor.unsqueeze(0)*2-1, ddim_image_tensor.unsqueeze(0)*2-1)
        ddim_eval_dict['lpips']+=ori_ddim_lpips_score.detach()
        ours_eval_dict['lpips']+=ori_ours_lpips_score.detach()
        # print('LPIPS scores:', ori_ours_lpips_score.detach(), ori_ddim_lpips_score.detach())

        # 2. SSIM scores, bigger is better.
        ssim_ddim =ssim(ddim_image_tensor.unsqueeze(0),ori_img_tensor.unsqueeze(0))
        ssim_ours =ssim(our_image_tensor.unsqueeze(0),ori_img_tensor.unsqueeze(0))
        ddim_eval_dict['ssim']+=ssim_ddim
        ours_eval_dict['ssim']+=ssim_ours
        # print('SSIM scores:', ssim_ours, ssim_ddim)

        # 3. CLIP score, bigger is better ? 
        clip_ddim = metric(ddim_image_tensor,  prompt)
        clip_ours = metric(our_image_tensor,  prompt)
        ddim_eval_dict['clip_score']+=clip_ddim
        ours_eval_dict['clip_score']+=clip_ours
        # print("CLIP score:", clip_ours.detach(), clip_ddim.detach())

        # 4. PSNR scores, bigger is better.
        psnr_ddim = psnr(ddim_image_tensor.unsqueeze(0),ori_img_tensor.unsqueeze(0))
        psnr_ours = psnr(our_image_tensor.unsqueeze(0),ori_img_tensor.unsqueeze(0))
        ddim_eval_dict['psnr']+=psnr_ddim
        ours_eval_dict['psnr']+=psnr_ours
        # print('PSNR scores:', psnr_ours, psnr_ddim)

        # MSE 
        ddim_mse = np.mean((np.array(ddim_image_tensor) - np.array(ori_img_tensor)) **2 )
        ours_mse = np.mean((np.array(our_image_tensor) - np.array(ori_img_tensor)) **2 )
        ddim_eval_dict['mse']+=ddim_mse
        ours_eval_dict['mse']+=ours_mse

# count_imgs
for key in ddim_eval_dict.keys():
    ddim_eval_dict[key] /= count_imgs
for key in ours_eval_dict.keys():
    ours_eval_dict[key] /= count_imgs

print(ours_eval_dict)
print(ddim_eval_dict)
