### NOTE: to do the inversion for imgs, our methods

### NOTE: inversion paramters:   
# 1. classificaer free grandce of stage1  ### stage2 stage3     GUIDE_1: 1.0 3.0 5.0 7.0 
# 2. Noise level,            
# 3. LR_stage1               
# 4. scale of stage 3   
# 5. NPI vs. NTI 


import os 
import torch
# End2end inversion of ours, Runing for pix2pix-zero datasets. 18 images (cat, dog)

GUIDE_1=1.0        
NOISE_LEVEL_3=100  
LR_1=1e-3                
SCALE_FACTOR_3=0.3 

with open('prompt_all_imgs.txt', 'r') as file:
    for line in file:
        input_img, prompt = line.strip().split(': ')
        result = os.system(f"CUDA_VISIBLE_DEVICES=5  python end2end_inv.py  \
                            --input_image '{input_img}' \
                            --prompt_str '{prompt}' \
                            --results_folder 'output/all_imgs_inversion_in_prompt_file' \
                            --enable_3for2 \
                            --enable_3 \
                            --inner_steps_1 51 \
                            --inner_steps_3 21 \
                            --guidance_1 {GUIDE_1} \
                            --guidance_3 1.0 \
                            --noise_level_3 {NOISE_LEVEL_3} \
                            --lr_1 {LR_1} \
                            --scale_factor_3 {SCALE_FACTOR_3} \
                            --enable_1 \
                            --no_is_NPI \
                            ") 
        if result == 0: 
            print("Command executed successfully.")
        else:
            print("Command failed.")
            torch.cuda.empty_cache()

