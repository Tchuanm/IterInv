
import os 
import torch
# End2end inversion of ours, Runing for pix2pix-zero datasets. 18 images (cat, dog)

SCALE_FACTOR_2=0.5
idx = 0
# 打开文本文件以读取模式
with open('prompt_all_imgs.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        idx += 1
        # 使用分号分隔每一行 
        input_img, longprompt = line.strip().split(': ')
        if idx < 50:
            prompt = input_img.split('/')[-2]
        else:
            prompt = input_img.split('/')[-1].split('.')[0].split('_')[0]
        prompt = 'a ' + prompt

        result = os.system(f"CUDA_VISIBLE_DEVICES=7  python end2end_inv_stage2.py  \
                            --input_image '{input_img}' \
                            --prompt_str '{prompt}' \
                            --long_prompt_str '{longprompt}'    \
                            --results_folder 'output/all_imgs_inversion_in_prompt_file' \
                            --scale_factor_2 {SCALE_FACTOR_2}   \
                            ") 
        if result == 0: 
            print("Command executed successfully.")
        else:
            print("Command failed.")
            torch.cuda.empty_cache()

 