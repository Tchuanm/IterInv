### NOTE: to do the inversion for imgs, our methods
# python bashes/ddim_inv_multi_prompts.py
# 


"""==========================Only stage 3 runing DDIM failure cases=========================="""
# import os
# # CUDA_VISIBLE_DEVICES=7 python ddim_failure_stage_3.py   --input_image 'images/Plug-and-Play-datasets/imagenetr-ti2i/cat/sketch_20.jpg' --prompt_str 'a pencil drawing of cat face' --output_fold 'Plug-and-Play-datasets'

# # 打开文本文件以读取模式
## with open('prompt_Plug-and-Play-datasets.txt', 'r') as file:
## with open('prompt_pix2pix-zero.txt', 'r') as file:

# with open('prompt_all_imgs.txt', 'r') as file:
#     # 逐行读取文件内容
#     for line in file:
#         # 使用分号分隔每一行
#         input_img, prompt = line.strip().split(': ')
#         result = os.system(f"CUDA_VISIBLE_DEVICES=7  python ddim_failure_stage_3.py  \
#                            --input_image '{input_img}' --prompt_str '{prompt}' --output_fold 'only_stage3'   ") 
#         if result == 0:
#             print("Command executed successfully.")
#         else:
#             print("Command failed.")

# CUDA_VISIBLE_DEVICES=7 python ddim_failure_stage_3.py   --input_image 'images/gnochi_mirror.jpeg' --prompt_str 'a cat sitting on a wooden counter looking at itself in the mirror'  --output_fold ''



"""==========================stage 2 and 3 only  runing DDIM failure cases=========================="""
import os

# 打开文本文件以读取模式
with open('prompt_all_imgs.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file: 
        # 使用分号分隔每一行
        input_img, prompt = line.strip().split(': ')
        result = os.system(f"CUDA_VISIBLE_DEVICES=6  python ddim_stage23_inv.py  \
                           --input_image '{input_img}' --prompt_str '{prompt}' --output_fold 'stages23' ") 
        if result == 0: 
            print("Command executed successfully.")
        else:
            print("Command failed.")


# """==========================end2end stage 1-2-3 runing DDIM failure cases=========================="""
# import os

# # 打开文本文件以读取模式
# with open('prompt_all_imgs.txt', 'r') as file:
#     # 逐行读取文件内容
#     for line in file:
#         # 使用分号分隔每一行
#         input_img, prompt = line.strip().split(': ')
#         result = os.system(f"CUDA_VISIBLE_DEVICES=7  python end2end_ddim_inv.py  \
#                            --input_image '{input_img}' --prompt_str '{prompt}' --output_fold 'stages123' ") 
#         if result == 0: 
#             print("Command executed successfully.")
#         else:
#             print("Command failed.")

