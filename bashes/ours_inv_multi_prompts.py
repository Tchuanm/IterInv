### NOTE: to do the inversion for imgs, our methods
# 
# 
import os

# End2end inversion of ours, Runing for Plug-and-Play-datasets datasets. 18 images (cat, dog)

with open('prompt_others.txt', 'r') as file:
    for line in file:
        input_img, prompt = line.strip().split(': ')
        result = os.system(f"CUDA_VISIBLE_DEVICES=4  python end2end_inv.py  \
                           --input_image '{input_img}' --prompt_str '{prompt}' --results_folder 'output/Plug-and-Play-datasets' ") 
        if result == 0: 
            print("Command executed successfully.")
        else:
            print("Command failed.")
        
## generally 20mins for 1 image. 56 / 3 = 19 hours. 


# End2end inversion of ours, Runing for pix2pix-zero datasets. 18 images (cat, dog)

# with open('prompt_pix2pix-zero.txt', 'r') as file:
#     for line in file:
#         input_img, prompt = line.strip().split(': ')
#         result = os.system(f"CUDA_VISIBLE_DEVICES=5  python end2end_inv.py  \
#                            --input_image '{input_img}' --prompt_str '{prompt}' --results_folder 'output/pix2pix-zero' ") 
#         if result == 0: 
#             print("Command executed successfully.")
#         else:
#             print("Command failed.")
        
# # CUDA_VISIBLE_DEVICES=5  python end2end_inv.py   --input_image 'images/Plug-and-Play-datasets/wild-ti2i/data/bear_sketch.jpeg' --prompt_str 'a drawing of a bear head' --results_folder 'output/all_imgs_inversion_in_prompt_file/'  --no_is_NPI
