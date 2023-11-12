
# CUDA_VISIBLE_DEVICES=7  python end2end_edit.py  \
#                            --input_image 'images/gnochi_mirror.jpeg' --prompt_str 'a cat sitting on a wooden counter looking at itself in the mirror' --results_folder 'output/all_imgs_inversion_in_prompt_file' \
#                            --edit_prompt_str 'a dog sitting on a wooden counter looking at itself in the mirror'   \
#                            --guidance_1 3.0  \
#                            --no_is_NPI  

import argparse

import os 
food_names = [
    "Burger",
    "Sushi",
    "ice cream",
    "steak",
    "pasta",
    "salad",
    "burrito",
    "fried chicken",
    "sandwich",
    "cakes",]

animal_names = [
    "a lion",
    "an elephant",
    "a tiger",
    "a zebra",
    "a panda",
    "a dolphin",
    "a cheetah",
    "a penguin",
    "a koala",
    "a chimpanzee",
    "a bear",
    "a fox",
    "an wolf",
    "a bird",
    "a cat",
    "a dog"
]
inpaint_strength_list = [0.8, 0.6, 0.4, 0.3]
CFG1_list = [3.0, 5.0, 7.5, 6.0]

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_img', type=str, default='images/pix2pix-zero/dog/dog_7.png')
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--edit_prompt', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--mask_encode_strength', type=float, default=0.1)
    parser.add_argument('--mask_thresholding_ratio', type=float, default=1)
    parser.add_argument('--is_food', action='store_true')
    parser.add_argument('--no_is_food', dest='is_food', action='store_false')
    parser.set_defaults(is_food=False)

    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = arguments()
    input_img = args.input_img
    prompt =  args.prompt
    gpu = args.gpu
    if args.is_food:
        edit_prompts_list = food_names
    elif args.edit_prompt is not None:
        edit_prompts_list =  ['tank', 'a porsche sports car', 'a truck', 'a bus']
    else:
        edit_prompts_list = animal_names

    for edit_prompt in edit_prompts_list:
        for inpaint_strength in inpaint_strength_list:
            for CFG1 in CFG1_list:
                result = os.system(f"CUDA_VISIBLE_DEVICES={gpu}  python end2end_diffedit.py \
                                --input_image '{input_img}' \
                                --enable_1 \
                                --enable_3for2   \
                                --inpaint_strength  {inpaint_strength} \
                                --prompt_str '{prompt}' \
                                    --edit_prompt_str '{edit_prompt}'  \
                                --guidance_1_edit {CFG1} \
                                    --mask_encode_strength  {args.mask_encode_strength}  \
                                    --mask_thresholding_ratio   {args.mask_thresholding_ratio} \
                                    ") 
                # result = os.system(f"CUDA_VISIBLE_DEVICES=6  python end2end_diffedit.py \
                #                     --input_image '{input_img}' \
                #                    --enable_3for2 --enable_3 \
                #                    --inpaint_strength {inpaint_strength}\
                #                     --prompt_str '{prompt}' \
                #                     --edit_prompt_str '{edit_prompt}'  \
                #                     --guidance_1_edit 3.0 \
                #                     --inpaint_strength  0.6 \
                #                         ") 
                if result == 0: 
                    print("Command executed successfully.")
                else:
                    print("Command failed.")



""" 
CUDA_VISIBLE_DEVICES=6  python end2end_diffedit.py \
                            --input_image 'images/cat_hq.jpg' \
                           --enable_1 --no_enable_3for2 \
                            --prompt_str 'a cat' \
                            --edit_prompt_str  'a dog'  \
                           --guidance_1_edit 3.0 \
                            --inpaint_strength  0.4   --num_inference_steps_1 100

CUDA_VISIBLE_DEVICES=6  python end2end_diffedit.py \
                            --input_image 'images/cat_hq.jpg' \
                           --enable_3for2 --enable_3  --no_enable_1\
                            --prompt_str 'a cat' \
                            --edit_prompt_str  'a dog'  \
                           --guidance_1_edit  3.0 \
                            --inpaint_strength  0.4 \
                            --num_inference_steps_1 100

python bashes/edit_multi_prompts.py --gpu 5  --input_img 'images/Plug-and-Play-datasets/imagenetr-ti2i/panda/cartoon_30.jpg'    --prompt 'a panda' 
python bashes/edit_multi_prompts.py --gpu 6  --input_img 'images/Plug-and-Play-datasets/imagenetr-ti2i/panda/sketch_36.jpg'    --prompt 'a panda' 
python bashes/edit_multi_prompts.py --gpu 3  --input_img 'images/Plug-and-Play-datasets/imagenetr-ti2i/panda/sculpture_29.jpg'    --prompt 'a panda' 



"""
