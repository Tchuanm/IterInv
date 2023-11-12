import shutil
import os
idx = 0
scale2=0.3
with open('prompt_all_imgs.txt', 'r') as file:
    # 逐行读取文件内容
    for line in file:
        # 使用分号分隔每一行
        idx += 1 
        input_img, prompt = line.strip().split(': ')
        bname = os.path.basename(input_img).split(".")[0]
        if idx < 50:
            prompt = input_img.split('/')[-2]
        else:
            prompt = input_img.split('/')[-1].split('.')[0].split('_')[0]
        prompt = 'a_' + prompt

        old_path = f'/disk1/users/ctang/deepfloyd/output/all_imgs_inversion_in_prompt_file/{bname}/CFG1_3.0_CFG3_1.0_noise3_100_lr1_0.001_scale3_0.3_NPI_False/{prompt}/'
        # source_path = old_path + f'if_stage_II_rec_scale2_{scale2}.png'

        
        source_path = old_path + f'if_stage_III_rec.png'
        if not os.path.exists(source_path):
            continue
        new_path = f'/disk1/users/ctang/deepfloyd/output/stage2_inv/stage123_a_object_scale_{scale2}/'
        if not os.path.exists(new_path):
            os.mkdir(new_path)
        destination_path = new_path + f'{bname}.png'

        shutil.copy(source_path, destination_path)



# object scale_0.3 = 27 / 65
# object scale_0.4 = 18 / 65
# object scale_0.5 = 18 / 65
# a object scale_0.3 = 27 / 65
# a object scale_0.4 = 18 
# a object scale_0.5 = 18 
