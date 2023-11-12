# [IterInv](https://arxiv.org/abs/2310.19540)
The official implement of "IterInv: Iterative Inversion for Pixel-Level T2I Models". 

Accepted paper at [NeurIPS 2023 Workshop on Diffusion Models](https://neurips.cc/virtual/2023/74859).


# [IterInv: Iterative Inversion for Pixel-Level T2I Models](https://arxiv.org/abs/2310.19540) [(eurIPS 2023 Workshop on Diffusion Models)](https://neurips.cc/virtual/2023/74859) 

### [Chuanming Tang](https://scholar.google.com/citations?user=BiRPM9AAAAAJ), [Kai Wang](https://scholar.google.com/citations?user=j14vd0wAAAAJ), [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

## Environment Setting
0. Our code is based on diffusers-0.19.0
1. Download the dataset from [GoogleDrive](https://drive.google.com/drive/folders/1dTWpCPYRJqYaCNy7YkG9c97XTagylWbt?usp=drive_link). 
2. create environment.
```
conda create --name floyd --file environment.yml
conda activate floyd
```

3. If you want to get a prompt of our own images, you can use BLIP.ipynb to get the text prompt.





## Reconstruction Image based on IterInv. 
1. Reconstruct a iamge  based on IterInv. 
```
CUDA_VISIBLE_DEVICES=7 python end2end_inv.py \
                            --input_image 'images/pix2pix-zero/cat/cat_7.png'  \
                            --results_folder  'output/all_imgs_inversion_in_prompt_file'  \
                            --prompt_str 'a cat' 
                            --enable_1 \
                            --enable_3for2 \
                            --enable_3 \
# or
bash bashes/bash_inv_1img.sh                 
```

2. Reconstruct multiple images based on IterInv. 
```
python bashes/ours_inv_multi_prompt.py
```

3. Reconstruct based on DDIM Inversion. 
Choose stage 1/2/3 to groups what you want.

```
python bashes/ddim_inv_multi_prompts.py 
# including ddim_stage23_inv.py ddim_failure_stage_3.py  end2end_ddim_inv.py to chooose.
```

4. Reconstruct based on  SDXL.
```
python SDXL.py
# or in SDXL.ipynb to run it one-by-one step for better development. 
```



## Editing images
1. editing with IterInv + [DiffEdit](https://huggingface.co/docs/diffusers/api/pipelines/diffedit).

```
CUDA_VISIBLE_DEVICES=7 python end2end_diffedit.py \ 
                    --enable_1  --enable_3for2 --enable_3     \
                    --inpaint_strength 0.4 

```

2. We also realise the experiments of combined with p2p and pix2pix_zero, but the editing results is not satisfied. 
See end2end_p2p_edit.py and end2end_zero_edit.py. 




# Quantitative comparison of image inversion results. 
```
python evaluation_scores.py  
# or single-step debug in evaluation_scores.ipynb
```
Change the folder to find what you want to evaluate. 


## Acknowledgement
Thanks for the [diffusers](https://huggingface.co/docs/diffusers/index) and [DeepFloyd-IF](https://github.com/deep-floyd/IF),  which helps us to quickly implement our ideas. \
Note: this is our draft code release for paper. The cleaned version will be released later.


## Citation

If our work is useful for your research, you can consider citing:


```
@article{tang2023iterinv,
  title={IterInv: Iterative Inversion for Pixel-Level T2I Models},
  author={Tang, Chuanming and Wang, Kai and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2310.19540},
  year={2023}
}
```