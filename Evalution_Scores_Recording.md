
# Hyperparameter ablation study of  quantitative comparison. 





## 4. 3.0 NPI vs. stage123 
{'lpips': tensor(0.0356), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3158, grad_fn=<DivBackward0>), 'psnr': tensor(40.5955), 'mse': 0.00013039794848209831}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366), 'mse': 0.07992389089549365}
## 5.0 NPI vs. 123
{'lpips': tensor(0.0355), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3157, grad_fn=<DivBackward0>), 'psnr': tensor(40.6037)}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366)}

## 7.0 NPI vs. 123
{'lpips': tensor(0.0356), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3158, grad_fn=<DivBackward0>), 'psnr': tensor(40.5959)}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366)}

## 1.0 NTI vs. 123
{'lpips': tensor(0.0356), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3155, grad_fn=<DivBackward0>), 'psnr': tensor(40.5930), 'mse': 0.0001304753707607734}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366), 'mse': 0.07992389089549365}


## 3.0 NTI vs. 123
{'lpips': tensor(0.0353), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3161, grad_fn=<DivBackward0>), 'psnr': tensor(40.6459), 'mse': 0.0001290986220355726}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366), 'mse': 0.07992389089549365}

## 5.0 NTI vs. 123
{'lpips': tensor(0.0353), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3161, grad_fn=<DivBackward0>), 'psnr': tensor(40.6484), 'mse': 0.00012907061201089613}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366), 'mse': 0.07992389089549365}

## 7.0 NTI vs. 123
{'lpips': tensor(0.0353), 'ssim': tensor(0.9806), 'clip_score': tensor(21.3161, grad_fn=<DivBackward0>), 'psnr': tensor(40.6427), 'mse': 0.00012912355525285406}
{'lpips': tensor(0.6393), 'ssim': tensor(0.5821), 'clip_score': tensor(21.2223, grad_fn=<DivBackward0>), 'psnr': tensor(11.8366), 'mse': 0.07992389089549365}

# ours123_new vs. SDXL
SDXL： {'lpips': tensor(0.1269), 'ssim': tensor(0.9048), 'clip_score': tensor(21.3172, grad_fn=<DivBackward0>), 'psnr': tensor(33.4993), 'mse': 0.009015853739043267}



CUDA_VISIBLE_DEVICES=7  python end2end_inv.py \
            --input_image "images/gnochi_mirror.jpeg" \
            --prompt_str "a cat sitting on a wooden counter looking at itself in the mirror" \
            --results_folder "output/all_imgs_inversion_in_prompt_file"  \
            --guidance_1 5.0 \
            --is_NPI

