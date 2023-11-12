## chmod +x bash.sh 


IMG_FORMAT='jpg'
IMG_FOLDER='images/'
FILE_NAME='buildings'
PROMPT="a boat is docked in the water at sunset"

CUDA_VISIBLE_DEVICES=7  python end2end_inv.py \
            --input_image "${IMG_FOLDER}/${FILE_NAME}.${IMG_FORMAT}" \
            --prompt_str "${PROMPT}" \
            --enable_3for2 \
            --enable_3 \
            --inner_steps_1 51 \
            --inner_steps_3 21 \
            --guidance_1 3.0 \
            # --guidance_3 ${GUIDE_3} \
            --noise_level_3 100 \
            --lr_1 1e-3 \
            --scale_factor_3 0.3 \
            --enable_1 \

# done

