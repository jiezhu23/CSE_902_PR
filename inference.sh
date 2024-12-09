CUDA_VISIBLE_DEVICES=0 
# python -m llava.serve.cli \

python -m demo.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --device cuda \
    --load-4bit \
    --task image_text \
    --saved-json ./LTCC_Image_Text.json
