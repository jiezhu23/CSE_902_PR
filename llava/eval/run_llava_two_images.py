"""
 Reference from: https://github.com/MaxFBurg/LLaVA/blob/main/llava/eval/run_llava_two_images.py

"""

import argparse
import torch
import os
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import json


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def parse_img_path(inp):
    img_paths = []
    text_parts = []
    while True:
        start_idx = inp.find('<img_path>')
        end_idx = inp.find('</img_path>')
        if start_idx != -1 and end_idx != -1:
            text_parts.append(inp[:start_idx].strip())
            img_path = inp[start_idx + len('<img_path>'):end_idx].strip()
            img_paths.append(img_path)
            inp = inp[end_idx + len('</img_path>'):]
        else:
            text_parts.append(inp.strip())
            break
    text = ' '.join(filter(None, text_parts))
    return img_paths, text

def eval_model_conversation(args):
     # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
    images = [load_image(img) for img in args.image_file.split(",")]
    image_size = images[0].size
    # Similar operation in model_worker.py
    image_tensor = process_images([img for img in images], image_processor, model.config) # (N, 3, 336, 336)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    while True:
        try:
            inp = input(f"{roles[0]}: ")
            # define special tokens for input images path
            inp_images, inp = parse_img_path(inp)
            if len(inp_images) > 0:
                images = [load_image(img) for img in inp_images]
                image_size = images[0].size
                # Similar operation in model_worker.py
                image_tensor = process_images([img for img in images], image_processor, model.config) # (N, 3, 336, 336)
                if type(image_tensor) is list:
                    image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
                else:
                    image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if len(images)!=0:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n') * len(images) +  inp
            else:
                inp = (DEFAULT_IMAGE_TOKEN + '\n') * len(images)+ inp
            images = []
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True)

        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


def eval_model(args, inps):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # assert inps is List type
    assert type(inps) == list, "Input should be a list"
    responses = []
    # Load existing data if the file exists
    if args.saved_json is not None and os.path.exists(args.saved_json):
        with open(args.saved_json, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    for idx, inp in enumerate(inps):
        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        # define special tokens for input images path
        inp_images, inp = parse_img_path(inp)
        if f'{os.path.basename(inp_images[0])}, {os.path.basename(inp_images[1])}' in existing_data:
            continue
        if len(inp_images) > 0:
            images = [load_image(img) for img in inp_images]
            image_size = images[0].size
            # Similar operation in model_worker.py
            image_tensor = process_images([img for img in images], image_processor, model.config) # (N, 3, 336, 336)
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
                
        print(f"{roles[1]}: ", end="")

        if len(images)!=0:
            # first message
            if model.config.mm_use_im_start_end:
                inp = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n') * len(images) +  inp
            else:
                inp = (DEFAULT_IMAGE_TOKEN + '\n') * len(images) + inp
            images = []
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=None,
                use_cache=False)
        outputs = tokenizer.decode(output_ids[0]).strip()
        conv.messages[-1][-1] = outputs
        torch.cuda.empty_cache()
        # Transform responses into JSON format and save to a file
        try:
            response_json = {}
            output_json = json.loads(outputs.replace('<s>', '').replace('</s>', '').strip())
            response_json[f'{os.path.basename(inp_images[0])}, {os.path.basename(inp_images[1])}'] = output_json
        except json.JSONDecodeError:
            print(f"Error decoding JSON response {outputs}")
            if existing_data.get('error_logs') is None:
                existing_data['error_logs'] = {}
            existing_data['error_logs'][f'{os.path.basename(inp_images[0])}, {os.path.basename(inp_images[1])}'] = outputs
            continue
        
        # Update existing data with new response
        existing_data.update(response_json)

        # Write the updated data back to the file after each response
        with open(args.saved_json, 'w') as f:
            json.dump(existing_data, f, indent=4)

        if args.debug:
            print("\n", f"[{idx}/{len(inps)}]:", {"prompt": prompt, "outputs": outputs}, "\n")
        responses.append(outputs)
    return responses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    eval_model(args)
