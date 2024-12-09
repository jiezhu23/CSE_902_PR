# Exploring Biometric Recognition Capability with MLLMs

This GitHub Repo is for the final project of CSE 902 Selected Topics in Recognition bu Machine Leanring.

We use [LLaVA 1.5](https://github.com/haotian-liu/LLaVA/) as the MLLM model for the experiment.

## Install

If you are not using Linux, do *NOT* proceed, see instructions for [macOS](https://github.com/haotian-liu/LLaVA/blob/main/docs/macOS.md) and [Windows](https://github.com/haotian-liu/LLaVA/blob/main/docs/Windows.md).

<!-- 1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
``` -->

1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Upgrade to latest code base

```Shell
git pull
pip install -e .

# if you see some import errors when you upgrade,
# please try running the command below (without #)
# pip install flash-attn --no-build-isolation --no-cache-dir
```
## LTCC Dataset

Submitting requests for the LTCC dataset from [link](https://naiq.github.io/LTCC_Perosn_ReID.html) or you can use other person ReID dataset for the experiment.

## How to Run

### Image-Only Setting

run the following script to get the zero-shot performance of LLaVA visual encoder on LTCC:

```
python -m demo.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --device cuda \
    --load-4bit \
    --task image_only \
```

### Image-Text Setting

run the following script to get the performance of Image-Text setting on LTCC:

```
python -m demo.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --device cuda \
    --load-4bit \
    --task image_text \
    --saved-json ./LTCC_Image_Text.json
```

Sicne querying answer for the dataset is time-comsuing, it will update the responses in ```args.save_json``` file. 

For performance evaluation and cases visualzation, you can refer to ```sample_visualize()``` in ```demo.py```

### Funny things 

We also support a command-line conversation with multiple images in one prompt. Also we support multi-round query for different images.

run:
```
python -m demo.py \
    --model-path liuhaotian/llava-v1.5-7b \
    --device cuda \
    --load-4bit \
    --use-conversation
```
You can give the image by inputing prompt like:

```
<img_path>/data/1.png</img_path><img_path>/data/abc.png</img_path>{instruction} CONTINUE YOUR SENTENCES.

# <img_path></img_path> is a special token for image processing
```

## Others

Of courese it supports all the task in original LLaVA GitHub repo (Please refer to [README.MD](https://github.com/haotian-liu/LLaVA/blob/main/README.md)). Feel free to try!
