import random
import os
import argparse
import torch
from llava.eval.run_llava_two_images import eval_model_conversation, eval_model
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
from llava.model.builder import load_pretrained_model
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import seed_everything

from PIL import Image
from tqdm import tqdm
import json

class ImageDataset(Dataset):
        def __init__(self, image_list, image_processor, model_config):
            self.image_list = image_list
            self.image_processor = image_processor
            self.model_config = model_config

        def __len__(self):
            return len(self.image_list)

        def __getitem__(self, idx):
            img_path = self.image_list[idx]
            label = int(os.path.basename(img_path).split('_')[0])
            img = Image.open(img_path).convert("RGB")
            img_tensor = process_images([img], self.image_processor, self.model_config)[0]
            return img_tensor, torch.tensor([label])


def performace_evaluate_image_only(query_features, query_labels, gallery_features, gallery_labels):
    # calculate similarity between query and gallery features using cosine similarity
    query_features = query_features / query_features.norm(dim=1, keepdim=True)
    gallery_features = gallery_features / gallery_features.norm(dim=1, keepdim=True)
    
    similarity_matrix = torch.mm(query_features, gallery_features.t())
    
    # CMC and Mean Average Precision (mAP)
    num_queries = query_features.size(0)
    CMC = torch.zeros(len(gallery_labels))
    average_precisions = []
    
    for i in range(num_queries):
        relevant = (gallery_labels == query_labels[i]).float()
        sorted_indices = similarity_matrix[i].argsort(descending=True)
        sorted_relevant = relevant[sorted_indices]
        # CMC
        rows_good = torch.where(sorted_relevant == 1)[0]
        CMC[rows_good[0]:] += 1.0
        # mAP
        cumulative_relevant = sorted_relevant.cumsum(0)
        precision_at_k = cumulative_relevant / (torch.arange(1, len(sorted_relevant) + 1).float())
        average_precision = (precision_at_k * sorted_relevant).sum() / relevant.sum()
        average_precisions.append(average_precision.item())
    
    CMC = CMC / num_queries
    mAP = sum(average_precisions) / num_queries
    
    return CMC, mAP


def performance_evaluate_image_text(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    FP, FN, TP, TN = 0, 0, 0, 0
    tp_keys, tn_keys, fp_keys, fn_keys = [], [], [], []
    for img_pair, result in data.items():
        if img_pair == 'error_logs':
            continue
        img_path1, img_path2 = img_pair.split(', ')
        img_label1, img_label2 = img_path1.split('_')[0], img_path2.split('_')[0]
        y = 1 if img_label1 == img_label2 else 0
        y_pred = 1 if result['Decision'].lower() == 'yes' else 0
        if y == 1 and y_pred == 1:
            TP += 1
            tp_keys.append(img_pair)
        elif y == 1 and y_pred == 0:
            FN += 1
            fn_keys.append(img_pair)
        elif y == 0 and y_pred == 1:
            FP += 1
            fp_keys.append(img_pair)
        else:
            TN += 1
            tn_keys.append(img_pair)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print(f"Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.1%}, Accuracy: {accuracy:.1%}")
    return tp_keys, tn_keys, fp_keys, fn_keys


def sample_query_gallery(dataset_root, query_img_size, gallery_img_size, dataset='LTCC'):
    # Get all image files in the folder
    if dataset == 'LTCC':
        query_folder = os.path.join(dataset_root, 'query')
        gallery_folder = os.path.join(dataset_root, 'test')
    
    query_all_imgs = [f for f in os.listdir(query_folder) if f.endswith(('.png'))]
    gallery_all_imgs = [f for f in os.listdir(gallery_folder) if f.endswith(('.png'))]
    
    # Create a dictionary based on the id
    query_img_dict = defaultdict(list)
    gallery_img_dict = defaultdict(list)
    for img in query_all_imgs:
        img_id = img.split('_')[0]
        query_img_dict[img_id].append(os.path.join(query_folder, img))
    for img in gallery_all_imgs:
        img_id = img.split('_')[0]
        gallery_img_dict[img_id].append(os.path.join(gallery_folder, img))
    
    # Calculate number of ids
    M_query = len(query_img_dict)
    M_gallery = len(gallery_img_dict)
    
    # Sample N//M images for each id
    sampled_query, sampled_galleries = [], []
    for img_list in query_img_dict.values():
        sampled_query.extend(random.sample(img_list, min(len(img_list), query_img_size  // M_query)))
    for img_list in gallery_img_dict.values():
        sampled_galleries.extend(random.sample(img_list, min(len(img_list), gallery_img_size  // M_gallery)))
    
    print(f"Sampled {len(sampled_query)} images from query, {len(sampled_galleries)} images from gallery for dataset {dataset}")
    
    return sampled_query, sampled_galleries

@torch.no_grad()
def get_image_features(dataloader, model, device):
    features, labels = torch.tensor([]), torch.tensor([])
    for x, x_label in tqdm(dataloader):
        # we use the CLS token from last layer of visual encoder as the image feature for similarity comparison
        # NOTE: LLAVA use the image token from the last 2 layers of visual encoder as the image feature input to LLM
        images_tensor = model.get_vision_tower().vision_tower(x.to(device)).last_hidden_state.cpu() # (B, patch_size=577, 1024)
        images_cls_token = images_tensor[:, 0, :]
        features = torch.cat([features, images_cls_token])
        labels = torch.cat([labels, x_label])
        torch.cuda.empty_cache()
    return features, labels


def get_text_response(img_paths, instruction):
    if type(img_paths[0]) is tuple:
        # multiple images for each instruction
        inps = []
        for img_pairs in img_paths:
            assert len(img_pairs) == 2, "Each instruction should only have 2 image"
            img_inps = ' '.join([f'<img_path>{img_path}</img_path>' for img_path in img_pairs])
            inps.append(f"{img_inps} {instruction}")
    else:
        inps = [f"<img_path>{img_path}</img_path> {instruction}"for img_path in img_paths]
    responses = eval_model(args, inps) # it will save the responses to a json file
    return responses


def sample_visualize(json_file, selected_idx=0, selected_key=None, dataset_root='/data/LTCC_ReID'):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    if selected_key is not None:
        img_pair = selected_key
        result = data[selected_key]
    else:
        img_pair = list(data.keys())[selected_idx]
        result = data[img_pair]
    img_path1, img_path2 = img_pair.split(', ')
    img_label1, img_label2 = img_path1.split('_')[0], img_path2.split('_')[0]
    img_path1 = os.path.join(dataset_root, 'query', img_path1)
    img_path2 = os.path.join(dataset_root, 'test', img_path2)
    
    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")
        
    # Resize images to the same height
    height = min(img1.height, img2.height)
    img1 = img1.resize((int(img1.width * height / img1.height), height))
    img2 = img2.resize((int(img2.width * height / img2.height), height))
    
    # Add padding between images
    padding = 10
    width = img1.width + img2.width + padding
    
    # Concatenate images side by side with padding
    integrated_img = Image.new('RGB', (width, height), (255, 255, 255))  # White background
    integrated_img.paste(img1, (0, 0))
    integrated_img.paste(img2, (img1.width + padding, 0))
    
    # Save the integrated image
    save_path = './tmp.png'
    integrated_img.save(save_path)
    print(f"Image pair: {img_path1}, {img_path2}, Decision: {result['Decision']}, Reason: {result['Reason']}")
        

def main(args):
    query_list, gallery_list = sample_query_gallery('/data/LTCC_ReID', 75*2, 75*2)
    if args.task == 'image_only':
        batch_size = 32
        model_name = get_model_name_from_path(args.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, device=args.device)
        query_dataset = ImageDataset(query_list, image_processor, model.config)
        gallery_dataset = ImageDataset(gallery_list, image_processor, model.config)
        query_dataloader = DataLoader(query_dataset, batch_size=batch_size, shuffle=False)
        gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False)
        
        query_features, query_labels = get_image_features(query_dataloader, model, args.device)
        gallery_features, gallery_labels = get_image_features(gallery_dataloader, model, args.device)
        # performance evaluate
        top1, mAP = performace_evaluate_image_only(query_features, query_labels, gallery_features, gallery_labels)
        print(f"Rank-1 accuracy: {top1:.1%}, mAP: {mAP}")
        
    elif args.task == 'image_text':
        # Create 1:1 image pairs
        img_pairs = [(query, gallery) for query in query_list for gallery in gallery_list]
        img_pairs_label = [1 if os.path.basename(query).split('_')[0] == os.path.basename(gallery).split('_')[0] else 0 for query, gallery in img_pairs]
        instruction = """Focus on the main person in the image. Describe the difference between these two images and tell me if these two persons are the same. Reply strictly in a json format. First key "Decision" is the decision (yes or no only), second key "Reason" is the reason of the decision (detailed as possible)."""
        responses = get_text_response(img_pairs, instruction)
        
        # Evaluate the performance
        performance_evaluate_image_text(args.saved_json)
    else:
        raise ValueError(f"Invalid task {args.task}, please choose from ['image_only', 'image_text']")


def get_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=True)
    parser.add_argument("--use-conversation", action="store_true", default=False)
    parser.add_argument("--task", type=str, default='image_text', choices=['image_only', 'image_text'])
    parser.add_argument("--saved-json", type=str, default='./LTCC_Image_Text.json')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    seed_everything(3)

    if args.use_conversation:
        eval_model_conversation(args)
    else:
        main(args)
        # instruction = "describe the image"
        # imgs_path_list = ["/home/jiezhu/Desktop/code/LLaVA/images/llava_logo.png", "/data/LTCC_ReID/test/001_1_c4_015855.png"]*10
        # inps = [f"<img_path>{img}</img_path> {instruction}"for img in imgs_path_list]
        # eval_model(args, inps)
        
        
    # evaluate the performance of image-text task    
    # # tp_keys, tn_keys, fp_keys, fn_keys = performance_evaluate_image_text('./LTCC_Image_Text.json')
    # sample_visualize('./LTCC_Image_Text.json', selected_idx=0)