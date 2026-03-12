"""
MoLoRAG Baseline: Main QA Script
--------------------------------
Main entry point for evaluating various Visual Language Models (VLMs) 
on document-based question answering tasks.
"""

import os
import json
import argparse
import torch 
import time 
import re
from PIL import Image
from tqdm import tqdm
from utils import convert_page_snapshot_to_image, concat_images

# Disable image pixel limit for large document pages
Image.MAX_IMAGE_PIXELS = None

def load_vlm_model(model_name, device):
    """Dynamically loads the requested VLM model."""
    if "QwenVL" in model_name:
        from VLMModels.Qwen_VL import init_model, get_response_concat
    elif "DeepSeek-VL" in model_name:
        from VLMModels.DeepSeek_VL import init_model, get_response_concat
    elif "LLaVA-Next" in model_name:
        from VLMModels.LLaVA_Next import init_model, get_response_concat
    elif model_name == "LLaMA-VL-11B":
        from VLMModels.LLaMA_VL import init_model, get_response_concat
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")
    
    model = init_model(model_name, device)
    return model, get_response_concat
    

def main_lvlm_QA(args):
    """Main execution loop for VLM-based QA."""
    st_time = time.time()
    
    # Setup output path
    document_folder = f"./dataset/{args.dataset}"
    img_folder = f"./tmp/tmp_imgs/{args.dataset}"
    result_folder = f"./results/{args.dataset}/{args.model_name}"
    os.makedirs(result_folder, exist_ok=True)

    retrieve_suffix = "Direct" if args.retriever == "None" else f"{args.retriever}_top{args.topk}"
    output_path = f"{result_folder}/{retrieve_suffix}.json"

    # Load baseline samples
    if os.path.exists(output_path):
        print(f"[*] Resuming from {output_path}...")
        samples = json.load(open(output_path, "r"))
    else:
        input_path = f"./dataset/samples_{args.dataset}.json"
        if args.retriever != "None":
            retrieved_path = f"./dataset/retrieved/samples_{args.dataset}_{args.retriever}.json"
            if os.path.exists(retrieved_path):
                input_path = retrieved_path
                print(f"[*] Loading retrieved pages from {args.retriever}...")
        
        print(f"[*] Loading samples from {input_path}...")
        samples = json.load(open(input_path, "r"))

    # Initialize model
    model, get_response_concat_fn = load_vlm_model(args.model_name, args.device)

    # Process samples
    for sample in tqdm(samples, desc="Processing Samples"):
        if args.response_key in sample and sample[args.response_key] != "None":
            continue
        
        # Convert document pages to images
        input_image_list = convert_page_snapshot_to_image(
            doc_path=f"{document_folder}/{sample['doc_id']}", 
            save_path=img_folder, 
            resolution=args.resolution, 
            max_pages=args.max_pages
        )
        
        # Handle page ranking if available
        if "pages_ranking" in sample:
            ranked_pages = eval(sample["pages_ranking"])[:args.topk]
            input_image_list = [input_image_list[page-1] for page in ranked_pages]   

        # Handle image concatenation for specific VLMs
        if args.concat_num > 0:
            name_suffix = "concat" if args.retriever == "None" else f"{args.retriever}_top{args.topk}-concat"
            input_image_list = concat_images(image_list=input_image_list, concat_num=args.concat_num, name_suffix=name_suffix) 
        
        try:
            query_prompt = f"Based on the document, please answer the question: {sample['question']}"
            response = get_response_concat_fn(model, query_prompt, input_image_list, max_new_tokens=args.max_tokens, temperature=args.temperature)
        except Exception as e:
            print(f"[ERROR] VLM prediction failure: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
            response = "None"

        sample[args.response_key] = response 
     
        # Periodically save results
        with open(output_path, 'w') as file:
            json.dump(samples, file, indent=4, sort_keys=True)
        
    print(f"\n[DONE] Dataset: {args.dataset} | Model: {args.model_name} | Retriever: {args.retriever}")
    print(f"Total time: {(time.time() - st_time)/60:.2f} Mins\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MMLong", choices=["MMLong", "LongDocURL", "PaperTab", "FetaTab"])
    parser.add_argument("--model_name", type=str, default="QwenVL-3B", choices=["QwenVL-3B", "QwenVL-7B", "DeepSeek-VL-tiny", "DeepSeek-VL-small", "LLaVA-Next-7B", "LLaVA-Next-8B", "LLaMA-VL-11B"])
    parser.add_argument("--max_pages", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concat_num", type=int, default=0)
    parser.add_argument("--retriever", type=str, default="None", choices=["None", "base", "beamsearch", "beamsearch_LoRA"])
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--response_key", type=str, default="raw_response")
    args = parser.parse_args()
    
    # Device configuration
    if "," in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.replace("cuda:", "")
        args.device = "auto"
    else:
        args.device = torch.device(args.device) if torch.cuda.is_available() else "cpu"
    
    args.max_pages = 1000 if args.retriever != "None" else args.max_pages
    
    main_lvlm_QA(args)
