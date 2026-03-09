from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


import os

def format_image_path(raw_path):
    # Adjust to handle both absolute and relative paths from the repo root
    if raw_path.startswith("/"):
        # If it was intended to be relative to some root, we adjust it
        # The repo seems to use paths like /tmp/tmp_imgs/...
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return f"file://{project_root}{raw_path}"
    return f"file://{os.path.abspath(raw_path)}"


def init_model(model_name, device="cpu"):
    if "lora" in model_name: 
        model_path = "xxwu/MoLoRAG-QwenVL-3B"
        print(f"Loading LoRA model from {model_path}")
    elif "3B" in model_name:
        model_path =  "Qwen/Qwen2.5-VL-3B-Instruct"
    elif "7B" in model_name:
        model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Map device string to torch device
    if isinstance(device, str):
        device_type = device.split(":")[0]
    else:
        device_type = device.type

    # Flash Attention is not supported on Mac/CPU
    attn_impl = "sdpa" if device_type in ["mps", "cpu"] else "flash_attention_2"
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device_type != "cpu" else torch.float32,
        attn_implementation=attn_impl,
        device_map=device).eval()
    
    min_pixels = 256 * 28 * 28 
    max_pixels = 512 * 28 * 28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    model.processor = processor
    
    return model 


def get_response_concat(model, question, image_path_list, max_new_tokens=1024, temperature=1.0):
    msgs = []

    if isinstance(image_path_list, list):
        msgs.extend([dict(type='image', image=format_image_path(p)) for p in image_path_list])
    else:
        msgs = [dict(type='image', image=format_image_path(image_path_list))]
    msgs.append(dict(type='text', text=question))
    messages = [{
        "role": "user",
        "content": msgs
    }]

    text = model.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = model.processor(
        text=[text],
        images=image_inputs,
        video_inputs=video_inputs,
        padding=True, 
        return_tensors='pt'
    )
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = model.processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]
