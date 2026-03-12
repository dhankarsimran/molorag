import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import re

# --- Configuration ---
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
ADAPTER_PATH = "/Users/niteeshkumar/Documents/molorag/molorag_plus_v2/outputs/final_adapter"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class MoLoRAGPlusV2Retriever:
    def __init__(self):
        print(f"Loading MoLoRAG+ v2 (Base: {BASE_MODEL_ID} + Adapter: {ADAPTER_PATH})...")
        
        # Load Base
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE
        )
        
        # Load Adapter if exists
        if os.path.exists(ADAPTER_PATH):
            print("Integrating Fine-Tuned LoRA Adapter...")
            self.model = PeftModel.from_pretrained(self.model, ADAPTER_PATH)
        else:
            print("Warning: Adapter path not found. Running with base model.")
            
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    def get_logical_score(self, question, page_image):
        prompt = f"""# QUERY #
{question}
# INSTRUCTION #
Output ONLY the relevance score (1-5)."""

        messages = [{"role": "user", "content": [{"type": "image", "image": page_image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=5)
            out = self.processor.batch_decode(ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        score_match = re.search(r'[1-5]', out)
        score = int(score_match.group(0)) if score_match else 3
        return (score - 1) / 4.0

# Simple Demo
if __name__ == "__main__":
    from PIL import Image
    # Just a placeholder test
    retriever = MoLoRAGPlusV2Retriever()
    print("Retriever Ready for MoLoRAG+ v2 Traversal.")
