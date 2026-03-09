import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image

class MoLoRAGGenerator:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct", device="cpu"):
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def generate_answer(self, question, retrieved_images):
        """Synthesize the final answer using the top-K images."""
        
        # Prepare content list with images and text
        content = []
        for img in retrieved_images:
            content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": f"Answer the following question based on the provided document pages.\n\nQuestion: {question}"})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Generate answer
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            answer = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
        return answer

if __name__ == "__main__":
    print("MoLoRAGGenerator implemented.")
