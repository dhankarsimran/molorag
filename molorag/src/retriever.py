import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

class LogicAwareRetriever:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-3B-Instruct", device="cpu"):
        self.device = device
        # Using 4-bit or 8-bit quantization if possible to save memory, but for now standard loading
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32, 
            device_map="auto" if self.device == "cuda" else None
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def get_semantic_score(self, query_emb, page_emb):
        """Calculate semantic relevance (cosine similarity)."""
        # Ensure they are normalized
        score = np.dot(query_emb, page_emb)
        return float(score)

    def get_logical_score(self, question, page_image):
        """Prompt the VLM to score logical relevance (1-5)."""
        prompt = f"""Evaluate the logical relevance of this document page image to the following question. 
Specifically, determine if this page is logically necessary to answer the question, even if it doesn't contain direct keyword matches.
For example, it might provide context, definitions, or data referenced in other pages.

Question: {question}

Rate the logical relevance on a scale of 1 to 5:
1: Not relevant at all.
2: Slightly relevant (provides minor context).
3: Moderately relevant (provides important background).
4: Highly relevant (essential logical stepping stone).
5: Directly contains the answer or critical evidence.

Output ONLY the integer score (1, 2, 3, 4, or 5)."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": prompt},
                ],
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

        # Inference: Generation of the response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=5)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # Parse the score
        try:
            score = int(output_text.strip()[0])
            # Normalize to 0-1 range (optional, but MoLoRAG uses 1-5 scale)
            # Re-scaling to be comparable to semantic similarity (0-1)
            return (score - 1) / 4.0
        except:
            # Fallback if parsing fails
            return 0.5 

if __name__ == "__main__":
    # Test (requires images from indexing.py or dummy images)
    # This is a placeholder for unit testing
    print("LogicAwareRetriever implemented.")
