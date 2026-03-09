import torch
import numpy as np
from indexing import DocumentGraphIndex
from retriever import LogicAwareRetriever
from transformers import AutoProcessor, CLIPModel

class MoLoRAGTraversal:
    def __init__(self, index: DocumentGraphIndex, retriever: LogicAwareRetriever, clip_model_name="openai/clip-vit-large-patch14"):
        self.index = index
        self.retriever = retriever
        self.device = index.device
        # Use CLIPModel for aligned vision and text features
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = AutoProcessor.from_pretrained(clip_model_name)

    def embed_query(self, query):
        inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        outputs = self.clip_model.get_text_features(**inputs)
        # Ensure we have the tensor
        query_features = outputs if torch.is_tensor(outputs) else outputs.pooler_output
        query_features /= query_features.norm(dim=-1, keepdim=True)
        return query_features.cpu().detach().numpy().flatten()

    def run_traversal(self, query, w=3, n_hops=4):
        query_emb = self.embed_query(query)
        
        # 1. Initialization: Top-w semantic matches
        semantic_scores = []
        for i, page_emb in enumerate(self.index.embeddings):
            score = self.retriever.get_semantic_score(query_emb, page_emb)
            semantic_scores.append((i, score))
        
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        start_nodes = [node for node, score in semantic_scores[:w]]
        
        exploration_set = set(start_nodes)
        visited = set()
        final_scores = {}

        # 2. Iterative Hopping (BFS style)
        current_frontier = list(start_nodes)
        
        for hop in range(n_hops + 1):
            next_frontier = []
            for node in current_frontier:
                if node in visited:
                    continue
                
                # Get logical score from VLM
                page_img = self.index.page_images[node]
                logical_score = self.retriever.get_logical_score(query, page_img)
                
                # Combine: Final score is average of semantic and logical
                # Fetch semantic score
                s_sem = next(s for n, s in semantic_scores if n == node)
                combined_score = (s_sem + logical_score) / 2.0
                final_scores[node] = combined_score
                
                visited.add(node)
                
                # Expand to neighbors
                for neighbor in self.index.graph.neighbors(node):
                    if neighbor not in visited:
                        next_frontier.append(neighbor)
            
            # Limited traversal to keep it efficient
            current_frontier = next_frontier[:5] # Max 5 new nodes per hop
            if not current_frontier:
                break
                
        # 3. Final Re-ranking
        ranked_pages = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_pages

if __name__ == "__main__":
    print("MoLoRAGTraversal implemented.")
