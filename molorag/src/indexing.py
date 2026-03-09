import os
import fitz  # PyMuPDF
import torch
import networkx as nx
import numpy as np
from PIL import Image
from transformers import AutoProcessor, CLIPModel
import matplotlib.pyplot as plt

class DocumentGraphIndex:
    def __init__(self, model_name="openai/clip-vit-large-patch14", threshold=0.7):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # CLIPModel handles both vision and text with aligned projections
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.threshold = threshold
        self.graph = nx.Graph()
        self.page_images = []
        self.embeddings = []

    def load_pdf(self, pdf_path):
        """Convert PDF pages to images and store them."""
        doc = fitz.open(pdf_path)
        self.page_images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            self.page_images.append(img)
        doc.close()
        print(f"Loaded {len(self.page_images)} pages from {pdf_path}")

    def generate_embeddings(self):
        """Generate CLIP embeddings for each page."""
        self.embeddings = []
        self.model.eval()
        with torch.no_grad():
            for img in self.page_images:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                outputs = self.model.get_image_features(**inputs)
                # Ensure we have the tensor
                image_features = outputs if torch.is_tensor(outputs) else outputs.pooler_output
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                pooled_emb = image_features.cpu().numpy().flatten()
                self.embeddings.append(pooled_emb)
        print(f"Generated embeddings for {len(self.embeddings)} pages")

    def build_graph(self):
        """Construct the Page Graph based on similarity threshold."""
        num_pages = len(self.embeddings)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(num_pages))

        for i in range(num_pages):
            for j in range(i + 1, num_pages):
                # Cosine similarity (already normalized)
                similarity = np.dot(self.embeddings[i], self.embeddings[j])
                
                if similarity > self.threshold:
                    self.graph.add_edge(i, j, weight=float(similarity))
        
        # Connect consecutive pages
        for i in range(num_pages - 1):
            if not self.graph.has_edge(i, i + 1):
                self.graph.add_edge(i, i + 1, weight=0.5)

        print(f"Graph built with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")

    def save_graph_image(self, output_path):
        """Save a visualization of the graph."""
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', font_weight='bold', node_size=500)
        plt.title("Document Page Graph")
        plt.savefig(output_path)
        plt.close()

if __name__ == "__main__":
    # Test script
    pdf_file = "data/paper.pdf"
    if os.path.exists(pdf_file):
        indexer = DocumentGraphIndex()
        indexer.load_pdf(pdf_file)
        indexer.generate_embeddings()
        indexer.build_graph()
        indexer.save_graph_image("output/graph.png")
        print("Indexing completed and graph visualization saved to output/graph.png")
    else:
        print(f"File not found: {pdf_file}")
