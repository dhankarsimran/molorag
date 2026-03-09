import os
import sys
# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from indexing import DocumentGraphIndex
from retriever import LogicAwareRetriever
from traversal import MoLoRAGTraversal
from generation import MoLoRAGGenerator

def run_full_pipeline(pdf_path, question):
    print("--- Starting MoLoRAG Pipeline ---")
    
    # 1. Indexing
    print("Step 1: Indexing...")
    index = DocumentGraphIndex()
    index.load_pdf(pdf_path)
    index.generate_embeddings()
    index.build_graph()
    
    # 2. Setup Retriever and Traversal
    print("Step 2: Initializing Retrieval Engine...")
    retriever = LogicAwareRetriever(device=index.device)
    traversal = MoLoRAGTraversal(index, retriever)
    
    # 3. Graph Traversal
    print(f"Step 3: Traversal for query: '{question}'")
    ranked_pages = traversal.run_traversal(question)
    
    # Select top-K pages
    top_k = 3
    retrieved_indices = [idx for idx, score in ranked_pages[:top_k]]
    retrieved_images = [index.page_images[i] for i in retrieved_indices]
    print(f"Retrieved pages: {retrieved_indices}")
    
    # 4. Generation
    print("Step 4: Generating Answer...")
    generator = MoLoRAGGenerator(device=index.device)
    answer = generator.generate_answer(question, retrieved_images)
    
    print("\n--- Final Answer ---")
    print(answer)
    return answer

if __name__ == "__main__":
    pdf = "data/paper.pdf"
    q = "What are the average improvements of MoLoRAG in accuracy and retrieval precision?"
    if os.path.exists(pdf):
        run_full_pipeline(pdf, q)
    else:
        print(f"PDF not found at {pdf}")
