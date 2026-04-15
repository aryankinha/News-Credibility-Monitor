import os
import chromadb
from sentence_transformers import SentenceTransformer

# Optional Streamlit support for caching
try:
    import streamlit as st
except ImportError:
    st = None

CHROMA_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "chroma_db")
COLLECTION_NAME = "isot_news"

def cache_resource_fallback(func):
    """Fallback decorator if streamlit is not present."""
    if st is not None and hasattr(st, "cache_resource"):
        return st.cache_resource()(func)
    return func

@cache_resource_fallback
def get_embedding_model():
    print("Loading embedding model (this should happen only once in an active session)...")
    return SentenceTransformer('all-MiniLM-L6-v2')

@cache_resource_fallback
def get_chroma_client():
    print("Loading ChromaDB client...")
    return chromadb.PersistentClient(path=CHROMA_DB_DIR)

def retrieve_similar_news(query, k=3):
    """
    Accepts text input, converts it to an embedding, and retrieves 
    top-k similar documents from the local Chroma database.
    """
    model = get_embedding_model()
    
    # We clean the query same as we did docs (optional, but good practice)
    from src.utils.text_cleaner import clean_text
    query_clean = clean_text(query)
    if not query_clean.strip():
        # Fallback to raw if cleaner stripped it completely
        query_clean = query
        
    client = get_chroma_client()
    
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error accessing collection '{COLLECTION_NAME}': {e}. Please build the DB first.")
        return []
    
    query_embedding = model.encode([query_clean]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    
    retrieved_docs = []
    if results and 'documents' in results and len(results['documents'][0]) > 0:
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for doc, meta, dist in zip(docs, metadatas, distances):
            retrieved_docs.append({
                "text": doc,
                "metadata": meta,
                "distance": dist
            })
            
    return retrieved_docs

def test_retriever():
    sample_text = "Donald Trump signs executive order withdrawing US from TPP."
    print(f"Testing retrieval with query: '{sample_text}'\n")
    
    results = retrieve_similar_news(sample_text, k=3)
    if not results:
        print("No results found. Did you run build_db.py yet?")
        return
        
    for i, res in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"Label: {res['metadata'].get('label')}")
        print(f"Subject: {res['metadata'].get('subject')}")
        print(f"Source: {res['metadata'].get('source')}")
        print(f"Text Snippet: {res['text'][:150]}...\n")

if __name__ == "__main__":
    test_retriever()
