from embedding_model import CLIPEmbeddingModel
import time, logging
from faiss_db import FAISSDatabase

logging.basicConfig(level=logging.INFO)

def test_model_with_faiss():
    model = CLIPEmbeddingModel()
    logging.info("Model initialized.")

    # Text embeddings
    texts = ["The quick brown fox jumps over the lazy dog.", "An apple a day keeps the doctor away."]
    start_time = time.time()
    text_embeddings = model.generate_text_embeddings(texts)
    end_time = time.time()
    logging.info(f"Text embeddings shape: {text_embeddings.shape}, Time taken: {end_time - start_time:.2f} seconds")

    # Image embeddings
    image_paths = ["../../data/figures/APOE4-Figure1-1.png", "../../data/figures/APOE4-Figure4-1.png"]
    start_time = time.time()
    image_embeddings = model.generate_image_embeddings(image_paths)
    end_time = time.time()
    logging.info(f"Image embeddings shape: {image_embeddings.shape}, Time taken: {end_time - start_time:.2f} seconds")

    # Initialize FAISS database
    faiss_db = FAISSDatabase(dimension=model.embedding_dimension, metric='cosine', index_type='flat')

    # Add text embeddings
    text_indices = faiss_db.add_vectors(text_embeddings, metadatas=[{'type': 'text', 'content': text} for text in texts])
    logging.info(f"Added {len(text_indices)} text embeddings to FAISS database.")

    # Add image embeddings
    image_indices = faiss_db.add_vectors(image_embeddings, metadatas=[{'type': 'image', 'path': path} for path in image_paths])
    logging.info(f"Added {len(image_indices)} image embeddings to FAISS database.")

    # Query embedding
    query = "Statistics about APOE4: Multiomics Analysis of Alzheimer's Disease"
    query_embedding = model.embed_query(query)
    logging.info(f"Query embedding shape: {query_embedding.shape}")

    # Search in FAISS database
    start_time = time.time()
    distances, indices, metadatas = faiss_db.search(query_embedding, k=4)  # Searching for 2 nearest neighbors
    end_time = time.time()
    logging.info(f"Search completed in {end_time - start_time:.2f} seconds")

    #rerank based on distance and print the content
    search_results = [{"distance": distances[i], "index": indices[i], "metadata": metadatas[i]} for i in range(len(distances))]

    # Rerank the search results based on distance
    reranked_results = sorted(search_results, key=lambda x: x['distance'])

    # Extract the sorted distances, indices, and metadatas
    # sorted_distances = [result['distance'] for result in reranked_results]
    # sorted_indices = [result['index'] for result in reranked_results]
    # sorted_metadatas = [result['metadata'] for result in reranked_results]

    # Print reranked results
    for result in reranked_results:
        print(f"Distance: {result['distance']}, Index: {result['index']}, Metadata: {result['metadata']}")


if __name__ == '__main__':
    test_model_with_faiss()