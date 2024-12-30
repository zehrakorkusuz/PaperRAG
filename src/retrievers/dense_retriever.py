# dense_retriever.py
from retrievers.base_retriever import BaseRetriever
from embedding_models.base_embedding_model import BaseEmbeddingModel
from vector_db.base_db import BaseVectorDB
from typing import List, Tuple
import numpy as np

class DenseRetriever(BaseRetriever):
    def __init__(self, embedding_model: BaseEmbeddingModel, vector_db: BaseVectorDB):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def retrieve(self, query_vector: np.ndarray, top_k: int = 5, modality: str = "text") -> List[Tuple[float, dict]]:
        """
        Retrieve top-k results based on dense retrieval for the specified modality.

        Args:
            query_vector (np.ndarray): The vector representing the query.
            top_k (int): The number of top results to return.
            modality (str): The modality to query within (e.g., "text").

        Returns:
            List[Tuple[float, dict]]: List of the top-k results with their scores.
        """
        results = self.vector_db.query(query_vector, top_k=top_k)
        # # Assuming the vector_db.query method returns distances, convert them to similarity scores if needed
        # scores = [1 / (1 + np.exp(distance)) for distance, _ in results] # Convert distances to similarity scores
        # return [(score, meta) for score, (_, meta) in zip(scores, results)]
        results = self.vector_db.query(query_vector, top_k=top_k)
        return [(distance, meta) for distance, meta in results]

    def add_documents(self, documents: List[str], modality: str = "text") -> None:
        """
        Add documents to the dense retriever.

        Args:
            documents (List[str]): List of documents to add.
            modality (str): The modality of the documents (e.g., "text", "image").
        """
        if modality == "text":
            vectors = self.embedding_model.generate_text_embeddings(documents)
            metadata = [{"text": doc} for doc in documents]
        elif modality == "image":
            vectors = self.embedding_model.generate_image_embeddings(documents)
            metadata = [{"image_path": doc} for doc in documents]
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        self.vector_db.add_vectors(vectors, metadata, modality=modality)

    def reset(self) -> None:
        """
        Reset the dense retriever by clearing all vectors in the vector database.
        """
        self.vector_db.reset_database()