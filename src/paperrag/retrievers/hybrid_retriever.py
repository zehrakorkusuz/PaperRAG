from typing import List, Tuple, Dict, Optional
from paperrag.retrievers.sparse_retriever import SparseRetriever
from paperrag.retrievers.dense_retriever import DenseRetriever
import numpy as np
import logging

logger = logging.getLogger(__name__)

class HybridRetriever:
    def __init__(self, sparse_retriever: SparseRetriever, dense_retriever: DenseRetriever):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        logger.info("HybridRetriever initialized.")

    def retrieve(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[np.ndarray] = None,
        query_vector: Optional[np.ndarray] = None,
        top_k: int = 5,
        modality: str = "all",
        weights: Tuple[float, float] = (0.5, 0.5),
        filter_metadata: Optional[Dict[str, any]] = None
    ) -> List[Tuple[float, Dict[str, any]]]:
        """
        Retrieve top-k results combining sparse and dense retrieval mechanisms.

        Args:
            query_text (Optional[str]): Text query string, if any.
            query_image (Optional[np.ndarray]): Image query vector, if any.
            top_k (int): Number of top results to return.
            modality (str): Modality to filter results ("all", "text", "image").
            weights (Tuple[float, float]): Weights for sparse and dense scores.
            filter_metadata (Optional[Dict[str, any]]): Metadata filters.

        Returns:
            List[Tuple[float, Dict[str, any]]]: Combined ranked results.
        """
        sparse_results = []
        dense_results = []

        # Sparse retrieval for text queries
        if query_text and isinstance(query_text, str):
            logger.info(f"Performing sparse retrieval for text query: '{query_text}'")
            sparse_results = self.sparse_retriever.retrieve(query_text, top_k)

            # Use the provided query_vector for dense retrieval
            if query_vector is not None:
                logger.info("Performing dense retrieval for text query.")
                if modality in ["all", "text"]:
                    dense_results += self.dense_retriever.retrieve(query_vector, top_k, modality="text")
                if modality in ["all", "image"]:
                    dense_results += self.dense_retriever.retrieve(query_vector, top_k, modality="image")
            else:
                logger.info("Generating dense embeddings for text query.")
                query_vector = self.dense_retriever.embedding_model.generate_text_embeddings([query_text])[0]

                logger.info("Performing dense retrieval for text query.")
                if modality in ["all", "text"]:
                    dense_results += self.dense_retriever.retrieve(query_vector, top_k, modality="text")
                if modality in ["all", "image"]:
                    dense_results += self.dense_retriever.retrieve(query_vector, top_k, modality="image")

        # Dense retrieval for image queries
        if query_image is not None:
            logger.info("Generating dense embeddings for image query.")
            query_vector = self.dense_retriever.embedding_model.generate_image_embeddings([query_image])[0]

            logger.info("Performing dense retrieval for image query.")
            if modality in ["all", "image"]:
                dense_results += self.dense_retriever.retrieve(query_vector, top_k, modality="image")
            if modality in ["all", "text"]:
                dense_results += self.dense_retriever.retrieve(query_vector, top_k, modality="text")

        if not sparse_results and not dense_results:
            logger.info("No queries provided.")
            return []

        # Normalize scores using z-score normalization
        sparse_scores = np.array([score for score, _ in sparse_results])
        dense_scores = np.array([1 / (1 + score) for score, _ in dense_results])  # Convert FAISS distances to similarities

        normalized_sparse = self._z_score_normalize(sparse_scores)
        normalized_dense = self._z_score_normalize(dense_scores)

        # Combine scores with weights
        combined_scores = {}
        self._update_combined_scores(combined_scores, sparse_results, normalized_sparse, weights[0], "text")
        self._update_combined_scores(combined_scores, dense_results, normalized_dense, weights[1], "dense")

        # Filter by requested modality
        if modality in ["text", "image", "audio"]:
            combined_scores = {
                k: v for k, v in combined_scores.items()
                if v["meta"].get("type") == modality
            }

        # Sort and take top_k results
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        filtered_results = [(x[1]["score"], x[1]["meta"]) for x in sorted_results[:top_k]]

        # Apply optional metadata filters
        if filter_metadata:
            filtered_results = [
                r for r in filtered_results
                if all(r[1].get(k) == v for k, v in filter_metadata.items())
            ]

        logger.info(f"Returning {len(filtered_results)} combined results after filtering.")
        return filtered_results

    @staticmethod
    def _z_score_normalize(scores: np.ndarray) -> np.ndarray:
        """
        P.S. This method transforms the scores to have a mean of 0 and a standard deviation of 1.
        It better handles outliers and ensures scale independence, making it suitable for
        scenarios such as cross-modality search.
        Args:
            scores (np.ndarray): An array of scores to be normalized.
        Returns:
            np.ndarray: The z-score normalized scores. If the input array is empty, returns an empty array.
        """

        if len(scores) == 0:
            return np.array([])
        mean = np.mean(scores)
        std = np.std(scores)
        return (scores - mean) / std if std > 0 else np.zeros_like(scores)

    @staticmethod
    def _update_combined_scores(combined_scores, results, normalized_scores, weight, result_type):
        for idx, (score, meta) in enumerate(results):
            key_component = meta.get('text') or meta.get('image_path') or meta.get('audio_path', 'unknown')
            key = f"{meta.get('type', result_type)}:{key_component}"
            if key not in combined_scores:
                combined_scores[key] = {"score": 0.0, "meta": meta}
            combined_scores[key]["score"] += weight * normalized_scores[idx]
