from typing import List, Tuple, Optional, Dict
import numpy as np

class BaseVectorDB:
    def add_vectors(self, vectors: np.ndarray, metadata: List[dict], modality: Optional[str] = None) -> None:
        """
        Add vectors to the database with associated metadata and modality.

        Args:
            vectors (np.ndarray): Array of vectors to add.
            metadata (List[dict]): Metadata for each vector.
            modality (Optional[str]): The modality of the embeddings (e.g., 'text', 'image').
        """
        raise NotImplementedError("This method should be overridden.")

    def query(self, vector: np.ndarray, top_k: int = 5, modality: Optional[str] = None) -> List[Tuple[np.ndarray, dict]]:
        """
        Query the database for the closest vectors to the given vector.

        Args:
            vector (np.ndarray): The query vector.
            top_k (int): Number of nearest neighbors to retrieve.
            modality (Optional[str]): The modality to query within (e.g., 'text', 'image').

        Returns:
            List[Tuple[np.ndarray, dict]]: List of (vector, metadata) tuples.
        """
        raise NotImplementedError("This method should be overridden.")

    def reset_database(self) -> None:
        """Clears all data in the database."""
        raise NotImplementedError("This method should be overridden.")

    def delete_vectors(self, ids: List[int]) -> None:
        """
        Deletes vectors by their IDs.

        Args:
            ids (List[int]): List of IDs to delete.
        """
        raise NotImplementedError("This method should be overridden.")

    def list_metadata(self) -> List[dict]:
        """
        Returns a list of all metadata stored in the database.

        Returns:
            List[dict]: Metadata of all vectors.
        """
        raise NotImplementedError("This method should be overridden.")

    def save(self, file_path: str) -> None:
        """Save the database to a file."""
        raise NotImplementedError("This method should be overridden.")

    def load(self, file_path: str) -> None:
        """Load the database from a file."""
        raise NotImplementedError("This method should be overridden.")
