# faiss_db.py 
import faiss
import numpy as np
from typing import List, Tuple, Optional, Dict

class FAISSVectorDB:
    """
    A FAISS-based vector database for storing and retrieving vectors with associated metadata.
    Supports multimodal data through separate indices for each modality.

    Attributes:
        dimension (int): The dimensionality of the vectors being stored.
        indices (Dict[str, faiss.IndexFlatL2]): A dictionary mapping each modality 
            ("text", "image", "audio", etc.) to its corresponding FAISS index.
        metadata (Dict[str, List[Dict[str, any]]]): A dictionary mapping each modality 
            to a list of metadata associated with the stored vectors.

    Methods:
        add_vectors(vectors: np.ndarray, metadata: List[Dict[str, any]], modality: str) -> None:
            Adds vectors and their metadata to the database for a specified modality.
        
        query(vector: np.ndarray, top_k: int = 5, modality: Optional[str] = None) -> List[Tuple[float, Dict[str, any]]]:
            Queries the database to retrieve the top-k closest vectors for a given query vector.
            Optionally filters results by modality.
        
        reset_database(modality: Optional[str] = None) -> None:
            Resets the database by clearing all vectors and metadata.
            If a modality is specified, only that modality is reset.
        
        list_metadata(modality: Optional[str] = None) -> List[Dict[str, any]]:
            Lists all metadata stored in the database. If a modality is specified, 
            only metadata for that modality is returned.
        
        save(file_path: str) -> None:
            Saves all indices to files, with one file per modality. The file names are 
            suffixed with the modality name (e.g., "file_path_text.index").
        
        load(file_path: str) -> None:
            Loads indices from files, with one file per modality. The file names must be 
            suffixed with the modality name (e.g., "file_path_text.index").

    Example:
        >>> db = FAISSVectorDB(dimension=128)
        >>> text_vectors = np.random.rand(10, 128).astype(np.float32)
        >>> text_metadata = [{"text": f"Document {i}"} for i in range(10)]
        >>> db.add_vectors(text_vectors, text_metadata, modality="text")
        
        >>> query_vector = np.random.rand(1, 128).astype(np.float32)
        >>> results = db.query(query_vector, top_k=5, modality="text")
        >>> for score, meta in results:     
        >>>     print(f"Score: {score}, Metadata: {meta}")
    """
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.indices = {
            "text": faiss.IndexFlatL2(dimension),
            "image": faiss.IndexFlatL2(dimension),
            "audio": faiss.IndexFlatL2(dimension)
        }
        self.metadata = {
            "text": [],
            "image": [],
            "audio": []
        }

    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict[str, any]], modality: str) -> None:
        if modality not in self.indices:
            raise ValueError(f"Unsupported modality: {modality}")
        
        self.indices[modality].add(vectors)
        self.metadata[modality].extend(metadata)

    def query(self, vector: np.ndarray, top_k: int = 5, modality: Optional[str] = None) -> List[Tuple[float, Dict[str, any]]]:
        results = []

        modalities_to_query = [modality] if modality and modality in self.indices else self.indices.keys()

        for mod in modalities_to_query:
            distances, indices = self.indices[mod].search(vector.reshape(1, -1), top_k)
            for i, idx in enumerate(indices[0]):
                if idx != -1:
                    results.append((distances[0][i], self.metadata[mod][int(idx)]))
        
        # Sort results by distance (lower is better)
        results.sort(key=lambda x: x[0])
        return results[:top_k]

    def reset_database(self, modality: Optional[str] = None) -> None:
        if modality:
            if modality not in self.indices:
                raise ValueError(f"Unsupported modality: {modality}")
            self.indices[modality] = faiss.IndexFlatL2(self.dimension)
            self.metadata[modality] = []
        else:
            for mod in self.indices:
                self.indices[mod] = faiss.IndexFlatL2(self.dimension)
                self.metadata[mod] = []

    def list_metadata(self, modality: Optional[str] = None) -> List[Dict[str, any]]:
        if modality:
            if modality not in self.metadata:
                raise ValueError(f"Unsupported modality: {modality}")
            return self.metadata[modality]
        else:
            all_metadata = []
            for mod in self.metadata:
                all_metadata.extend(self.metadata[mod])
            return all_metadata

    def save(self, file_path: str) -> None:
        for mod, index in self.indices.items():
            faiss.write_index(index, f"{file_path}_{mod}.index")

    def load(self, file_path: str) -> None:
        for mod in self.indices.keys():
            self.indices[mod] = faiss.read_index(f"{file_path}_{mod}.index")
