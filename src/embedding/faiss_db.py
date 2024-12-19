import faiss
import logging
import numpy as np
from typing import List, Optional, Union, Tuple, Any
import os, json

print(faiss.__version__)
print(faiss.METRIC_INNER_PRODUCT)
# Add GPU support

class FAISSDatabase:
    def __init__(
        self, 
        dimension: int, 
        metric: str = 'l2', 
        index_type: str = 'flat', 
        nlist: int = 100,
        nprobe: int = 10,
        gpu: bool = False,
        persist: bool = False,
        path: str = "faiss_db"
    ):
        """
        Initialize FAISS database with configurable parameters.
        
        :param dimension: Dimensionality of vectors
        :param metric: Distance metric ('l2', 'ip', 'cosine')
        :param index_type: Index type ('flat', 'ivf', 'hnsw')
        :param nlist: Number of clusters for IVF index
        :param nprobe: Number of clusters to probe during search
        :param gpu: Use GPU index if available
        :param persist: If True, saves the index and metadata on object destruction
        :param path: Directory path to save/load the index and metadata
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        self.dimension = dimension
        self.metric = metric
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        self.gpu = gpu
        self.persist = persist
        self.path = path
        
        self.metadata = {}
        
        if self.persist and os.path.exists(self.path):
            self.load()
        else:
            self._create_index()
    
    def _create_index(self):
        """
        Create FAISS index based on specified parameters.
        """
        try:
            # Determine metric
            if self.metric == 'l2':
                metric_type = faiss.METRIC_L2
            elif self.metric in ['ip', 'inner_product']:
                metric_type = faiss.METRIC_INNER_PRODUCT
            elif self.metric == 'cosine':
                metric_type = faiss.METRIC_INNER_PRODUCT
                # Normalize vectors for cosine similarity
                self.normalize_vectors = True
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
            
            # Base quantizer
            quantizer = faiss.IndexFlatL2(self.dimension)
            
            # Create index based on type
            if self.index_type == 'flat':
                self.index = faiss.IndexFlatL2(self.dimension) if self.metric == 'l2' else faiss.IndexFlatIP(self.dimension)
            
            elif self.index_type == 'ivf':
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist, metric_type)
                # Train the index
                training_data = np.random.random((1000, self.dimension)).astype('float32')
                self.index.train(training_data)
                self.index.nprobe = self.nprobe
            
            elif self.index_type == 'hnsw':
                # HNSW index with inner product or L2
                self.index = faiss.IndexHNSWFlat(self.dimension, 32, metric_type)
            
            else:
                raise ValueError(f"Unsupported index type: {self.index_type}")
            
            # Optional GPU support
            # if self.gpu:
            #     try:
            #         res = faiss.StandardGpuResources()
            #         self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            #     except Exception as gpu_error:
            #         self.logger.warning(f"GPU index creation failed: {gpu_error}")
            
            self.logger.info(f"Created FAISS index: {self.index_type} with {self.metric} metric")
        
        except Exception as e:
            self.logger.error(f"Index creation failed: {e}")
            raise

    def load(self):
        """
        Load the FAISS index and metadata from disk if persist is True.
        """
        if not self.persist:
            return
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(self.path, "index.faiss"))
        self.logger.info(f"FAISS index loaded from {self.path}")
        
        # Load metadata
        with open(os.path.join(self.path, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        self.logger.info(f"Metadata loaded from {self.path}")

    def __del__(self):
        """
        Destructor method to save the index and metadata if persist is True.
        """
        if self.persist:
            self.save()

    def save(self):
        """
        Save the FAISS index and metadata to disk.
        """
        os.makedirs(self.path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(self.path, "index.faiss"))
        self.logger.info(f"FAISS index saved to {self.path}")
        
        # Save metadata
        with open(os.path.join(self.path, "metadata.json"), 'w') as f:
            json.dump(self.metadata, f)
        self.logger.info(f"Metadata saved to {self.path}")
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        metadatas: Optional[List[Any]] = None
    ) -> List[int]:
        """
        Add multiple vectors to the database.
        
        :param vectors: NumPy array of vectors
        :param metadatas: Optional list of metadata for each vector
        :return: List of vector indices
        """
        # Validate input
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector must have {self.dimension} dimensions")
        
        # Normalize if cosine similarity
        if getattr(self, 'normalize_vectors', False):
            vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
        
        # Convert to float32
        vectors = vectors.astype(np.float32)
        
        # Add vectors
        self.index.add(vectors)
        
        # Store metadata
        start_index = self.index.ntotal - len(vectors)
        if metadatas:
            for i, metadata in enumerate(metadatas):
                self.metadata[start_index + i] = metadata
        
        return list(range(start_index, self.index.ntotal))
    
    def search(
        self, 
        query: np.ndarray, 
        k: int = 5, 
        filter_fn: Optional[callable] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
        """
        Search for k nearest neighbors.
        
        :param query: Query vector
        :param k: Number of neighbors to retrieve
        :param filter_fn: Optional function to filter results
        :return: Distances, indices, and metadata of nearest neighbors
        """
        # Validate query
        if query.shape[1] != self.dimension:
            raise ValueError(f"Query must have {self.dimension} dimensions")
        
        # Normalize if cosine similarity
        if getattr(self, 'normalize_vectors', False):
            query = query / np.linalg.norm(query, axis=1)[:, np.newaxis]
        
        # Ensure query is float32
        query = query.astype(np.float32)
        
        # Perform search
        distances, indices = self.index.search(query, k)
        
        # Retrieve metadata and apply optional filter
        results_metadata = []
        filtered_distances = []
        filtered_indices = []
        
        for i in range(query.shape[0]):  # Handle multiple queries
            for dist, idx in zip(distances[i], indices[i]):
                metadata = self.metadata.get(idx, None)
                
                # Apply filter if provided
                if filter_fn is None or filter_fn(metadata):
                    results_metadata.append(metadata)
                    filtered_distances.append(dist)
                    filtered_indices.append(idx)
        
        return (
            np.array(filtered_distances), 
            np.array(filtered_indices), 
            results_metadata
        )
    
    def remove_vector(self, index: int):
        """
        Remove a vector by its index for flat indices.
        
        :param index: Index of vector to remove
        """
        if self.index_type == 'flat':
            # Since we can't directly access vectors, we'll reconstruct them
            vectors = []
            for i in range(self.index.ntotal):
                if i != index:
                    # Reconstruct each vector if not the one we're removing
                    vectors.append(self.index.reconstruct(i))
            vectors = np.array(vectors, dtype=np.float32)
            
            # Recreate the index
            self._create_index()
            self.index.add(vectors)
        else:
            raise NotImplementedError("Vector removal not implemented for this index type.")

        # Remove corresponding metadata
        if index in self.metadata:
            del self.metadata[index]
        
        # Adjust metadata indices
        new_metadata = {}
        for i, (k, v) in enumerate(self.metadata.items()):
            if k > index:
                new_metadata[k - 1] = v
            else:
                new_metadata[k] = v
        self.metadata = new_metadata
    
    def get_vector_count(self) -> int:
        """
        Get the number of vectors in the database.
        
        :return: Total number of vectors
        """
        return self.index.ntotal
    
    def export_database(self) -> Tuple[np.ndarray, List[Any]]:
        """
        Export all vectors and their metadata.
        
        :return: Tuple of vectors and metadata
        """
        vectors = faiss.vector_to_array(self.index).reshape(-1, self.dimension)
        metadatas = [self.metadata.get(i, None) for i in range(vectors.shape[0])]
        return vectors, metadatas
    
    def train(self, training_data: np.ndarray):
        """
        Train the index with additional data.
        
        :param training_data: NumPy array of training vectors
        """
        if not hasattr(self.index, 'train'):
            raise ValueError("Current index type does not support training")
        
        if training_data.shape[1] != self.dimension:
            raise ValueError(f"Training data must have {self.dimension} dimensions")
        
        self.index.train(training_data.astype(np.float32))
    
    def reset(self):
        """
        Reset the index, removing all vectors.
        """
        self.index.reset()
        self.metadata.clear()

def main():

    # Setup for testing
    dimension = 64
    index_type = 'flat'
    metric = 'l2'
    db_path = "database"

    # Initialize FAISSDatabase with persistence
    faiss_db = FAISSDatabase(dimension=dimension, metric=metric, index_type=index_type, persist=True, path=db_path)

    # Create some random data
    nb = 1000  # database size
    np.random.seed(1234)
    xb = np.random.random((nb, dimension)).astype('float32')

    # Metadata can be in various formats
    # metadatas = [f"Vector_{i}" for i in range(nb)]
    metadatas = [{"id": i, "description": f"Vector_{i}"} for i in range(nb)]
    
    # Add vectors to the database along with metadata
    indices = faiss_db.add_vectors(xb, metadatas=metadatas)
    print(f"Number of vectors added: {len(indices)}")
    print(f"Total vectors in database: {faiss_db.get_vector_count()}")

    # Search for nearest neighbors
    nq = 1  # number of queries
    xq = np.random.random((nq, dimension)).astype('float32')

    # Option 1: Search with all query vectors ### Implement the batch logic instead VectorDB manager for search
    distances, indices, metadatas = faiss_db.search(xq, k=3)
    
    # Option 2: Search with a single query vector
    # single_query = xq[0].reshape(1, -1)
    # distances, indices, metadatas = faiss_db.search(single_query, k=5)

    print("Search Results:")
    for dist, idx, meta in zip(distances, indices, metadatas):
        print(f"Distance: {dist}, Index: {idx}, Metadata: {meta or 'None'}")

    # Remove a vector (first one)
    faiss_db.remove_vector(0)
    print(f"After removal, vectors count: {faiss_db.get_vector_count()}")

    # Reset the database
    faiss_db.reset()
    print(f"After reset, vectors count: {faiss_db.get_vector_count()}")

if __name__ == "__main__":
    main()