# Vector Database

This repository contains the implementation of a vector database using FAISS for multimodal data. The database supports adding and querying vectors for different modalities such as text and images.

## Directory Structure

```
vector_db/
├── db_factory.py
├── README.md
```

## How to Use

1. **Setup the Database**: Use the `get_vector_db` function from `db_factory.py` to create a vector database.

    ```python
    from db.db_factory import get_vector_db
    import numpy as np

    # Create a multimodal FAISS DB
    dimension = 512
    db = get_vector_db("faiss", dimension=dimension)
    ```

2. **Add Vectors**: Add text and image vectors to the database along with their metadata.

    ```python
    # Add text and image vectors
    text_vectors = np.random.rand(5, dimension).astype(np.float32)
    image_vectors = np.random.rand(5, dimension).astype(np.float32)
    text_metadata = [{"id": i, "type": "text", "info": f"Text {i}"} for i in range(5)]
    image_metadata = [{"id": i, "type": "image", "info": f"Image {i}"} for i in range(5)]

    db.add_vectors(text_vectors, text_metadata, modality="text")
    db.add_vectors(image_vectors, image_metadata, modality="image")
    ```

3. **Query the Database**: Query the database using a vector to retrieve the most similar vectors and their metadata.

    ```python
    # Query
    query_vector = np.random.rand(dimension).astype(np.float32)
    results = db.query(query_vector, modality="text")
    for vector, meta in results:
        print("Metadata:", meta)
    ```

