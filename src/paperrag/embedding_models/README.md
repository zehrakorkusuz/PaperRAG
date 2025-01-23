# Embedding Models

This repository contains various embedding models used for different modalities. The current structure of the repository is as follows:

## Directory Structure

```
embedding_models/
├── base_model.py
├── text_embedding_model.py
├── image_embedding_model.py
├── audio_embedding_model.py
└── README.md
```

## How to Use

1. **Base Model**: All embedding models inherit from the `BaseEmbeddingModel` class defined in `base_model.py`. This class provides common functionalities and interfaces for all embedding models.

2. **Text Embedding Model**: The `text_embedding_model.py` file contains the implementation of the text embedding model. To use it:
    ```python
    from text_embedding_model import TextEmbeddingModel

    model = TextEmbeddingModel()
    embeddings = model.embed(["sample text"])
    ```

3. **Image Embedding Model**: The `image_embedding_model.py` file contains the implementation of the image embedding model. To use it:
    ```python
    from image_embedding_model import ImageEmbeddingModel

    model = ImageEmbeddingModel()
    embeddings = model.embed(["path/to/image.jpg"])
    ```

4. **Audio Embedding Model**: The `audio_embedding_model.py` file contains the implementation of the audio embedding model. To use it:
    ```python
    from audio_embedding_model import AudioEmbeddingModel

    model = AudioEmbeddingModel()
    embeddings = model.embed(["path/to/audio.wav"])
    ```

## How to Add a New Modality

1. **Create a New File**: Create a new Python file for your modality, e.g., `video_embedding_model.py`.

2. **Inherit from Base Model**: In your new file, create a class that inherits from `BaseEmbeddingModel`.

3. **Implement the `embed` Method**: Implement the `embed` method to handle the specific embedding logic for your modality.

    ```python
    from base_model import BaseEmbeddingModel

    class VideoEmbeddingModel(BaseEmbeddingModel):
        def embed(self, inputs):
            # Implement video embedding logic here
            pass
    ```
