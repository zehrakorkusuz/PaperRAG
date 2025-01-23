from typing import List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseEmbeddingModel:
    """
    Base class for embedding models. This class provides a common interface for generating
    embeddings from different types of data such as text, images, audio, video, and tabular data.
    """

    def __init__(self):
        """
        Initialize the BaseEmbeddingModel.
        """
        logger.info("BaseEmbeddingModel initialized")

    def generate_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.

        :param texts: List of text strings to generate embeddings for.
        :return: List of text embeddings.
        """
        logger.debug("generate_text_embeddings called with texts: %s", texts)
        raise NotImplementedError("This method should be overridden.")

    def generate_image_embeddings(self, image_paths: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of image file paths.

        :param image_paths: List of image file paths to generate embeddings for.
        :return: List of image embeddings.
        """
        logger.debug("generate_image_embeddings called with image_paths: %s", image_paths)
        raise NotImplementedError("This method should be overridden.")

    def generate_audio_embeddings(self, audio_paths: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of audio file paths.

        :param audio_paths: List of audio file paths to generate embeddings for.
        :return: List of audio embeddings.
        """
        logger.debug("generate_audio_embeddings called with audio_paths: %s", audio_paths)
        raise NotImplementedError("This method should be overridden.")

    def generate_video_embeddings(self, video_paths: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of video file paths.

        :param video_paths: List of video file paths to generate embeddings for.
        :return: List of video embeddings.
        """
        logger.debug("generate_video_embeddings called with video_paths: %s", video_paths)
        raise NotImplementedError("This method should be overridden.")

    def generate_tabular_embeddings(self, data: List[List[float]]) -> List[List[float]]:
        """
        Generate embeddings for tabular data.

        :param data: List of tabular data to generate embeddings for.
        :return: List of tabular data embeddings.
        """
        logger.debug("generate_tabular_embeddings called with data shape: %s", len(data))
        raise NotImplementedError("This method should be overridden.")
    
    @staticmethod
    def get_embedding_dim(self) -> int:
        """
        Returns the dimensionality of the embeddings.

        :return: Dimensionality of the embeddings.
        """
        raise NotImplementedError("This method should be overridden.")
    
    @staticmethod
    def available_models(self) -> List[str]:
        """
        Returns a list of available models.

        :return: List of available models.
        """
        raise NotImplementedError("This method should be overridden.")
