import logging
import torch
import clip
import numpy as np
from typing import List, Union
from PIL import Image
import time
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "../../models/clip-vit-base-patch32")

class CLIPEmbeddingModel:
    """
    Manages embedding generation using CLIP model.
    Provides methods for generating embeddings for text, images, and queries.
    """
    def __init__(self, model_name: str = "ViT-B/32"):
        """
        Initialize CLIP embedding model.
        
        :param model_name: Name of the CLIP model to load
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        try:
            self.logger.info(f"Starting to load CLIP model: {model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.debug(f"Using device: {self.device}")

            if os.path.exists(MODEL_DIR):
                self.logger.info(f"Loading CLIP model from local directory: {MODEL_DIR}")
                self.model, self.preprocess = clip.load(model_name, device=self.device)
                self.model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location=self.device))
            else:
                self.logger.info(f"Downloading and saving CLIP model to: {MODEL_DIR}")
                self.model, self.preprocess = clip.load(model_name, device=self.device)
                os.makedirs(MODEL_DIR, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(MODEL_DIR, "pytorch_model.bin"))

            self.embedding_dim = self.model.visual.output_dim

        except Exception as e:
            self.logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of text documents.
        
        :param texts: List of text strings
        :return: NumPy array of text embeddings
        """
        self.logger.info(f"Generating embeddings for {len(texts)} text documents.")
        
        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        embeddings = text_features.cpu().numpy()
        self.logger.debug(f"Generated {embeddings.shape[0]} text embeddings.")
        return embeddings
    
    def generate_image_embeddings(self, image_inputs: Union[List[str], List[Image.Image]]) -> np.ndarray:
        """
        Generate embeddings for a list of images.
        
        :param image_inputs: List of image file paths or PIL Image objects
        :return: NumPy array of image embeddings
        """
        self.logger.debug(f"Generating embeddings for {len(image_inputs)} images.")
        
        # Preprocess images
        processed_images = []
        for img_input in image_inputs:
            # Handle both file paths and PIL Image objects
            if isinstance(img_input, str):
                img = Image.open(img_input)
            elif isinstance(img_input, Image.Image):
                img = img_input
            else:
                raise ValueError("Image input must be a file path or PIL Image")
            
            processed_images.append(self.preprocess(img).to(self.device))
        
        image_input = torch.stack(processed_images)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        embeddings = image_features.cpu().numpy()
        self.logger.info(f"Generated {embeddings.shape[0]} image embeddings.")
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a single query.
        
        :param query: Query string
        :return: NumPy array of query embedding
        """
        self.logger.info(f"Generating embedding for query: {query}")
        
        with torch.no_grad():
            tokens = clip.tokenize([query]).to(self.device)
            query_features = self.model.encode_text(tokens)
            query_features /= query_features.norm(dim=-1, keepdim=True)
        
        embedding = query_features.cpu().numpy()
        self.logger.info("Query embedding generated.")
        return embedding
    
    @property
    def embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings.
        
        :return: Embedding dimension
        """
        return self.embedding_dim


def test_model():
    model = CLIPEmbeddingModel()
    print("Model initialized.")

    texts = ["The quick brown fox jumps over the lazy dog.", "An apple a day keeps the doctor away."]
    start_time = time.time()
    text_embeddings = model.generate_text_embeddings(texts)
    end_time = time.time()
    print(text_embeddings)
    text_embeddings = np.array(text_embeddings)
    print(f"Text embeddings shape: {text_embeddings.shape}, Time taken: {end_time - start_time:.2f} seconds")

    image_paths = ["../../data/figures/APOE4-Figure1-1.png", "../../data/figures/APOE4-Figure4-1.png"]
    start_time = time.time()
    image_embeddings = model.generate_image_embeddings(image_paths)
    end_time = time.time()
    #print(image_embeddings)
    #print(f"Image embeddings shape: {image_embeddings.shape}, Time taken: {end_time - start_time:.2f} seconds")

    query = "A cat sitting on a table"
    start_time = time.time()
    query_embedding = model.embed_query(query)
    end_time = time.time()
    #print(query_embedding)
    #print(f"Query embedding shape: {query_embedding.shape}, Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    test_model()