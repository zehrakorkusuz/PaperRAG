from embedding_models.base_embedding_model import BaseEmbeddingModel
import clip
import numpy as np
from typing import List, Union
import torch
from PIL import Image

class CLIPEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.embedding_dim = self.model.visual.output_dim

    def generate_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generates embeddings for a list of text inputs."""
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_embeddings = self.model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            return text_embeddings.cpu().numpy()

    def generate_image_embeddings(self, image_inputs: List[Union[str, np.ndarray]]) -> np.ndarray:
        """Generates embeddings for a list of image file paths or numpy arrays."""
        images = []
        for image_input in image_inputs:
            if isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input).convert("RGB")
            else:
                image = Image.open(image_input).convert("RGB")
            images.append(image)

        processed_images = torch.stack([self.preprocess(image) for image in images]).to(self.device)

        with torch.no_grad():
            image_embeddings = self.model.encode_image(processed_images)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
            return image_embeddings.cpu().numpy()
        
    @property
    def get_embedding_dim(self) -> int:
        """Returns the dimensionality of the embeddings."""
        return self.embedding_dim
    
    @staticmethod
    def available_models(self) -> List[str]:
        """Returns a list of available CLIP models."""
        return clip.available_models()