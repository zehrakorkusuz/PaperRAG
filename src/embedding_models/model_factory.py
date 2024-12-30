from embedding_models.clip_model import CLIPEmbeddingModel
from embedding_models.base_embedding_model import BaseEmbeddingModel

def get_embedding_model(model_name: str, **kwargs) -> BaseEmbeddingModel:
    """
    Factory function to return the appropriate embedding model.

    :param model_name: Name of the embedding model (e.g., 'clip', 'bert').
    :param kwargs: Additional arguments to pass to the model's constructor.
    :return: An instance of a subclass of BaseEmbeddingModel.
    """
    model_registry = {
        "clip": CLIPEmbeddingModel,
    }

    if model_name not in model_registry:
        raise ValueError(f"Unknown model '{model_name}'. Available models: {list(model_registry.keys())}")

    return model_registry[model_name](**kwargs)
