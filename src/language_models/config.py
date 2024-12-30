from .ollama_lm import OllamaLanguageModel

def get_language_model(config):
    """
    Factory method to instantiate a language model based on the config.

    :param config: Dictionary containing model configuration.
    :return: An instance of a language model.
    """
    model_type = config.get("type", "ollama")
    model_name = config["model_name"]
    system_message = config.get("system_message", None)

    if model_type == "ollama":
        base_url = config["base_url"]
        return OllamaLanguageModel(base_url, model_name, system_message)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
