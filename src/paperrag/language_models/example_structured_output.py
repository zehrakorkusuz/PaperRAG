from paperrag.language_models.config import get_language_model
from paperrag.language_models.utils import stream_responses, log_metrics

# Configuration for the Ollama language model API
config = {
    "type": "ollama",
    "base_url": "http://localhost:11434",  # Ensure this is the correct URL where Ollama is running
    "model_name": "llama3.2:latest"  # Ensure that this model exists and is accessible
}

# Define JSON Schema for structured output regarding brain region (Hippocampus in this case)
brain_region_schema = {
    "type": "object",
    "properties": {
        "region": {"type": "string", "description": "Name of the brain region"},
        "functions": {
            "type": "array",
            "description": "Primary cognitive functions",
            "items": {"type": "string"}
        },
        "coordinates": {
            "type": "object",
            "description": "MNI152 coordinates",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "z": {"type": "number"}
            }
        },
        "neurotransmitters": {
            "type": "array",
            "description": "Major neurotransmitter systems",
            "items": {"type": "string"}
        }
    },
    "required": ["region", "functions"]
}

# Instantiate the language model using the provided configuration
language_model = get_language_model(config)

# User input for the query about hippocampus functions and coordinates
user_query = "Describe the functions of the hippocampus and its coordinates."


# Context about the hippocampus (can be extended or dynamically fetched)
retrieved_context = "The hippocampus is critical for memory formation and spatial navigation."

# Format messages to include the user query, context, and schema instructions
messages = language_model.format_messages(user_query, context=retrieved_context, schema=brain_region_schema)

# Stream responses from the language model with structured output based on the JSON schema
stream_responses(language_model, messages, schema=brain_region_schema, log_metrics_fn=log_metrics)
