from paperrag.language_models.config import get_language_model
from paperrag.language_models.utils import stream_responses, log_metrics

class SchemaGenerator:
    def __init__(self, config):
        """
        Initialize the SchemaGenerator with the Ollama language model configuration.
        
        Parameters:
            config (dict): Configuration for the Ollama language model API.
        """
        self.config = config
        self.language_model = get_language_model(self.config)
    
    def create_dynamic_prompt(self, user_query, context):
        """
        Create a comprehensive prompt that instructs the language model to generate a JSON schema
        based on user query and context.
        
        Parameters:
            user_query (str): The user's query or intent.
            context (str): Contextual information related to the user's query.
        
        Returns:
            str: The formatted prompt to send to the language model.
        """
        return f"""
        Given the following user query and context, generate a JSON schema that structures the response data:
        
        User Query: {user_query}
        Context: {context}
        
        Please create a JSON schema with the following specifications:
        1. The schema should include relevant fields based on the user's query and context.
        2. Each field should have a description and a type.
        3. The schema should be as detailed as possible, considering the user's intent and the information provided in the context.
        4. The schema may include nested objects, arrays, or any necessary data structures to represent the data in an organized and logical manner.
        
        The output should be in valid JSON format Be mindful of brevity only capturing essential and important details and insights.
        """
    
    def generate_schema(self, user_query, context):
        """
        Generate a JSON schema by creating a dynamic prompt based on the user's query and context,
        and then streaming the response from the language model.
        
        Parameters:
            user_query (str): The user's query or intent.
            context (str): Contextual information related to the user's query.
        
        Returns:
            dict: The generated schema in JSON format.
        """
        dynamic_prompt = self.create_dynamic_prompt(user_query, context)
        
        # Format the messages for the language model
        messages = self.language_model.format_messages(dynamic_prompt)
        
        # Stream responses from the language model with structured output
        response = stream_responses(self.language_model, messages, log_metrics_fn=log_metrics, json_format=True)
        
        return response


### EXAMPLE ###

config = {
    "type": "ollama",
    "base_url": "http://localhost:11434", 
    "model_name": "llama3.2:latest" 
}

# Instantiate the SchemaGenerator class
schema_generator = SchemaGenerator(config)

# User input (query and context)
user_query = "Describe the functions of the hippocampus and its coordinates."
context = "The hippocampus is critical for memory formation, spatial navigation, and is located in the medial temporal lobe."

# Generate schema in one line
generated_schema = schema_generator.generate_schema(user_query, context)

# Now, `generated_schema` contains the structured JSON response.
print(generated_schema)


### EXAMPLE OUTPUT ###

# {
#     "$schema": "http://json-schema.org/draft-07/schema#",
#     "title": "Hippocampus Functions",
#     "type": "object",
#     "properties": {
#         "name": {
#             "description": "Name of the hippocampus",
#             "type": "string"
#         },
#         "location": {
#             "description": "Location of the hippocampus in the medial temporal lobe",
#             "type": "string"
#         },
#         "functions": {
#             "description": "Functions of the hippocampus including memory formation and spatial navigation",
#             "type": "array",
#             "items": {
#                 "$ref": "#/definitions/Function"
#             }
#         },
#         "coordinates": {
#             "description": "Coordinates of the hippocampus (e.g. x, y, z)",
#             "type": "object",
#             "properties": {
#                 "x": {"type": "number"},
#                 "y": {"type": "number"},
#                 "z": {"type": "number"}
#             },
#             "required": ["x", "y", "z"]
#         }
#     },
#     "definitions": {
#         "Function": {
#             "description": "Individual function of the hippocampus",
#             "type": "string"
#         }
#     }
# }
