import requests
from .base_lm import BaseLanguageModel
import json
import logging

logger = logging.getLogger(__name__)

class OllamaLanguageModel(BaseLanguageModel):
    def __init__(self, base_url, model_name, system_message=None):
        """
        Ollama-specific language model implementation.
        
        :param base_url: Base URL for the Ollama API.
        :param model_name: Name of the model to use.
        :param system_message: Optional system-level instructions for the model.
        """
        super().__init__(model_name, system_message)
        self.base_url = base_url

    @staticmethod 
    def format_messages(user_message, context=None, schema=None):
        """
        Format messages to include user input, context, and optional schema instructions.
        
        :param user_message: The user's query.
        :param context: Optional retrieved context.
        :param schema: Optional JSON schema for structured output.
        :return: List of formatted messages.
        """
        messages = []
        if schema:
            schema_instruction = (
                f"Please return the result as a JSON object using the following schema:\n{json.dumps(schema)}"
            )
            system_message = f"You are a helpful assistant.\n{schema_instruction}"
        else:
            system_message = "You are a helpful assistant."

        if context:
            messages.append({"role": "assistant", "content": context})
        messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": user_message})
        return messages


    def send_request(self, messages, stream=True, **kwargs):
        """
        Send a chat request to the Ollama API.

        :param messages: List of message dictionaries.
        :param stream: Whether to stream the response.
        :param kwargs: Additional parameters for the API call.
        :return: Generator or dictionary with the response.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            **kwargs
        }

        try:
            with requests.post(f"{self.base_url}/api/chat", json=payload, stream=stream) as response:
                response.raise_for_status()
                if stream:
                    for line in response.iter_lines():
                        if line:
                            yield line.decode('utf-8')
                else:
                    return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            return None
