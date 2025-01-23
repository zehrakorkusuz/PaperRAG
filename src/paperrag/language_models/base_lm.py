from abc import ABC, abstractmethod

class BaseLanguageModel(ABC):
    def __init__(self, model_name, system_message=None):
        """
        Base class for language model interfaces.
        
        :param model_name: Name of the language model to use.
        :param system_message: Optional system-level instructions for the model.
        """
        self.model_name = model_name
        self.system_message = system_message

    @abstractmethod
    def send_request(self, messages, stream=True, **kwargs):
        """
        Send a request to the language model API.
        
        :param messages: List of message dictionaries.
        :param stream: Whether to stream responses.
        :param kwargs: Additional parameters for the API call.
        :return: Response data (streamed or single object).
        """
        pass

    def format_messages(self, user_message, context=None):
        """
        Format messages to include system instructions, user input, and optional context.
        
        :param user_message: User's query.
        :param context: Optional retrieved context to include.
        :return: List of formatted messages.
        """
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        if context:
            messages.append({"role": "assistant", "content": context})
        messages.append({"role": "user", "content": user_message})
        return messages
