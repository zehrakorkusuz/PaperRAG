from abc import ABC, abstractmethod
from typing import List, Any, Tuple

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Abstract method for retrieving relevant documents based on a query.

        Args:
            query (str): The query string.
            top_k (int): The number of top results to return.

        Returns:
            List[Tuple[Any, float]]: List of the top-k results with their scores.
        """
        pass

    @abstractmethod
    def add_documents(self, documents: List[str]) -> None:
        """
        Abstract method to add documents to the retriever's database.

        Args:
            documents (List[str]): List of documents to add.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Abstract method to reset the retriever's database.
        """
        pass