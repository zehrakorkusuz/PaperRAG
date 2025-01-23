from abc import ABC, abstractmethod
from typing import List, Tuple, Dict

class SparseRetriever(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, any]]) -> None:
        pass

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, any]]]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass