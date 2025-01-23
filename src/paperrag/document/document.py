# document.py
from langchain_core.documents import Document as BaseDocument
from pydantic import Field
from typing import Dict, Any, List, Optional


class Document(BaseDocument):
    """Document class for storing content and metadata"""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    type: str = "Document" 

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, page_content: str = "", metadata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            page_content=page_content,  # BaseDocument expects page_content
            metadata=metadata or {},
            **kwargs
        )
        self.metadata = metadata or {}

    def format_metadata(self) -> Dict[str, str]:
        """Format metadata with predefined attributes"""
        attributes = ['source', 'page_idx', 'block_idx', 'tag', 'type']
        return {key: str(self.metadata.get(key, 'unknown')) for key in attributes}

    @property
    def source(self) -> str:
        return str(self.metadata.get('source', 'unknown'))

    @property
    def page_idx(self) -> str:
        return str(self.metadata.get('page_idx', 'unknown'))

    @property
    def block_idx(self) -> str:
        return str(self.metadata.get('block_idx', 'unknown'))

    @property
    def tag(self) -> str:
        return str(self.metadata.get('tag', 'unknown'))
    
    def to_dict(self):
        return self.dict()
