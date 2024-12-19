from langchain_core.documents import Document as BaseDocument
from typing import Literal, Dict, Any, List

class Document(BaseDocument):
    """Class for storing a piece of text and associated metadata."""

    page_content: str  # Use page_content to align with the base class
    type: Literal["Document"] = "Document"

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None) -> None:
        """Initialize a Document instance.
        
        Args:
            page_content (str): The content of the document.
            metadata (dict): A dictionary containing metadata about the document.
        """
        if metadata is None:
            metadata = {}
        super().__init__(page_content=page_content, metadata=metadata)  

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "document"]

    def format_metadata(self) -> Dict[str, str]:
        """Format the metadata of the document.

        Returns:
            dict: A dictionary containing formatted metadata with predefined attributes.
        """
        attributes = ['source', 'page_idx', 'block_idx', 'tag', 'type']
        formatted_metadata = {key: self.metadata.get(key, 'unknown') for key in attributes}
        return formatted_metadata
    
    @property
    def source(self) -> str:
        """Get the source of the document."""
        return self.metadata.get('source', 'unknown')

    @property
    def page_idx(self) -> str:
        """Get the page index of the document."""
        return self.metadata.get('page_idx', 'unknown')

    @property
    def block_idx(self) -> str:
        """Get the block index of the document."""
        return self.metadata.get('block_idx', 'unknown')

    @property
    def tag(self) -> str:
        """Get the tag of the document."""
        return self.metadata.get('tag', 'unknown')

    @property
    def type(self) -> str:
        """Get the type of the document."""
        return self.metadata.get('type', 'unknown')

    def __repr__(self) -> str:
        """Return a string representation of the Document instance."""
        metadata_str = ', '.join(f"{key}={value}" for key, value in self.format_metadata().items())
        return f"Document({metadata_str}, page_content={self.page_content})"  # Update to use page_content

    def __str__(self) -> str:
        """Override __str__ to restrict it to page_content and metadata."""
        if self.metadata:
            return f"page_content='{self.page_content}' metadata={self.metadata}"  # Update to use page_content
        else:
            return f"page_content='{self.page_content}'"  