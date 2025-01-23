import os
import asyncio
from paperrag.embedding_models.model_factory import get_embedding_model
from paperrag.document.document_loader import DocumentLoader
from paperrag.vector_db.db_factory import get_vector_db
from paperrag.retrievers.retriever_factory import get_retriever
from paperrag.language_models.config import get_language_model
from paperrag.language_models.utils import stream_responses, log_metrics
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import random
import logging
# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# need to adjust the schema
general_schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "reasoning": {"type": "string"},
        "insights": {"type": "string"}
},
    "required": ["content", "reasoning"]
}

OUTPUT_DIR = "/tmp/paperrag"
MAX_CONTEXT_LENGTH = 77
LANGUEGE_MODEL_CONFIG = {
    "type": "ollama",
    "base_url": "http://localhost:11434",
    "model_name": "llama3.2:latest"
    #"stream": True
    # schema: {}
}
# PDFFIGURES SERVICE URL
# LLMSHERPA SERVICE URL
# EMBEDDING MODEL
# VECTOR DB
# RETRIEVER
# LANGUAGE MODEL // Move config to file

class Pipeline:
    def __init__(self, embedding_model: str, vector_db: str, retriever: str, language_model: str, documents: list):
        self.embedding_model = get_embedding_model(embedding_model)
        self.dimension = self.embedding_model.get_embedding_dim
        self.vector_db = get_vector_db(vector_db, dimension=self.dimension)
        self.documents = documents
        self.retriever = get_retriever(retriever, vector_db=self.vector_db, documents=self.documents, embedding_model=self.embedding_model)
        self.language_model = get_language_model(language_model)
        logger.info("Pipeline initialized with embedding model: %s, vector DB: %s, retriever: %s, language model: %s", embedding_model, vector_db, retriever, language_model)

    async def load_documents_async(self, file_path: str):
        logger.info("Loading documents asynchronously from %s", file_path)
        loader = DocumentLoader(pdf_path=file_path, output_dir=OUTPUT_DIR, async_mode=True)
        docs, stats = await loader.load_async()
        logger.info("Loaded %d documents asynchronously", len(docs))
        return docs, stats

    def load_documents_sync(self, file_path: str):
        logger.info("Loading documents synchronously from %s", file_path)
        loader = DocumentLoader(pdf_path=file_path, async_mode=False)
        docs, stats = loader.load_sync()
        logger.info("Loaded %d documents synchronously", len(docs))
        return docs, stats
    
    async def embed_and_store_async(self, documents): ## Save the documents as JSON to inspect // also consider expanding modality cases 
        logger.info("Starting embedding and storing process for %d documents", len(documents))
        
        # Initialize the text splitter for document text processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=77,  # Max chunk size for CLIP
            chunk_overlap=20,  # Appropriate overlap size
            length_function=len,
            is_separator_regex=False,
        )

        text_chunks = []
        text_metadata = []
        image_chunks = []
        image_metadata = []

        # save documents as json to inspect
        import json
        with open("documentsX.json", "w") as f:
            json.dump([doc.dict() for doc in documents], f, indent=4)
        
        # Iterate over documents and split them into chunks
        for doc in documents:
            logger.debug("Processing document with metadata: %s", doc.metadata)

            # Handle 'Document' type: text-based content
            if doc.type == 'Document' and hasattr(doc, 'page_content'):
                text_content = doc.page_content  # Text content of the document
                metadata = doc.metadata  # Metadata of the document
                
                # Split the text content into chunks
                chunks = text_splitter.create_documents([text_content])
                for chunk in chunks:
                    text_chunks.append(chunk.page_content)  # Add chunked text
                    text_metadata.append(metadata)  # Link metadata to text

            # Handle 'Figure' type: image content (URL)
            elif doc.type == 'Figure' and hasattr(doc, 'page_content'):
                print("Found a figure")
                image_url = doc.page_content  # Image URL in page_content
                metadata = doc.metadata  # Metadata of the image
                
                # Treat the page_content (image URL) as the content to embed
                image_chunks.append(image_url)  # Add the image URL to be embedded
                image_metadata.append(metadata)  # Link metadata to image

        logger.info("Generated %d text chunks and %d image chunks from documents", len(text_chunks), len(image_chunks))

        # Generate text embeddings
        text_embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedding_model.generate_text_embeddings(text_chunks)
        )
        logger.info("Generated embeddings for %d text chunks", len(text_embeddings))

        # Generate image embeddings
        image_embeddings = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedding_model.generate_image_embeddings(image_chunks)
        )
        logger.info("Generated embeddings for %d image chunks", len(image_embeddings))

        # Store text embeddings and metadata in the vector database
        for chunk, emb, metadata in zip(text_chunks, text_embeddings, text_metadata):
            logger.debug("Storing text chunk to vector DB with metadata: %s", metadata)
            self.vector_db.add_vectors(np.array([emb]), [metadata], modality="text")

        # Store image embeddings and metadata in the vector database
        for chunk, emb, metadata in zip(image_chunks, image_embeddings, image_metadata):
            logger.debug("Storing image chunk to vector DB with metadata: %s", metadata)
            self.vector_db.add_vectors(np.array([emb]), [metadata], modality="image")

        logger.info("Completed embedding and storing process")
        return len(text_embeddings) + len(image_embeddings)


    async def retrieve_documents_async(self, query: str): ## To add cache / reranking etc -- better inherent 
        logger.info("Generating embedding for query: %s", query)
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.embedding_model.generate_text_embeddings([query])
        )
        logger.info("Retrieving documents for query")
        retrieved_docs = self.retriever.retrieve(query_text=query, query_vector=query_embedding[0], top_k=3)
        logger.info("Retrieved %d documents for query", len(retrieved_docs))
        logger.debug("Retrieved documents: %s", retrieved_docs)
        return retrieved_docs

    async def generate_response_async(self, query: str, retrieved_docs):
        logger.info("Generating response using language model")
        # Format the retrieved documents as context / first element is the score: (0.7071067811865475, {'page_idx': 4, 'sect..
        context = "\n".join([doc[1].get('text', '') for doc in retrieved_docs])        
        # Format messages to include the user query, context, and schema instructions
        messages = self.language_model.format_messages(query, context=context, schema=general_schema)
        # Stream responses from the language model with structured output based on the JSON schema
        answer = await asyncio.get_event_loop().run_in_executor(
            None, lambda: stream_responses(self.language_model, messages, schema=general_schema, log_metrics_fn=log_metrics)
        )
        
        logger.info("Generated response for query")
        return answer

    async def run_pipeline_async(self, file_path: str, query: str):
        logger.info("Running pipeline asynchronously with file: %s and query: %s", file_path, query)
        docs, stats = await self.load_documents_async(file_path)
        await self.embed_and_store_async(docs)
        retrieved_docs = await self.retrieve_documents_async(query)
        answer = await self.generate_response_async(query, retrieved_docs)
        logger.info("Pipeline run completed")
        return answer, stats
    
    async def retrieve_and_respond_async(self, query: str):
        logger.info("Running pipeline asynchronously with query: %s", query)
        retrieved_docs = await self.retrieve_documents_async(query)
        answer = await self.generate_response_async(query, retrieved_docs)
        logger.info("Pipeline run completed for query only")
        return answer

if __name__ == "__main__":
    documents = []
    # random doc generation cause hallucination / remove the docs as parameter from pipeline class
    def generate_random_documents(num_docs):
        documents = []
        for i in range(num_docs):
            doc = {
                "text": f"This is a randomly generated document number {i}.",
                "metadata": {"source": "generated", "id": i}
            }
            documents.append(doc)
        return documents

    documents = generate_random_documents(10)
    pipeline = Pipeline("clip", "faiss", "hybrid", LANGUEGE_MODEL_CONFIG, documents)
    #asyncio.run(pipeline.run_pipeline_async("/Users/zehrakorkusuz/PaperRAG/data", "What is the main idea?"))
    # only query
    asyncio.run(pipeline.retrieve_and_respond_async("What is the document about?"))

    # Let's add documents from a json file
    path = "/Users/zehrakorkusuz/PaperRAG/documents.json"

## After fixing the load 

### ADD A NEW DOCUMENT /DOCUMENTS and Output - completed
### Connect and query QUERY THE DB 
### Instantiate db without documents 
