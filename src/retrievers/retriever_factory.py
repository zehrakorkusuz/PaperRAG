from retrievers.base_retriever import BaseRetriever
# from retrievers.elasticsearch_retriever import ElasticsearchRetriever
from retrievers.bm25_retriever import BM25Retriever
from retrievers.dense_retriever import DenseRetriever
from retrievers.hybrid_retriever import HybridRetriever
from embedding_models.base_embedding_model import BaseEmbeddingModel
from vector_db.base_db import BaseVectorDB
from typing import List, Dict

def get_retriever(retriever_type: str, documents: List[Dict[str, any]] = None, embedding_model: BaseEmbeddingModel = None, vector_db: BaseVectorDB = None, index_name: str = "documents") -> BaseRetriever:
    if retriever_type == "elasticsearch":
        # if documents is None:
        #     raise ValueError("Documents must be provided for Elasticsearch retriever.")
        # retriever = ElasticsearchRetriever(index_name=index_name)
        # retriever.add_documents(documents)
        # return retriever
        raise NotImplementedError("Elasticsearch retriever is not implemented.")
    elif retriever_type == "bm25":
        if documents is None:
            raise ValueError("Documents must be provided for BM25 retriever.")
        tokenized_documents = [doc["text"].split() for doc in documents]
        return BM25Retriever(tokenized_documents)
    elif retriever_type == "dense":
        if embedding_model is None or vector_db is None:
            raise ValueError("Embedding model and vector database must be provided for dense retriever.")
        return DenseRetriever(embedding_model, vector_db)
    elif retriever_type == "hybrid":
        if documents is None or embedding_model is None or vector_db is None:
            raise ValueError("Documents, embedding model, and vector database must be provided for hybrid retriever.")
        sparse_retriever = ElasticsearchRetriever(index_name=index_name) if retriever_type == "elasticsearch" else BM25Retriever([doc["text"].split() for doc in documents])
        sparse_retriever.add_documents(documents)
        dense_retriever = DenseRetriever(embedding_model, vector_db)
        return HybridRetriever(sparse_retriever, dense_retriever)
    else:
        raise ValueError(f"Unknown retriever type '{retriever_type}'. Available types: 'elasticsearch', 'bm25', 'dense', 'hybrid'")