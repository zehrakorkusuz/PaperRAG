import string
from rank_bm25 import BM25Okapi
from paperrag.retrievers.sparse_retriever import SparseRetriever
from typing import List, Tuple, Dict, Optional


def preprocess(text):
    if isinstance(text, list):
        text = ' '.join(text)
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()


class BM25Retriever(SparseRetriever):
    def __init__(self, documents: List[str], min_doc_freq: int = 5):
        self.min_doc_freq = min_doc_freq
        self.documents = [preprocess(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.documents, k1=1.5, b=0.75)

    def add_documents(self, documents: List[Dict[str, any]]) -> None:
        tokenized_documents = [preprocess(doc["text"]) for doc in documents]
        self.documents.extend(tokenized_documents)
        self.bm25 = BM25Okapi(self.documents, k1=1.5, b=0.75)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, any]]]:
        if len(self.documents) < self.min_doc_freq:
            # TODO: Fallback: Jaccard Similarity for small document corpora
            # # Combined Score = α × BM25 + (1 − α) × Jaccard | α = 0.5 | 
            # For small corpora, using a lower α gives more weight to Jaccard, ensuring that even partial matches are prioritized. 
            # For larger corpora, increasing α prioritizes BM25, which excels in handling more complex term distributions.

            fallback_results = []
            query_tokens = set(preprocess(query))
            for doc in self.documents:
                doc_tokens = set(doc)
                intersection = query_tokens.intersection(doc_tokens)
                union = query_tokens.union(doc_tokens)
                score = len(intersection) / len(union) if union else 0.0
                fallback_results.append((score, {"text": " ".join(doc), "score": score}))
            sorted_fallback = sorted(fallback_results, key=lambda x: x[0], reverse=True)
            return sorted_fallback[:top_k]
        
        # BM25 Retrieval
        query_tokens = preprocess(query)
        doc_scores = self.bm25.get_scores(query_tokens)
        results = [(score, {"text": " ".join(self.documents[i]), "score": score}) 
                for i, score in enumerate(doc_scores)]
        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
        return sorted_results[:top_k]

    def reset(self) -> None:
        self.documents = []
        self.bm25 = BM25Okapi(self.documents, k1=1.5, b=0.75)



# if __name__ == "__main__": 
## Test full match, partial match and no match in different size of corpora
#     import time
#     small_corpus = [
#         "The quick brown fox jumps over the lazy dog.",
#         "A fast brown fox leaps over a sleepy dog.",
#         "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
#     ]

#     very_small_corpus = [
#         "Only one document in this corpus."
#     ]

#     # Sample large corpus (replicating small corpus)
#     large_corpus = small_corpus * 3000  

#     queries = [
#         "quick fox",
#         "sleepy dog",
#         "Lorem ipsum",
#         "",
#         "nonexistentterm",
#         "brown fox"
#     ]

#     def test_retriever(retriever, corpus_size, min_doc_freq, queries):
#         print(f"\nTesting retriever with corpus size: {corpus_size} and min_doc_freq: {min_doc_freq}")
#         total_time = 0
#         for query in queries:
#             start_time = time.time()
#             results = retriever.retrieve(query)
#             end_time = time.time()
#             retrieval_time = end_time - start_time
#             total_time += retrieval_time
#             print(f"\nQuery: '{query}'")
#             print(f"Retrieval Time: {retrieval_time:.6f} seconds")
#             print(f"Number of Results: {len(results)}")
#             for score, doc in results:
#                 print(f"Score: {score:.4f}, Document: {doc['text']}")
#         average_time = total_time / len(queries)
#         print(f"\nAverage Retrieval Time: {average_time:.6f} seconds")

#     # Test with small corpus where len(documents) >= min_doc_freq
#     small_retriever = BM25Retriever(small_corpus, min_doc_freq=2)
#     test_retriever(small_retriever, len(small_corpus), 2, queries)

#     #Test with small corpus where len(documents) < min_doc_freq
#     small_retriever = BM25Retriever(small_corpus, min_doc_freq=5)
#     test_retriever(small_retriever, len(small_corpus), 5, queries)

#     # Test with very small corpus where len(documents) < min_doc_freq
#     very_small_retriever = BM25Retriever(very_small_corpus, min_doc_freq=2)
#     test_retriever(very_small_retriever, len(very_small_corpus), 2, queries)

#     # Test with large corpus where len(documents) >= min_doc_freq
#     large_retriever = BM25Retriever(large_corpus, min_doc_freq=2)
#     test_retriever(large_retriever, len(large_corpus), 2, queries)