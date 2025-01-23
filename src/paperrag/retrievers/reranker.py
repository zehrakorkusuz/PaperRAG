from rerankers import Reranker

class Rerank:
    def __init__(self, model_name, api_key=None):
        self.reranker = Reranker(model_name, api_key=api_key)

    def rerank(self, query, results):
        docs = [result[1]['text'] for result in results]
        reranked_results = self.reranker.rank(query=query, docs=docs)
        return reranked_results
    

"""
For more information on the rerankers library, visit the official GitHub repository:
https://github.com/AnswerDotAI/rerankers
"""