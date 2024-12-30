from vector_db.base_db import BaseVectorDB
from vector_db.faiss_db import FAISSVectorDB
# from vector_db.pinecone_db import PineconeVectorDB

def get_vector_db(db_name: str, **kwargs) -> BaseVectorDB:
    db_registry = {
        "faiss": FAISSVectorDB
        #"pinecone": PineconeVectorDB,
    }

    if db_name not in db_registry:
        raise ValueError(f"Unknown database '{db_name}'. Available: {list(db_registry.keys())}")

    return db_registry[db_name](**kwargs)
