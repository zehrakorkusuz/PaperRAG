from paperrag.embedding_models.model_factory import get_embedding_model

def generate_embeddings(data):
    """Generate text and image embeddings from a data structure."""
    model = get_embedding_model("clip")
    texts = []
    image_paths = []
    for item in data:
        if item["type"] == "Document":
            texts.append(item["page_content"])
        elif item["type"] == "Figure":
            image_paths.append(item["page_content"])

    text_embeddings = model.generate_text_embeddings(texts)
    image_embeddings = model.generate_image_embeddings(image_paths)
    return text_embeddings, image_embeddings

# Example usage
data = [
    {"type": "Document", "page_content": "This is a document."},
    {"type": "Figure", "page_content": "/Users/zehrakorkusuz/PaperRAG/data/figures/APOE4-Figure2-1.png"},
]
text_embeddings, image_embeddings = generate_embeddings(data)
print(text_embeddings.shape)
print(image_embeddings.shape)