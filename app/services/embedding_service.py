from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def generate_embedding(text: str):
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding