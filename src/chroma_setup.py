
import json
from pathlib import Path

import chromadb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VECTOR_FILE = PROJECT_ROOT / "data" / "embeddings" / "vectors.npy"
METADATA_FILE = PROJECT_ROOT / "data" / "embeddings" / "metadata.json"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "microsoft_public_docs"
BATCH_SIZE = 250


def main():
    # Check that the existing files are available
    if not VECTOR_FILE.exists() or not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Embedding files not found. Run the embedding script first."
        )

    vectors = np.load(VECTOR_FILE)

    with METADATA_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if len(vectors) != len(chunks):
        raise ValueError("Vector count and chunk count do not match.")

    print(f"Loaded {len(chunks)} chunks.")
    print(f"Vector shape: {vectors.shape}")

    # Create a persistent local database
    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Create the collection if it does not already exist
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert or update chunks in manageable batches
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        batch_vectors = vectors[start:start + BATCH_SIZE]

        collection.upsert(
            ids=[chunk["chunk_id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[
                {
                    "document_id": chunk["document_id"],
                    "source_file": chunk["source_file"],
                    "tenant_id": chunk["tenant_id"],
                    "access_level": chunk["access_level"],
                    "chunk_index": chunk["chunk_index"],
                }
                for chunk in batch
            ],
            embeddings=batch_vectors.tolist(),
        )

        print(
            f"Stored {min(start + len(batch), len(chunks))}"
            f"/{len(chunks)} chunks"
        )

    print("\nChromaDB setup completed.")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")
    print(f"Database directory: {DB_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
