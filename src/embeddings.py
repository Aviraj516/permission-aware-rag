
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_ROOT / "data" / "chunks" / "chunks.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    if not INPUT_FILE.exists():
        print(f"Chunk file not found: {INPUT_FILE}")
        print("Run the chunking script first.")
        return

    # Load the document chunks
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if not chunks:
        print("No chunks found. Nothing to embed.")
        return

    print(f"Loaded {len(chunks)} chunks.")

    # Load the embedding model
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    # Extract the text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Convert text into numerical vectors
    print("Generating embeddings...")

    vectors = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Save the vectors
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    vector_file = OUTPUT_DIR / "vectors.npy"
    metadata_file = OUTPUT_DIR / "metadata.json"

    np.save(vector_file, vectors)

    metadata_file.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("EMBEDDING SUMMARY")
    print(f"Chunks embedded: {len(texts)}")
    print(f"Vector matrix shape: {vectors.shape}")
    print(f"Vector dimensions: {vectors.shape[1]}")
    print(f"Vectors saved to: {vector_file.relative_to(PROJECT_ROOT)}")
    print(f"Metadata saved to: {metadata_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
