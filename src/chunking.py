
import json
from pathlib import Path


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "chunks"

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def split_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks.

    CHUNK_SIZE: maximum number of characters per chunk.
    CHUNK_OVERLAP: characters shared between consecutive chunks.
    """

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than zero.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError(
            "CHUNK_OVERLAP must be non-negative "
            "and smaller than CHUNK_SIZE."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # Prefer ending at a word boundary
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            # Avoid producing tiny chunks
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        # Advance while retaining some overlapping text
        next_start = max(start + 1, end - CHUNK_OVERLAP)

        # Avoid starting in the middle of a word
        space_position = text.find(" ", next_start, end)

        if space_position != -1:
            next_start = space_position + 1

        start = next_start

    return chunks


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        print("Run the ingestion script first.")
        return

    text_files = sorted(INPUT_DIR.glob("*.txt"))

    if not text_files:
        print(f"No text files found in: {INPUT_DIR}")
        print("Run the ingestion script first.")
        return

    all_chunks = []
    total_documents = 0

    for file_path in text_files:
        text = file_path.read_text(encoding="utf-8")
        chunks = split_text(text)

        for index, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{file_path.stem}_{index:04d}",
                "document_id": file_path.stem,
                "source_file": file_path.name,
                "tenant_id": "microsoft_public",
                "access_level": "public",
                "chunk_index": index,
                "text": chunk_text,
            })

        total_documents += 1

        print(
            f"{file_path.name}: "
            f"{len(text):,} characters -> "
            f"{len(chunks)} chunks"
        )

    output_file = OUTPUT_DIR / "chunks.json"

    output_file.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 50)
    print("CHUNKING SUMMARY")
    print(f"Documents processed: {total_documents}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Saved to: {output_file.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
