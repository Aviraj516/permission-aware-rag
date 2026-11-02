
from pathlib import Path
from pypdf import PdfReader
from docx import Document


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "public" / "microsoft"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def extract_text(file_path: Path) -> str:
    """
    Extract text from a PDF or DOCX document.
    """

    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(file_path)

        pages = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            if page_text.strip():
                pages.append(
                    f"\n[PAGE {page_number}]\n{page_text}"
                )

        return "\n".join(pages)

    elif file_path.suffix.lower() == ".docx":
        document = Document(file_path)

        paragraphs = [
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n".join(paragraphs)

    else:
        raise ValueError(
            f"Unsupported file format: {file_path.suffix}"
        )


def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find supported documents
    files = sorted(
        file_path
        for file_path in DATA_DIR.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in {".pdf", ".docx"}
    )

    if not files:
        print(f"No PDF or DOCX files found in: {DATA_DIR}")
        return

    print(f"Found {len(files)} documents.\n")

    successful = 0
    failed = 0

    for file_path in files:
        try:
            text = extract_text(file_path)

            if not text.strip():
                print(f"WARNING: No text extracted from {file_path.name}")
                failed += 1
                continue

            # Use the original filename with a .txt extension
            output_file = OUTPUT_DIR / f"{file_path.stem}.txt"

            output_file.write_text(text, encoding="utf-8")

            print("-" * 60)
            print(f"Document: {file_path.name}")
            print(f"Characters: {len(text):,}")
            print(f"Saved to: {output_file.relative_to(PROJECT_ROOT)}")
            print(f"Preview: {text[:200].replace(chr(10), ' ')}")
            print()

            successful += 1

        except Exception as error:
            print(f"ERROR processing {file_path.name}: {error}")
            failed += 1

    print("=" * 60)
    print("INGESTION SUMMARY")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
