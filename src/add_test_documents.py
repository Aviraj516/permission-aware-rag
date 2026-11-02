
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"

COLLECTION_NAME = "microsoft_public_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


TEST_DOCUMENTS = [
    {
        "chunk_id": "public_ai_policy_001",
        "document_id": "public_ai_policy",
        "source_file": "public_ai_policy.txt",
        "tenant_id": "microsoft_public",
        "access_level": "public",
        "text": (
            "Public AI Policy: AI systems should be evaluated for "
            "reliability, safety, fairness, privacy, and security. "
            "Organizations should document risks and monitor systems "
            "after deployment."
        ),
    },
    {
        "chunk_id": "tenant_a_security_001",
        "document_id": "tenant_a_security",
        "source_file": "tenant_a_security_policy.txt",
        "tenant_id": "tenant_a",
        "access_level": "internal",
        "text": (
            "Tenant A Internal Security Policy: Tenant A requires "
            "multi-factor authentication for employee accounts. "
            "Access to sensitive records must follow least privilege. "
            "Security events must be reported to the Tenant A security "
            "team within 30 minutes."
        ),
    },
    {
        "chunk_id": "tenant_b_security_001",
        "document_id": "tenant_b_security",
        "source_file": "tenant_b_security_policy.txt",
        "tenant_id": "tenant_b",
        "access_level": "internal",
        "text": (
            "Tenant B Internal Security Policy: Tenant B requires "
            "hardware security keys for administrator accounts. "
            "Production access must be approved by two authorized "
            "reviewers. Security incidents must be reported to the "
            "Tenant B incident response team within 15 minutes."
        ),
    },
]


def main():
    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_collection(name=COLLECTION_NAME)

    model = SentenceTransformer(MODEL_NAME)

    texts = [doc["text"] for doc in TEST_DOCUMENTS]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    collection.upsert(
        ids=[doc["chunk_id"] for doc in TEST_DOCUMENTS],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[
            {
                "document_id": doc["document_id"],
                "source_file": doc["source_file"],
                "tenant_id": doc["tenant_id"],
                "access_level": doc["access_level"],
                "chunk_index": 0,
            }
            for doc in TEST_DOCUMENTS
        ],
    )

    print("Synthetic test documents added successfully.")
    print(f"Collection now contains {collection.count()} chunks.")

    for doc in TEST_DOCUMENTS:
        print(f"- {doc['source_file']} ({doc['tenant_id']})")


if __name__ == "__main__":
    main()
