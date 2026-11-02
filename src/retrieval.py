
from pathlib import Path
import json
import os
import time

import chromadb
from sentence_transformers import SentenceTransformer

from src.relevance import (
    filter_relevant_chunks,
    is_cross_tenant_request,
    detect_requested_tenant,
    format_tenant_title,
)

from src.generation import (
    generate_answer,
    get_model_name,
    typewriter_print,
    OllamaUnavailableError,
    OllamaTimeoutError,
    OllamaRequestError,
    OllamaResponseError,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
USER_REGISTRY_PATH = PROJECT_ROOT / "users.json"

COLLECTION_NAME = "microsoft_public_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# --------------------------------------------------
# DATABASE (CACHED)
# --------------------------------------------------

_CACHED_CLIENT = None
_CACHED_COLLECTION = None
_CACHED_MODEL = None


def load_database(force_reload=False):
    """Load the persistent ChromaDB collection with memory caching."""
    global _CACHED_CLIENT, _CACHED_COLLECTION

    if not force_reload and _CACHED_COLLECTION is not None:
        return _CACHED_CLIENT, _CACHED_COLLECTION

    if not DB_DIR.exists():
        raise FileNotFoundError(
            "ChromaDB not found. Run: python -m src.chroma_setup"
        )

    client = chromadb.PersistentClient(path=str(DB_DIR))

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as exc:
        raise RuntimeError(
            "The configured ChromaDB collection could not be opened."
        ) from exc

    if collection.count() == 0:
        raise ValueError(
            "The ChromaDB collection is empty. Add documents first."
        )

    _CACHED_CLIENT = client
    _CACHED_COLLECTION = collection
    return client, collection


# --------------------------------------------------
# EMBEDDING MODEL (CACHED)
# --------------------------------------------------

def load_embedding_model(force_reload=False):
    """Load the sentence-transformer embedding model with memory caching."""
    global _CACHED_MODEL

    if not force_reload and _CACHED_MODEL is not None:
        return _CACHED_MODEL

    print(f"Loading embedding model: {MODEL_NAME}")

    try:
        _CACHED_MODEL = SentenceTransformer(MODEL_NAME)
        return _CACHED_MODEL
    except Exception as exc:
        raise RuntimeError(
            "The embedding model could not be loaded. "
            "Check the model installation and local cache."
        ) from exc



# --------------------------------------------------
# AUTOMATIC DEVELOPMENT IDENTITY
# --------------------------------------------------

def get_user_context():
    """
    Resolve the development identity from users.json.

    RAG_USER_ID selects a registered development identity.
    This environment variable is not production authentication.
    """

    user_id = os.environ.get("RAG_USER_ID", "").strip()

    if not user_id:
        raise ValueError(
            "RAG_USER_ID is not configured. "
            "Set it to a user ID defined in users.json."
        )

    if not USER_REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"User registry not found: {USER_REGISTRY_PATH}"
        )

    try:
        with USER_REGISTRY_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            users = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            "users.json contains invalid JSON."
        ) from exc

    if not isinstance(users, dict):
        raise ValueError(
            "users.json must contain a JSON object of users."
        )

    user = users.get(user_id)

    if not isinstance(user, dict):
        raise ValueError(
            "The configured user ID is not registered."
        )

    required_fields = {
        "name",
        "tenant_id",
        "access_level",
    }

    if not required_fields.issubset(user):
        raise ValueError(
            "The configured user entry is missing required fields."
        )

    # Allow only the identities configured for this prototype.
    allowed_identities = {
        ("microsoft_public", "public"),
        ("tenant_a", "internal"),
        ("tenant_b", "internal"),
    }

    identity = (
        user["tenant_id"],
        user["access_level"],
    )

    if identity not in allowed_identities:
        raise ValueError(
            "The configured identity has an unsupported permission level."
        )

    return {
        "user_id": user_id,
        "name": user["name"],
        "tenant_id": user["tenant_id"],
        "access_level": user["access_level"],
    }


# --------------------------------------------------
# PERMISSION-AWARE RETRIEVAL
# --------------------------------------------------

def search(question, model, collection, user, top_k=TOP_K):
    """
    Retrieve only public documents and documents authorized
    for the current user's tenant.
    """

    query_vector = model.encode(
        question,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    # Primary authorization filter at the database query level.
    permission_filter = {
        "$or": [
            {
                "$and": [
                    {"tenant_id": "microsoft_public"},
                    {"access_level": "public"},
                ]
            },
            {
                "$and": [
                    {"tenant_id": user["tenant_id"]},
                    {
                        "access_level": {
                            "$in": ["internal", "confidential"]
                        }
                    },
                ]
            },
        ]
    }

    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, collection.count()),
        where=permission_filter,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    retrieved_chunks = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index]

        tenant_id = metadata.get("tenant_id")
        access_level = metadata.get("access_level")

        is_public = (
            tenant_id == "microsoft_public"
            and access_level == "public"
        )

        belongs_to_user = (
            tenant_id == user["tenant_id"]
            and access_level in {"internal", "confidential"}
        )

        # Defense in depth: never pass unauthorized evidence onward.
        if not (is_public or belongs_to_user):
            raise PermissionError(
                "An authorization check failed during retrieval."
            )

        distance = distances[index]

        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "source_file": metadata.get(
                "source_file",
                "Unknown source",
            ),
            "tenant_id": tenant_id,
            "access_level": access_level,
            "text": documents[index],
            "similarity": 1.0 - distance,
        })

    return retrieved_chunks


# --------------------------------------------------
# FRIENDLY ERROR MESSAGES
# --------------------------------------------------

def print_error(message, next_step=None):
    """Print a user-friendly error message."""

    print(f"\n{message}")

    if next_step:
        print(f"Next step: {next_step}")


# --------------------------------------------------
# MAIN RAG PIPELINE
# --------------------------------------------------

def main():
    # 1. Initialize the document database.
    try:
        client, collection = load_database()

    except FileNotFoundError as exc:
        print_error(
            "The document database was not found.",
            str(exc),
        )
        return

    except ValueError as exc:
        print_error(
            "The document database is empty.",
            str(exc),
        )
        return

    except Exception:
        print_error(
            "The document database could not be initialized.",
            "Check the database configuration and try again.",
        )
        return

    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")

    # 2. Load the embedding model.
    try:
        model = load_embedding_model()

    except Exception:
        print_error(
            "The embedding model could not be initialized.",
            "Check your Python environment and model availability.",
        )
        return

    # 3. Resolve the configured development identity.
    try:
        user = get_user_context()

    except (ValueError, FileNotFoundError) as exc:
        print_error(
            "The development identity could not be resolved.",
            str(exc),
        )
        return

    except OSError:
        print_error(
            "The user registry could not be read.",
            "Check file permissions and try again.",
        )
        return

    print(f"\nTest identity: {user['name']}")
    print("Permission-aware RAG is ready.")
    print("Type 'exit' to quit.")

    # 4. Process questions.
    while True:
        try:
            question = input("\nEnter your question: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nRAG stopped.")
            break

        if question.lower() in {"exit", "quit"}:
            print("RAG stopped.")
            break

        if not question:
            print_error(
                "Your question is empty.",
                "Enter a question about the available documents.",
            )
            continue

        try:
            # 5. Check for an explicit cross-tenant request BEFORE
            # retrieval or generation.
            if is_cross_tenant_request(question, user):
                target_tenant = detect_requested_tenant(question)
                target_title = format_tenant_title(target_tenant)
                user_title = format_tenant_title(user.get("tenant_id"))
                print("\n" + "=" * 60)
                print("YOU DON'T HAVE ACCESS TO THIS INFORMATION")
                print("=" * 60)
                print(
                    f"You are signed in as a {user_title} user. "
                    f"Your current permissions do not allow access to {target_title}'s private documents. "
                    f"Try asking about public documents or information authorized for {user_title}."
                )
                continue

            # 6. Retrieve authorized chunks from ChromaDB.
            t_retrieval_start = time.perf_counter()
            retrieved_chunks = search(
                question=question,
                model=model,
                collection=collection,
                user=user,
            )

            # 7. Filter for relevant, authorized evidence.
            relevant_chunks = filter_relevant_chunks(
                question=question,
                chunks=retrieved_chunks,
                user=user,
            )
            retrieval_duration_ms = (time.perf_counter() - t_retrieval_start) * 1000

            print("\n" + "=" * 60)
            print("RAG RESULTS")
            print("=" * 60)
            print(f"Retrieved chunks: {len(retrieved_chunks)}")
            print(
                "Relevant authorized chunks: "
                f"{len(relevant_chunks)}"
            )

            # 8. Abstain when no relevant evidence is available.
            if not relevant_chunks:
                print("\n" + "=" * 60)
                print("I COULDN'T FIND AN ANSWER")
                print("=" * 60)
                print(
                    "I searched the documents you’re allowed to access, but "
                    "I couldn't find enough information to answer this question confidently.\n\n"
                    "What you can try:\n"
                    "- Rephrase your question using different keywords.\n"
                    "- Ask about a more specific topic.\n"
                    "- Try a question related to your organization's available documents.\n\n"
                    "Your documents and access permissions have not been changed."
                )
                continue

            # 9. Generate the answer using the configured Ollama model.
            t_gen_start = time.perf_counter()
            answer = generate_answer(
                question=question,
                chunks=relevant_chunks,
            )
            generation_duration_ms = (time.perf_counter() - t_gen_start) * 1000
            total_duration_ms = retrieval_duration_ms + generation_duration_ms

            current_model_name = get_model_name()

            print(f"\n{current_model_name.upper()} ANSWER")
            print("-" * 60)

            typewriter_print(answer, delay=0.015)

            print("-" * 60)

            print(
                f"[Latency] Retrieval: {retrieval_duration_ms:.1f}ms | "
                f"Generation: {generation_duration_ms:.1f}ms | "
                f"Total: {total_duration_ms:.1f}ms"
            )
        except PermissionError:
            print_error(
                "This request could not be completed because an "
                "authorization check failed.",
                "Ask about public information or documents your "
                "account is authorized to access.",
            )

        except OllamaUnavailableError:
            print_error(
                "The local AI service could not be reached.",
                "Start Ollama and check that it is listening at "
                "http://localhost:11434. Then try again.",
            )

        except OllamaTimeoutError:
            print_error(
                "The local AI service took too long to respond.",
                "Try again. If the problem continues, check system "
                "resources or reduce the workload.",
            )

        except OllamaRequestError as exc:
            print_error(
                "The local AI service rejected or could not process "
                "the request.",
                str(exc),
            )

        except OllamaResponseError:
            print_error(
                "The local AI service returned an invalid response.",
                "Try again and check the Ollama service if the "
                "problem continues.",
            )

        except Exception as exc:
            print_error(
                "An unexpected error occurred while processing "
                "your question.",
                "Try again. If the problem continues, inspect "
                "the development logs.",
            )
            print(f"[DEBUG] Error type: {type(exc).__name__}")


if __name__ == "__main__":
    main()
