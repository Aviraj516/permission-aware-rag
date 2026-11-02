
import pytest

from src.retrieval import load_database, load_embedding_model, search


@pytest.fixture(scope="module")
def retrieval_resources():
    _, collection = load_database()
    model = load_embedding_model()
    return model, collection


@pytest.fixture
def tenant_a():
    return {
        "name": "Tenant A analyst",
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }


@pytest.fixture
def tenant_b():
    return {
        "name": "Tenant B analyst",
        "tenant_id": "tenant_b",
        "access_level": "internal",
    }


@pytest.fixture
def public_user():
    return {
        "name": "Public analyst",
        "tenant_id": "microsoft_public",
        "access_level": "public",
    }


def assert_results_are_authorized(results, user):
    """Ensure every returned chunk is permitted for this user."""
    for result in results:
        is_public = (
            result["tenant_id"] == "microsoft_public"
            and result["access_level"] == "public"
        )

        belongs_to_user = (
            result["tenant_id"] == user["tenant_id"]
            and result["access_level"] in {
                "internal",
                "confidential",
            }
        )

        assert is_public or belongs_to_user, (
            f"Unauthorized result returned: {result['chunk_id']}"
        )


def test_tenant_a_cannot_retrieve_tenant_b_private_chunks(
    retrieval_resources, tenant_a
):
    model, collection = retrieval_resources

    results = search(
        "Tenant B administrator security requirements",
        model,
        collection,
        tenant_a,
        top_k=5,
    )

    assert_results_are_authorized(results, tenant_a)

    forbidden = [
        result for result in results
        if result["tenant_id"] == "tenant_b"
        and result["access_level"] in {"internal", "confidential"}
    ]

    assert forbidden == [], "Tenant A received Tenant B private data"


def test_tenant_b_cannot_retrieve_tenant_a_private_chunks(
    retrieval_resources, tenant_b
):
    model, collection = retrieval_resources

    results = search(
        "Tenant A internal security policies",
        model,
        collection,
        tenant_b,
        top_k=5,
    )

    assert_results_are_authorized(results, tenant_b)

    forbidden = [
        result for result in results
        if result["tenant_id"] == "tenant_a"
        and result["access_level"] in {"internal", "confidential"}
    ]

    assert forbidden == [], "Tenant B received Tenant A private data"


def test_public_user_cannot_retrieve_private_chunks(
    retrieval_resources, public_user
):
    model, collection = retrieval_resources

    results = search(
        "Internal security policies and administrator requirements",
        model,
        collection,
        public_user,
        top_k=5,
    )

    assert_results_are_authorized(results, public_user)

    assert all(
        result["tenant_id"] == "microsoft_public"
        and result["access_level"] == "public"
        for result in results
    ), "Public user received a private document"
