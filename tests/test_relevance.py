
from src.relevance import (
    detect_requested_tenant,
    filter_relevant_chunks,
)


def test_detect_tenant_name():
    assert detect_requested_tenant(
        "What are Tenant B's security requirements?"
    ) == "tenant_b"


def test_other_tenant_question_returns_no_chunks():
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }

    chunks = [{
        "tenant_id": "tenant_a",
        "access_level": "internal",
        "similarity": 0.80,
        "text": "Tenant A policy",
    }]

    result = filter_relevant_chunks(
        "What are Tenant B's security requirements?",
        chunks,
        user,
    )

    assert result == []


def test_low_similarity_chunk_is_rejected():
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }

    chunks = [{
        "tenant_id": "tenant_a",
        "access_level": "internal",
        "similarity": 0.20,
        "text": "Unrelated content",
    }]

    result = filter_relevant_chunks(
        "What are Tenant A's security policies?",
        chunks,
        user,
    )

    assert result == []


def test_relevant_authorized_chunk_is_kept():
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }

    chunks = [{
        "tenant_id": "tenant_a",
        "access_level": "internal",
        "similarity": 0.80,
        "text": "Tenant A policy",
    }]

    result = filter_relevant_chunks(
        "What are Tenant A's security policies?",
        chunks,
        user,
    )

    assert len(result) == 1
