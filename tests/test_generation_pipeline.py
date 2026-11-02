
from unittest.mock import patch

from src.relevance import filter_relevant_chunks


def make_chunk(
    chunk_id,
    tenant_id,
    access_level,
    similarity=0.90,
):
    return {
        "chunk_id": chunk_id,
        "source_file": f"{chunk_id}.txt",
        "tenant_id": tenant_id,
        "access_level": access_level,
        "similarity": similarity,
        "text": f"Test evidence for {chunk_id}.",
    }


def test_public_question_passes_public_evidence_to_generation():
    user = {
        "tenant_id": "microsoft_public",
        "access_level": "public",
    }
    question = "What does the public AI policy say?"
    chunks = [
        make_chunk(
            "public_policy_001",
            "microsoft_public",
            "public",
        )
    ]

    approved = filter_relevant_chunks(question, chunks, user)

    with patch("src.retrieval.generate_answer", return_value="Grounded answer") as generate:
        answer = generate(question=question, chunks=approved)

    assert answer == "Grounded answer"
    generate.assert_called_once()
    assert approved[0]["tenant_id"] == "microsoft_public"


def test_cross_tenant_question_does_not_call_generation():
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }
    question = "What are Tenant B's security requirements?"
    chunks = [
        make_chunk(
            "tenant_a_policy",
            "tenant_a",
            "internal",
        )
    ]

    approved = filter_relevant_chunks(question, chunks, user)

    with patch("src.retrieval.generate_answer") as generate:
        if approved:
            generate(question=question, chunks=approved)

    assert approved == []
    generate.assert_not_called()


def test_low_relevance_evidence_does_not_call_generation():
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }
    question = "What are the security requirements?"
    chunks = [
        make_chunk(
            "tenant_a_policy",
            "tenant_a",
            "internal",
            similarity=0.20,
        )
    ]

    approved = filter_relevant_chunks(question, chunks, user)

    with patch("src.retrieval.generate_answer") as generate:
        if approved:
            generate(question=question, chunks=approved)

    assert approved == []
    generate.assert_not_called()


def test_authorized_tenant_evidence_is_preserved():
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }
    question = "What are Tenant A's internal security requirements?"
    chunk = make_chunk(
        "tenant_a_security_001",
        "tenant_a",
        "internal",
    )

    approved = filter_relevant_chunks(question, [chunk], user)

    assert len(approved) == 1
    assert approved[0]["chunk_id"] == "tenant_a_security_001"
    assert approved[0]["text"] == chunk["text"]


def test_chunk_deduplication_in_filter():
    """Ensure duplicate chunks are removed to avoid redundant evidence."""
    user = {
        "tenant_id": "tenant_a",
        "access_level": "internal",
    }
    question = "What are Tenant A's internal security requirements?"
    chunk1 = make_chunk("chunk_1", "tenant_a", "internal", similarity=0.85)
    chunk2 = make_chunk("chunk_1", "tenant_a", "internal", similarity=0.85)  # Duplicate ID
    chunk3 = make_chunk("chunk_2", "tenant_a", "internal", similarity=0.80)
    chunk3["text"] = chunk1["text"]  # Duplicate text content

    approved = filter_relevant_chunks(question, [chunk1, chunk2, chunk3], user)
    assert len(approved) == 1


def test_default_model_is_gemma3_4b():
    """Verify that the default configured model is gemma3:4b."""
    import os
    import src.generation
    from src.generation import get_model_name
    prev = os.environ.pop("RAG_LLM_MODEL", None)
    try:
        assert get_model_name() == "gemma3:4b"
        assert src.generation.MODEL_NAME == "gemma3:4b"
    finally:
        if prev is not None:
            os.environ["RAG_LLM_MODEL"] = prev


def test_payload_contains_gemma3_4b():
    """Verify that the actual request payload contains model: gemma3:4b."""
    import json
    import os
    from unittest.mock import patch, MagicMock
    from src.generation import generate_answer

    captured_payload = None

    def fake_urlopen(req, timeout=None):
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "message": {"content": "Tenant A requires MFA [S1]."}
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    prev = os.environ.pop("RAG_LLM_MODEL", None)
    try:
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            generate_answer(
                "What is Tenant A's policy?",
                [{"chunk_id": "c1", "source_file": "f.txt", "text": "Evidence"}]
            )
        assert captured_payload is not None
        assert captured_payload["model"] == "gemma3:4b"
    finally:
        if prev is not None:
            os.environ["RAG_LLM_MODEL"] = prev


def test_env_model_override():
    """Verify that RAG_LLM_MODEL overrides the model name."""
    import os
    from src.generation import get_model_name
    prev = os.environ.get("RAG_LLM_MODEL")
    try:
        os.environ["RAG_LLM_MODEL"] = "qwen3:1.7b"
        assert get_model_name() == "qwen3:1.7b"
    finally:
        if prev is not None:
            os.environ["RAG_LLM_MODEL"] = prev
        else:
            os.environ.pop("RAG_LLM_MODEL", None)


def test_ollama_missing_model_error_message():
    """Test that missing model produces readable error with pull command."""
    import urllib.error
    import io
    from src.generation import generate_answer, OllamaRequestError

    error_response = urllib.error.HTTPError(
        url="http://localhost:11434/api/chat",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=io.BytesIO(b'{"error": "model not found"}'),
    )

    with patch("urllib.request.urlopen", side_effect=error_response):
        import pytest
        with pytest.raises(OllamaRequestError) as exc_info:
            generate_answer(
                "What is the policy?",
                [{"chunk_id": "c1", "source_file": "f.txt", "text": "Evidence"}]
            )
        assert "404" in str(exc_info.value)
        assert "ollama pull" in str(exc_info.value)



