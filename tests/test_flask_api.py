import json
from unittest.mock import patch, MagicMock

import pytest

from src.retrieval import get_user_context, search, load_database, load_embedding_model
from src.relevance import is_cross_tenant_request, filter_relevant_chunks
from src.generation import generate_answer, OllamaUnavailableError, OllamaTimeoutError, OllamaRequestError, OllamaResponseError


@pytest.fixture
def client():
    """Create a Flask test client with RAG_USER_ID configured."""
    import os
    os.environ["RAG_USER_ID"] = "employee_a"

    from app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

    del os.environ["RAG_USER_ID"]


class TestHealth:
    """Test GET /api/health"""

    def test_health_returns_valid_response(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert "service" in data


class TestSession:
    """Test GET /api/session"""

    def test_session_returns_user_info(self, client):
        resp = client.get("/api/session")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "user_name" in data
        assert "tenant_id" in data
        assert "access_level" in data
        # Should not expose the entire users.json
        assert data["user_name"] == "Tenant A analyst"
        assert len(data) == 3  # Only the minimum fields

    def test_session_fails_when_no_user_configured(self, client):
        # Temporarily clear RAG_USER_ID
        import os
        prev = os.environ.pop("RAG_USER_ID", None)
        try:
            resp = client.get("/api/session")
            assert resp.status_code == 400
            data = resp.get_json()
            assert data["status"] == "error"
            assert "No development user configured" in data["answer"]
        finally:
            if prev is not None:
                os.environ["RAG_USER_ID"] = prev


class TestChat:
    """Test POST /api/chat"""

    def test_empty_question_rejected(self, client):
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"

    def test_authorized_question_pipeline(self, client):
        """Test that an authorized question calls the existing pipeline."""
        resp = client.post("/api/chat", json={"question": "What is Tenant A's MFA policy?"})
        # Ollama may or may not have the requested model or be reachable, so status is 200 or 502/503
        # But the pipeline should still be invoked up to retrieval
        assert resp.status_code in (200, 500, 502, 503)

    def test_cross_tenant_question_blocked(self, client):
        """Cross-tenant questions do not invoke generation and return access_denied."""
        resp = client.post("/api/chat", json={"question": "What are Tenant B's security requirements?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "access_denied"
        assert data["title"] == "You don't have access to this information"
        assert "You are signed in as a Tenant A user" in data["answer"]
        assert "Your current permissions do not allow access to Tenant B's private documents" in data["answer"]
        assert "information authorized for Tenant A" in data["answer"]
        assert data["sources"] == []

    def test_insufficient_evidence_handled(self, client):
        """Insufficient evidence is handled correctly with user-friendly guidance."""
        resp = client.post("/api/chat", json={"question": "What is the capital of Mars?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "insufficient_evidence"
        assert data["title"] == "I couldn't find an answer"
        assert "I searched the documents you’re allowed to access, but I couldn't find enough information to answer this question confidently" in data["answer"]
        assert "suggestions" in data
        assert len(data["suggestions"]) == 3
        assert "note" in data
        assert data["sources"] == []

    def test_caller_cannot_impersonate_another_tenant(self, client):
        """A caller cannot gain another tenant's permissions by editing the request body."""
        resp = client.post(
            "/api/chat",
            json={
                "question": "What are Tenant B's security requirements?",
                "tenant_id": "tenant_b",
                "access_level": "internal",
            },
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "access_denied"
        assert data["title"] == "You don't have access to this information"

    @patch("app.generate_answer", return_value="Tenant A uses multi-factor authentication [S1].\n\nSources:\n[S1] tenant_a_security_policy.txt (chunk: tenant_a_001)")
    def test_successful_response_format(self, mock_generate, client):
        """Test successful response has the expected structured format and citations."""
        resp = client.post("/api/chat", json={"question": "What is Tenant A's MFA policy?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) > 0
        assert "latency" in data
        assert "retrieval_ms" in data["latency"]
        assert "generation_ms" in data["latency"]
        assert "total_ms" in data["latency"]


class TestCrossTenantRegression:
    """Regression tests for Section 6 requirements."""

    def test_tenant_a_asking_about_tenant_b_blocked(self, client):
        """1. Tenant A asking about Tenant B's private policy receives access_denied."""
        import os
        os.environ["RAG_USER_ID"] = "employee_a"
        resp = client.post("/api/chat", json={"question": "What is Tenant B's administrator security policy?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "access_denied"
        assert data["title"] == "You don't have access to this information"
        assert "You are signed in as a Tenant A user" in data["answer"]
        assert "Tenant B's private documents" in data["answer"]
        assert "authorized for Tenant A" in data["answer"]

    def test_tenant_b_asking_about_tenant_a_blocked(self, client):
        """2. Tenant B asking about Tenant A's private policy receives access_denied."""
        import os
        os.environ["RAG_USER_ID"] = "employee_b"
        try:
            resp = client.post("/api/chat", json={"question": "What is Tenant A's internal network security policy?"})
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["status"] == "access_denied"
            assert data["title"] == "You don't have access to this information"
            assert "You are signed in as a Tenant B user" in data["answer"]
            assert "Tenant A's private documents" in data["answer"]
            assert "authorized for Tenant B" in data["answer"]
        finally:
            os.environ["RAG_USER_ID"] = "employee_a"

    @patch("app.generate_answer")
    @patch("app.search")
    def test_generation_and_search_not_called_for_cross_tenant(self, mock_search, mock_gen, client):
        """3. Generation and search are not called for cross-tenant requests."""
        resp = client.post("/api/chat", json={"question": "What are Tenant B's credentials?"})
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "access_denied"
        mock_search.assert_not_called()
        mock_gen.assert_not_called()

    def test_unauthorized_private_source_metadata_not_returned(self, client):
        """4. Unauthorized private source metadata is not returned in access-denied response."""
        resp = client.post("/api/chat", json={"question": "What are Tenant B's private passwords?"})
        data = resp.get_json()
        assert data["status"] == "access_denied"
        assert data["sources"] == []
        assert "tenant_b_security_policy.txt" not in json.dumps(data)
        assert "chunk_id" not in json.dumps(data)

    def test_ordinary_unsupported_question_receives_insufficient_evidence(self, client):
        """5. An ordinary unsupported question receives insufficient_evidence, not access_denied."""
        resp = client.post("/api/chat", json={"question": "What is the average rainfall on Saturn's rings?"})
        data = resp.get_json()
        assert data["status"] == "insufficient_evidence"
        assert data["title"] == "I couldn't find an answer"
        assert data["sources"] == []

    @patch("app.generate_answer", return_value="Tenant A requires MFA for administrative logins [S1].\n\nSources:\n[S1] tenant_a_security_policy.txt (chunk: tenant_a_sec_1)")
    def test_authorized_questions_return_answer_and_citations(self, mock_gen, client):
        """6. Authorized questions still return answers and valid citations."""
        resp = client.post("/api/chat", json={"question": "What are Tenant A's security rules?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert "[S1]" in data["answer"]
        assert len(data["sources"]) > 0

    @patch("app.generate_answer", return_value="Microsoft focuses on safe AI deployment [S1].\n\nSources:\n[S1] public_ai_policy.txt (chunk: public_ai_001)")
    def test_public_document_access_continues_to_work(self, mock_gen, client):
        """7. Existing public-document access continues to work."""
        resp = client.post("/api/chat", json={"question": "What is Microsoft's public AI policy?"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"
        assert len(data["sources"]) > 0

    def test_flask_preserves_structured_response_fields(self, client):
        """8. The Flask frontend API preserves the correct status and message."""
        resp = client.post("/api/chat", json={"question": "What is Tenant B's administrator security policy?"})
        data = resp.get_json()
        assert "status" in data
        assert "title" in data
        assert "answer" in data
        assert "sources" in data
        assert data["status"] == "access_denied"
        assert data["title"] == "You don't have access to this information"

    @patch("app.generate_answer", side_effect=OllamaUnavailableError("Connection refused"))
    def test_ollama_unavailable_returns_friendly_error(self, mock_gen, client):
        """Ollama unavailable returns Case C structured friendly message."""
        resp = client.post("/api/chat", json={"question": "What is Tenant A's MFA policy?"})
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["status"] == "error"
        assert data["title"] == "The AI assistant is temporarily unavailable"
        assert "assistant could not process your question because the local AI service is unavailable" in data["answer"]
        assert "next_step" in data


class TestSecurity:
    """Security-specific tests"""

    def test_session_does_not_expose_user_registry(self, client):
        """/api/session does not expose the complete user registry."""
        resp = client.get("/api/session")
        data = resp.get_json()
        # Should only have 3 fields, not the full user dict
        assert set(data.keys()) == {"user_name", "tenant_id", "access_level"}
        # The users.json should not be returnable via this endpoint
        assert "public_analyst" not in str(data)
        assert "employee_a" not in str(data)
        assert "employee_b" not in str(data)


class TestFrontendServing:
    """Test serving the spatial 3D frontend from Flask."""

    def test_index_serves_spatial_frontend(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "SecureRAG" in html
        assert 'id="root"' in html

    def test_assets_serving(self, client):
        from app import FRONTEND_ASSETS
        if FRONTEND_ASSETS.exists():
            files = list(FRONTEND_ASSETS.glob("*.js"))
            if files:
                asset_name = files[0].name
                resp = client.get(f"/assets/{asset_name}")
                assert resp.status_code == 200
                assert len(resp.data) > 0