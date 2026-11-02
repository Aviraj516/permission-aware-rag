import pytest

from src.generation import (
    OllamaResponseError,
    validate_citations,
)


@pytest.fixture
def sources():
    return {
        "S1": {
            "source_file": "tenant_a_security_policy.txt",
            "chunk_id": "tenant_a_security_001",
        },
        "S2": {
            "source_file": "public_ai_policy.txt",
            "chunk_id": "public_ai_policy_001",
        },
    }


def test_valid_citation_is_accepted(sources):
    answer = "Tenant A requires MFA [S1]."
    assert validate_citations(answer, sources) == answer


def test_unknown_citation_is_rejected(sources):
    with pytest.raises(OllamaResponseError):
        validate_citations("The policy says this [S99].", sources)


def test_multiple_valid_citations_are_accepted(sources):
    answer = "Evidence supports this [S1] and this [S2]."
    assert validate_citations(answer, sources) == answer


def test_citation_spacing_is_normalized(sources):
    answer = validate_citations("Evidence supports this [S 1].", sources)
    assert answer == "Evidence supports this [S1]."


def test_unknown_citation_among_valid_citations_is_rejected(sources):
    with pytest.raises(OllamaResponseError):
        validate_citations("Supported [S1], but unknown [S99].", sources)