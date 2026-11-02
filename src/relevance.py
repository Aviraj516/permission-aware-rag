
import re


def detect_requested_tenant(question):
    """Detect an explicitly named synthetic tenant in a question."""
    matches = re.findall(
        r"\btenant[\s_-]+([a-z]+)\b",
        question.lower(),
    )

    for name in matches:
        if name in {"a", "b"}:
            return f"tenant_{name}"

    return None


def format_tenant_title(tenant_id):
    """Return a clean, human-readable display title for a tenant identifier."""
    if not tenant_id:
        return "Other Tenant"
    if tenant_id == "tenant_a":
        return "Tenant A"
    if tenant_id == "tenant_b":
        return "Tenant B"
    if tenant_id == "microsoft_public":
        return "Microsoft Public"
    return tenant_id.replace("_", " ").title()


def is_cross_tenant_request(question, user):
    """
    Check whether a question explicitly targets another tenant.

    This checks the requested tenant against the current user's
    authorized tenant. It does not check whether documents belonging
    to the other tenant exist.

    Public information remains accessible to all authorized users.
    """
    requested_tenant = detect_requested_tenant(question)

    if requested_tenant is None:
        return False

    if requested_tenant == "microsoft_public":
        return False

    return requested_tenant != user["tenant_id"]


def filter_relevant_chunks(
    question,
    chunks,
    user,
    min_similarity=0.45,
    max_chunks=4,
):
    """
    Return authorized, relevant chunks.

    This is a relevance and defense-in-depth layer, not the primary
    authorization mechanism. Authorization must also be enforced
    during retrieval.

    Optimized to eliminate duplicate or redundant chunks and cap at
    max_chunks to avoid unnecessarily large prompts while keeping latency low.
    """
    # Prevent explicitly requested cross-tenant private information
    # from being used to answer the question.
    if is_cross_tenant_request(question, user):
        return []

    relevant = []
    seen_chunk_ids = set()
    seen_texts = set()

    for chunk in chunks:
        # Public Microsoft documents can be used by every user.
        is_public = (
            chunk.get("tenant_id") == "microsoft_public"
            and chunk.get("access_level") == "public"
        )

        # Internal and confidential documents are restricted to the
        # current user's tenant.
        belongs_to_user = (
            chunk.get("tenant_id") == user["tenant_id"]
            and chunk.get("access_level")
            in {"internal", "confidential"}
        )

        # Never pass unauthorized chunks to the generation model.
        if not (is_public or belongs_to_user):
            continue

        # Ignore chunks that do not meet the relevance threshold.
        similarity = chunk.get("similarity")

        if not isinstance(similarity, (int, float)):
            continue

        if similarity < min_similarity:
            continue

        # Avoid duplicate or redundant chunks
        cid = chunk.get("chunk_id")
        text = chunk.get("text", "").strip()

        if cid and cid in seen_chunk_ids:
            continue
        if text and text in seen_texts:
            continue

        if cid:
            seen_chunk_ids.add(cid)
        if text:
            seen_texts.add(text)

        relevant.append(chunk)

    # Sort by relevance descending and cap at max_chunks
    relevant.sort(key=lambda c: c.get("similarity", 0), reverse=True)
    return relevant[:max_chunks] if max_chunks else relevant

