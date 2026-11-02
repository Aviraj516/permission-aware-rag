
from generation import generate_answer


sample_chunks = [
    {
        "chunk_id": "public_ai_policy_001",
        "source_file": "public_ai_policy.txt",
        "tenant_id": "microsoft_public",
        "access_level": "public",
        "similarity": 0.90,
        "text": (
            "AI systems should be evaluated for reliability, safety, "
            "fairness, privacy, and security. Organizations should "
            "document risks and monitor systems after deployment."
        ),
    }
]

if __name__ == "__main__":
    question = "Which aspects should AI systems be evaluated for?"
    answer = generate_answer(question, sample_chunks)
    print("\nQuestion:", question)
    print("\nAnswer:\n", answer)
