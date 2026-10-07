"""Pick a semantic-cache threshold using a small labeled set of prompt pairs.

For each pair we ask: would the cache serve prompt B the answer cached for A?
"Should hit" pairs are rewordings that deserve the same answer; "should miss"
pairs look alike but need a different answer. Uses the real cache logic, so
the number/placeholder guards are included.

    python scripts/eval_semantic_threshold.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.cache import CachedResponse, ResponseCache, conversation_text  # noqa: E402
from gateway.embeddings import HashingEmbedder  # noqa: E402
from gateway.schemas import ChatCompletionRequest  # noqa: E402

SHOULD_HIT = [
    ("What is the capital of France?", "what is the capital of france"),
    ("What is the capital of France?", "What is the capital of France??"),
    ("Explain how transformers work in machine learning.", "Explain how transformers work in machine learning, please."),
    ("Summarize this article in 3 bullet points.", "summarize this article in 3 bullet points"),
    ("Write a SQL query that counts users by country.", "Write a SQL query that counts users by country"),
    ("Draft a polite email to <PERSON_1> about the late invoice.", "Draft a polite email to <PERSON_1> about the late invoice!"),
    ("Translate 'good morning' into Spanish.", "Translate \"good morning\" into Spanish"),
    ("Give me 5 tips for better sleep.", "Give me 5 tips for better sleep, thanks."),
    ("How do I reverse a list in Python?", "how do i reverse a list in python?"),
    ("What are the main causes of inflation?", "What are the main causes of inflation"),
]
SHOULD_MISS = [
    ("What is the capital of France?", "What is the capital of Spain?"),
    ("Write a poem about cats", "Write a poem about dogs"),
    ("Translate 'hello' to French", "Translate 'goodbye' to French"),
    ("Give me one fact about the number 12", "Give me one fact about the number 13"),
    ("Summarize the quarterly report for the team in three bullet points, focusing on revenue growth and churn.",
     "Summarize the quarterly report for the team in three bullet points, focusing on revenue growth and costs."),
    ("How do I reverse a list in Python?", "How do I reverse a list in JavaScript?"),
    ("Draft a polite email to <PERSON_1> about the late invoice.", "Draft a polite email to <EMAIL_ADDRESS_1> about the late invoice."),
    ("Is 17 a prime number?", "Is 21 a prime number?"),
    ("Explain the difference between TCP and UDP.", "Explain the difference between HTTP and HTTPS."),
    ("Write a SQL query that counts users by country.", "Write a SQL query that counts orders by country."),
]


def similarity(a: str, b: str) -> float | None:
    """Cosine similarity the cache would see, or None if the scope guard blocks the match."""
    cache = ResponseCache(HashingEmbedder(), semantic_threshold=-1.0)
    ra = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": a}])
    rb = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": b}])
    cache.put("k", cache.semantic_scope("t", ra), cache.embed(conversation_text(ra)),
              CachedResponse("", "stop", "m", 0, 0))
    hit = cache.get_semantic(cache.semantic_scope("t", rb), cache.embed(conversation_text(rb)))
    return None if hit is None else hit[1]


def main():
    hit_sims = [similarity(a, b) for a, b in SHOULD_HIT]
    miss_sims = [similarity(a, b) for a, b in SHOULD_MISS]
    print(f"{'threshold':>9} | {'hits caught':>11} | {'false hits':>10}")
    for t in (0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.98):
        caught = sum(s is not None and s >= t for s in hit_sims)
        false = sum(s is not None and s >= t for s in miss_sims)
        print(f"{t:>9.2f} | {caught:>5}/{len(hit_sims):<5} | {false:>4}/{len(miss_sims):<5}")
    print("\nShould-miss pairs and their similarity ('guard' = blocked by number/placeholder check):")
    for (a, b), s in zip(SHOULD_MISS, miss_sims):
        print(f"  {'guard' if s is None else f'{s:.3f}':>6}  {a[:45]!r} vs {b[:45]!r}")


if __name__ == "__main__":
    main()
