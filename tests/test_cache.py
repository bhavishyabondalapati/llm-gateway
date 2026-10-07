import time

from gateway.cache import CachedResponse, ResponseCache, conversation_text
from gateway.embeddings import HashingEmbedder
from gateway.schemas import ChatCompletionRequest

VALUE = CachedResponse("answer", "stop", "m", 10, 5)


def req(content, **kw):
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": content}], **kw)


def put(cache, team, r):
    cache.put(cache.exact_key(team, r), cache.semantic_scope(team, r), cache.embed(conversation_text(r)), VALUE)


def semantic(cache, team, r):
    return cache.get_semantic(cache.semantic_scope(team, r), cache.embed(conversation_text(r)))


def test_exact_hit_and_param_sensitivity():
    c = ResponseCache(HashingEmbedder())
    put(c, "a", req("hello there"))
    assert c.get_exact(c.exact_key("a", req("hello there"))) == VALUE
    assert c.get_exact(c.exact_key("a", req("hello there", temperature=0.9))) is None
    # stream flag doesn't change the answer, so it shares the entry
    assert c.get_exact(c.exact_key("a", req("hello there", stream=True))) == VALUE


def test_teams_are_isolated():
    c = ResponseCache(HashingEmbedder())
    put(c, "a", req("What is the capital of France?"))
    assert c.get_exact(c.exact_key("b", req("What is the capital of France?"))) is None
    assert semantic(c, "b", req("what is the capital of france")) is None


def test_semantic_hit_for_near_duplicate_and_miss_for_different_question():
    c = ResponseCache(HashingEmbedder(), semantic_threshold=0.95)
    put(c, "a", req("What is the capital of France?"))
    hit = semantic(c, "a", req("what is the capital of france"))
    assert hit is not None and hit[1] > 0.95
    assert semantic(c, "a", req("What is the capital of Spain?")) is None


def test_semantic_requires_same_placeholders():
    c = ResponseCache(HashingEmbedder(), semantic_threshold=0.5)
    put(c, "a", req("Write to <PERSON_1> about the invoice"))
    assert semantic(c, "a", req("write to <PERSON_1> about the invoice!")) is not None
    assert semantic(c, "a", req("Write to <EMAIL_ADDRESS_1> about the invoice")) is None


def test_ttl_expiry():
    c = ResponseCache(HashingEmbedder(), ttl_seconds=0)
    r = req("hi")
    put(c, "a", r)
    time.sleep(0.01)
    assert c.get_exact(c.exact_key("a", r)) is None


def test_lru_eviction():
    c = ResponseCache(None, max_entries=2)
    for q in ("one", "two"):
        put(c, "a", req(q))
    c.get_exact(c.exact_key("a", req("one")))  # touch "one" so "two" is the oldest
    put(c, "a", req("three"))
    assert len(c) == 2
    assert c.get_exact(c.exact_key("a", req("two"))) is None
    assert c.get_exact(c.exact_key("a", req("one"))) is not None


def test_semantic_requires_same_numbers():
    c = ResponseCache(HashingEmbedder(), semantic_threshold=0.5)
    put(c, "a", req("Give me one fact about the number 12"))
    assert semantic(c, "a", req("give me one fact about the number 12!")) is not None
    assert semantic(c, "a", req("Give me one fact about the number 13")) is None
