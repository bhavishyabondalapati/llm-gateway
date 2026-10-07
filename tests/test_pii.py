import pytest

from gateway.pii import StreamRestorer, restore
from gateway.schemas import ChatMessage


def msg(content, role="user"):
    return ChatMessage(role=role, content=content)


def test_redacts_common_pii(redactor):
    text = "Email John Smith at john.smith@acme.com or call 212-555-0199. Card 4111 1111 1111 1111."
    res = redactor.redact_messages([msg(text)])
    out = res.messages[0].content
    for secret in ("John Smith", "john.smith@acme.com", "212-555-0199", "4111 1111 1111 1111"):
        assert secret not in out
    assert "<PERSON_1>" in out and "<EMAIL_ADDRESS_1>" in out
    assert res.counts["CREDIT_CARD"] == 1


def test_round_trip_restores_original(redactor):
    text = "Hi, I'm Priya Patel and my email is priya@example.org."
    res = redactor.redact_messages([msg(text)])
    assert restore(res.messages[0].content, res.mapping) == text


def test_same_value_gets_same_placeholder_across_messages(redactor):
    res = redactor.redact_messages([
        msg("Remember the name Maria Garcia.", "system"),
        msg("What did Maria Garcia say? Also ask Wei Chen."),
    ])
    assert "<PERSON_1>" in res.messages[0].content
    assert "<PERSON_1>" in res.messages[1].content
    assert "<PERSON_2>" in res.messages[1].content
    assert res.mapping["<PERSON_1>"] == "Maria Garcia"


def test_text_without_pii_is_unchanged(redactor):
    text = "Write a haiku about autumn leaves."
    res = redactor.redact_messages([msg(text)])
    assert res.messages[0].content == text and res.total == 0


@pytest.mark.parametrize("text", [
    "Give me one fact about the number 4 and 0.266331.",
    "The ratio was 0.760948 in 2023.",
])
def test_decimals_are_not_phone_numbers(redactor, text):
    assert redactor.redact_messages([msg(text)]).total == 0


def test_content_parts_are_redacted(redactor):
    res = redactor.redact_messages([msg([{"type": "text", "text": "Contact bob@example.com"}])])
    assert res.messages[0].content[0]["text"] == "Contact <EMAIL_ADDRESS_1>"


def test_original_messages_are_not_mutated(redactor):
    original = msg("Contact bob@example.com")
    redactor.redact_messages([original])
    assert original.content == "Contact bob@example.com"


def test_restore_leaves_unknown_placeholders():
    assert restore("Hi <PERSON_9> and <PERSON_1>", {"<PERSON_1>": "Ann"}) == "Hi <PERSON_9> and Ann"


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 100])
def test_stream_restorer_handles_split_placeholders(size):
    mapping = {"<PERSON_1>": "Ann Lee", "<EMAIL_ADDRESS_1>": "ann@x.io"}
    text = "Dear <PERSON_1>, we wrote to <EMAIL_ADDRESS_1>. 3 < 4 is true. <PERSON_1>"
    r = StreamRestorer(mapping)
    out = "".join(r.feed(text[i : i + size]) for i in range(0, len(text), size)) + r.flush()
    assert out == restore(text, mapping)


def test_stream_restorer_does_not_hold_back_forever():
    r = StreamRestorer({"<PERSON_1>": "Ann"})
    first = r.feed("a < b and then a lot of ordinary text follows without closing")
    assert first.startswith("a < b")  # released once longer than any placeholder
