"""PII redaction (before forwarding) and restoration (in the response).

Flow:
    "Email John Smith at john@acme.com"
        -> redact ->  "Email <PERSON_1> at <EMAIL_ADDRESS_1>"   (sent to the LLM)
        mapping   = {"<PERSON_1>": "John Smith", "<EMAIL_ADDRESS_1>": "john@acme.com"}
    LLM reply "Sure, I'll email <PERSON_1>."
        -> restore -> "Sure, I'll email John Smith."             (sent to the client)

Presidio's AnalyzerEngine finds the PII spans. Replacement is done here (not with
presidio-anonymizer) because we need numbered, reversible placeholders that are
consistent across every message in the conversation.
"""

from __future__ import annotations

import re
import threading
from collections import Counter
from dataclasses import dataclass, field

from gateway.schemas import ChatMessage

DEFAULT_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "IBAN_CODE",
    "LOCATION",
]

PLACEHOLDER_RE = re.compile(r"<([A-Z_]+_\d+)>")
MAX_PLACEHOLDER_LEN = 32  # longest placeholder we might need to buffer while streaming


@dataclass
class RedactionResult:
    messages: list[ChatMessage]
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    counts: Counter = field(default_factory=Counter)  # entity type -> occurrences

    @property
    def total(self) -> int:
        return sum(self.counts.values())


class PIIRedactor:
    def __init__(
        self,
        entities: list[str] | None = None,
        score_threshold: float = 0.4,
        spacy_model: str = "en_core_web_md",
    ):
        self.entities = entities or DEFAULT_ENTITIES
        self.score_threshold = score_threshold
        self.spacy_model = spacy_model
        self._analyzer = None
        self._lock = threading.Lock()  # spaCy pipelines aren't guaranteed thread-safe

    @property
    def analyzer(self):
        if self._analyzer is None:
            # Imported lazily: loading spaCy takes a few seconds.
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import NlpEngineProvider

            nlp = NlpEngineProvider(
                nlp_configuration={
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": self.spacy_model}],
                }
            ).create_engine()
            self._analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        return self._analyzer

    def find_spans(self, text: str) -> list[tuple[int, int, str]]:
        """Return non-overlapping (start, end, entity_type) spans, sorted by start."""
        if not text.strip():
            return []
        with self._lock:
            results = self.analyzer.analyze(
                text=text,
                language="en",
                entities=self.entities,
                score_threshold=self.score_threshold,
            )
        # When spans overlap, keep the higher-scoring (then longer) one.
        results = sorted(results, key=lambda r: (-r.score, -(r.end - r.start)))
        chosen: list[tuple[int, int, str]] = []
        for r in results:
            if all(r.end <= s or r.start >= e for s, e, _ in chosen):
                chosen.append((r.start, r.end, r.entity_type))
        return sorted(chosen)

    def redact_messages(self, messages: list[ChatMessage]) -> RedactionResult:
        mapping: dict[str, str] = {}
        reverse: dict[tuple[str, str], str] = {}  # (type, original) -> placeholder
        per_type: Counter = Counter()
        counts: Counter = Counter()

        def redact_text(text: str) -> str:
            out, last = [], 0
            for start, end, etype in self.find_spans(text):
                original = text[start:end]
                key = (etype, original)
                if key not in reverse:
                    per_type[etype] += 1
                    ph = f"<{etype}_{per_type[etype]}>"
                    reverse[key] = ph
                    mapping[ph] = original
                counts[etype] += 1
                out.append(text[last:start])
                out.append(reverse[key])
                last = end
            out.append(text[last:])
            return "".join(out)

        new_messages = []
        for msg in messages:
            m = msg.model_copy(deep=True)
            if isinstance(m.content, str):
                m.content = redact_text(m.content)
            elif isinstance(m.content, list):
                for part in m.content:
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        part["text"] = redact_text(part["text"])
            new_messages.append(m)
        return RedactionResult(messages=new_messages, mapping=mapping, counts=counts)


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put original values back in place of placeholders."""
    if not mapping or not text:
        return text
    return PLACEHOLDER_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)


class StreamRestorer:
    """Restore placeholders in a streamed response.

    A placeholder can be split across chunks ("<PER" + "SON_1>"), so text after an
    unclosed "<" is held back until we know whether it is a placeholder.
    """

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.buffer = ""

    def feed(self, chunk: str) -> str:
        if not self.mapping:
            return chunk
        self.buffer += chunk
        cut = self.buffer.rfind("<")
        if cut != -1 and ">" not in self.buffer[cut:] and len(self.buffer) - cut < MAX_PLACEHOLDER_LEN:
            ready, self.buffer = self.buffer[:cut], self.buffer[cut:]
        else:
            ready, self.buffer = self.buffer, ""
        return restore(ready, self.mapping)

    def flush(self) -> str:
        out, self.buffer = restore(self.buffer, self.mapping), ""
        return out
