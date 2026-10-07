"""Micro-benchmark: how long does Presidio redaction take per message size?

    python scripts/bench_pii.py
"""

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway.pii import PIIRedactor  # noqa: E402
from gateway.schemas import ChatMessage  # noqa: E402

SENTENCE = ("Please email Maria Garcia at maria.garcia@example.com or call 617-555-0142 "
            "about the contract renewal we discussed last week. ")
FILLER = "The quarterly numbers look good and the team is on track for the launch. "


def main():
    r = PIIRedactor(memo_size=0)  # measure real Presidio work, not memo hits
    t0 = time.perf_counter()
    r.analyzer  # noqa: B018
    print(f"model load: {time.perf_counter() - t0:.1f} s\n")
    print(f"{'chars':>6} | {'p50 ms':>7} | {'p95 ms':>7}")
    for repeats in (1, 5, 20, 80):
        text = SENTENCE + FILLER * (repeats - 1)
        msgs = [ChatMessage(role="user", content=text)]
        for _ in range(3):
            r.redact_messages(msgs)  # warm-up
        times = []
        for _ in range(30):
            t = time.perf_counter()
            r.redact_messages(msgs)
            times.append((time.perf_counter() - t) * 1000)
        times.sort()
        print(f"{len(text):>6} | {statistics.median(times):>7.1f} | {times[int(len(times) * 0.95) - 1]:>7.1f}")


if __name__ == "__main__":
    main()
