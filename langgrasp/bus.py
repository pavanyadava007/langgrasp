"""A tiny in-process publish/subscribe bus with ROS 2-like topic names.

The in-process pipeline and the ROS 2 nodes share the same stage functions; this bus lets the pipeline run
and be traced on a machine without ROS 2 (timestamps are stamped at publish time like header.stamp).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Message:
    topic: str
    data: Any
    stamp: float  # seconds, monotonic
    seq: int
    meta: dict = field(default_factory=dict)


class Bus:
    def __init__(self):
        self._subs: dict[str, list[Callable[[Message], None]]] = defaultdict(list)
        self._last: dict[str, Message] = {}
        self._seq = 0
        self.log: list[tuple[str, float]] = []

    def subscribe(self, topic: str, fn: Callable[[Message], None]):
        self._subs[topic].append(fn)

    def publish(self, topic: str, data: Any, stamp: float | None = None, **meta) -> Message:
        self._seq += 1
        msg = Message(topic, data, time.perf_counter() if stamp is None else stamp, self._seq, dict(meta))
        self._last[topic] = msg
        self.log.append((topic, msg.stamp))
        for fn in list(self._subs.get(topic, [])):
            fn(msg)
        return msg

    def last(self, topic: str) -> Message | None:
        return self._last.get(topic)

    def age(self, topic: str, now: float | None = None) -> float:
        m = self._last.get(topic)
        if m is None:
            return float("inf")
        return (time.perf_counter() if now is None else now) - m.stamp
