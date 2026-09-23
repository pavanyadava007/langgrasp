"""Fan-out of worker events to WebSocket clients, with the backpressure policy the design asks for.

One thread drains the worker's event queue and hands each event to the event loop. From there:

* JSON events are queued per client and never dropped while the buffer holds (a client that falls 500 events
  behind loses the oldest, and the UI notices because the run it is watching has moved on);
* frames are kept as one slot per camera and kind, so a slow client sees fewer frames rather than stale ones
  piling up. The worker itself never waits for any of this.

The hub also keeps a short replay buffer so a browser that connects mid-run can draw the current state, and
it resolves the futures that turn a fire-and-forget worker command into an HTTP response.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections import deque
from typing import Any, Callable

from langgrasp.gui.trace import pack_frame

REPLAY = 300  # events a late client gets on connect
CLIENT_JSON_BUFFER = 500


class ClientChannel:
    """One browser. Holds the JSON backlog and the newest frame per camera and kind."""

    def __init__(self, cameras: set[str] | None = None, depth: bool = False):
        self.json: deque[dict] = deque(maxlen=CLIENT_JSON_BUFFER)
        self.frames: dict[tuple[str, str], tuple[dict, bytes]] = {}
        self.wake = asyncio.Event()
        self.cameras = cameras or {"front"}
        self.depth = depth
        self.dropped_json = 0
        self.dropped_frames = 0

    def offer(self, event: dict, jpeg: bytes | None = None) -> None:
        if event["type"] == "frame":
            if event["camera"] not in self.cameras or (event["kind"] == "depth" and not self.depth):
                return
            key = (event["camera"], event["kind"])
            if key in self.frames:
                self.dropped_frames += 1
            self.frames[key] = (event, jpeg or b"")
        else:
            if len(self.json) == self.json.maxlen:
                self.dropped_json += 1
            self.json.append(event)
        self.wake.set()

    def take(self) -> tuple[list[dict], list[tuple[dict, bytes]]]:
        js = list(self.json)
        self.json.clear()
        frames = list(self.frames.values())
        self.frames.clear()
        self.wake.clear()
        return js, frames


class EventHub:
    """Bridges the worker's queue (a thread) and the WebSocket clients (the event loop)."""

    def __init__(self, worker):
        self.worker = worker
        self.clients: set[ClientChannel] = set()
        self.replay: deque[dict] = deque(maxlen=REPLAY)
        self.latest_frames: dict[tuple[str, str], tuple[dict, bytes]] = {}
        self.last: dict[str, dict] = {}  # newest event per type, for /api/system and late joiners
        self.stages: dict[str, dict] = {}  # stage -> its newest stage_finished event, for the current run
        self.run_id: str | None = None
        self.scene: dict | None = None
        self.models: dict = {}
        self.waiters: list[tuple[Callable[[dict], bool], asyncio.Future]] = []
        self.counts: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop = threading.Event()
        self.started_at = time.monotonic()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(target=self._drain, name="langgrasp-events", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _drain(self) -> None:
        """Thread: block on the worker queue, hand every event to the loop."""
        while not self._stop.is_set():
            event = self.worker.next_event(0.2)
            if event is None:
                continue
            loop = self._loop
            if loop is None or loop.is_closed():
                continue
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self.publish, event)

    # ------------------------------------------------------------------ publishing
    def publish(self, event: dict) -> None:
        jpeg = event.pop("jpeg", None)
        kind = event.get("type", "?")
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if kind == "frame":
            self.latest_frames[(event["camera"], event["kind"])] = (event, jpeg or b"")
        else:
            self.replay.append(event)
            self.last[kind] = event
            rid = event.get("run_id")
            if rid is not None and rid != self.run_id:
                self.run_id = rid
                self.stages = {}
            if kind == "stage_finished":
                self.stages[event["stage"]] = event
            elif kind == "system":
                if event.get("scene"):
                    self.scene = event["scene"]
                if event.get("models"):
                    self.models = event["models"]
        for c in list(self.clients):
            c.offer(event, jpeg)
        self._resolve(event)

    def _resolve(self, event: dict) -> None:
        still: list[tuple[Callable[[dict], bool], asyncio.Future]] = []
        for pred, fut in self.waiters:
            if fut.done():
                continue
            try:
                hit = pred(event)
            except Exception:  # noqa: BLE001 - a bad predicate must not poison the stream
                hit = False
            if hit:
                fut.set_result(event)
            else:
                still.append((pred, fut))
        self.waiters = still

    async def wait_for(self, predicate: Callable[[dict], bool], timeout: float) -> dict | None:
        """Wait for the next event matching a predicate. Returns None on timeout."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self.waiters.append((predicate, fut))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return None

    async def request(self, timeout: float = 30.0, **command: Any) -> dict:
        """Send a command that answers with a Reply event and wait for it."""
        token = f"r{int(time.monotonic() * 1e6)}"
        waiter = asyncio.ensure_future(self.wait_for(lambda e: e["type"] == "reply" and e.get("reply_to") == token, timeout))
        await asyncio.sleep(0)  # let the waiter register before the worker can answer
        self.worker.send(reply_to=token, **command)
        reply = await waiter
        if reply is None:
            raise TimeoutError(f"the worker did not answer '{command.get('cmd')}' within {timeout:.0f} s")
        if not reply.get("ok", True):
            raise RuntimeError(reply.get("error") or "the worker refused the command")
        return reply.get("data") or {}

    # ------------------------------------------------------------------ clients
    def add_client(self, cameras: set[str], depth: bool) -> ClientChannel:
        c = ClientChannel(cameras, depth)
        for event in self.replay:
            c.offer(event)
        for meta, jpeg in self.latest_frames.values():
            c.offer(meta, jpeg)
        self.clients.add(c)
        return c

    def remove_client(self, c: ClientChannel) -> None:
        self.clients.discard(c)

    @staticmethod
    def encode_frame(meta: dict, jpeg: bytes) -> bytes:
        return pack_frame(meta, jpeg)

    def stats(self) -> dict:
        return {
            "clients": len(self.clients),
            "events_published": dict(self.counts),
            "dropped_frames": sum(c.dropped_frames for c in self.clients),
            "dropped_json": sum(c.dropped_json for c in self.clients),
            "uptime_s": round(time.monotonic() - self.started_at, 1),
        }
