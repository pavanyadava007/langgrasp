// The live connection. Two things it must never do: block on a slow consumer, and re-render React on every
// frame. JSON events go to the store; JPEG frames go into a plain module-level slot per camera that the canvas
// reads on its own animation frame, so a 10 Hz stream costs one draw per frame and no React work.

import type { CameraName, FrameKind, FrameMetaEvent, WorkerEvent } from "./types";

const MAGIC = "LGF1";

export interface DecodedFrame {
  meta: FrameMetaEvent;
  bitmap: ImageBitmap;
  received: number;
}

type FrameKey = string;
const frames = new Map<FrameKey, DecodedFrame>();
const frameVersion = new Map<FrameKey, number>();
const fpsWindow: number[] = [];

export const frameStore = {
  key: (camera: CameraName, kind: FrameKind = "rgb"): FrameKey => `${camera}:${kind}`,
  get(camera: CameraName, kind: FrameKind = "rgb"): DecodedFrame | undefined {
    return frames.get(frameStore.key(camera, kind));
  },
  version(camera: CameraName, kind: FrameKind = "rgb"): number {
    return frameVersion.get(frameStore.key(camera, kind)) ?? 0;
  },
  /** Frames per second measured over the last two seconds of arrivals, for the viewport caption. */
  fps(): number {
    const now = performance.now();
    while (fpsWindow.length && now - fpsWindow[0] > 2000) fpsWindow.shift();
    return fpsWindow.length / 2;
  },
  put(meta: FrameMetaEvent, bitmap: ImageBitmap) {
    const key = frameStore.key(meta.camera, meta.kind);
    frames.get(key)?.bitmap.close();
    frames.set(key, { meta, bitmap, received: performance.now() });
    frameVersion.set(key, (frameVersion.get(key) ?? 0) + 1);
    if (meta.camera === "front" && meta.kind === "rgb") fpsWindow.push(performance.now());
  },
  clear() {
    for (const f of frames.values()) f.bitmap.close();
    frames.clear();
  },
};

function decodeEnvelope(buf: ArrayBuffer): { meta: FrameMetaEvent; jpeg: Blob } | null {
  const view = new DataView(buf);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== MAGIC) return null;
  const metaLen = view.getUint32(4, true);
  const metaText = new TextDecoder().decode(new Uint8Array(buf, 8, metaLen));
  return { meta: JSON.parse(metaText) as FrameMetaEvent, jpeg: new Blob([new Uint8Array(buf, 8 + metaLen)], { type: "image/jpeg" }) };
}

export type ConnectionState = "connecting" | "open" | "closed";

/** Replays a recorded run in the static build: the same events the worker emitted, in the order and at the
 *  spacing they happened, with the frames loaded from the files the run recorded. It is a replay and the page
 *  says so; nothing here pretends a simulator is running. */
export class ReplayConnection {
  private opts: LiveOptions;
  private timers: number[] = [];
  private stopped = false;

  constructor(opts: LiveOptions) {
    this.opts = opts;
  }

  async connect() {
    this.stopped = false;
    this.opts.onState("connecting");
    try {
      const manifest = (await (await fetch("data/replay.json")).json()) as { run_id: string; frames_url: string; hello: Record<string, unknown> };
      const detail = (await (await fetch(`data/runs/${manifest.run_id}.json`)).json()) as { events: (WorkerEvent & { t: number })[] };
      this.opts.onHello(manifest.hello);
      this.opts.onState("open");
      this.schedule(detail.events, manifest.frames_url);
    } catch (e) {
      this.opts.onState("closed", `the recorded run could not be loaded: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  private schedule(events: (WorkerEvent & { t: number })[], framesUrl: string) {
    if (!events.length) return;
    const t0 = Math.min(...events.map((e) => e.t));
    // Real spacing, but a run that takes six seconds should not make a visitor wait six seconds twice over.
    const speed = 1;
    for (const event of events) {
      const delay = Math.max(0, ((event.t - t0) * 1000) / speed);
      const id = window.setTimeout(() => {
        if (this.stopped) return;
        this.opts.onEvent(event);
        if (event.type === "tick") void this.loadFrame(framesUrl, (event as { tick: number }).tick, event.run_id ?? null);
      }, delay);
      this.timers.push(id);
    }
  }

  private async loadFrame(framesUrl: string, tick: number, runId: string | null) {
    const url = `${framesUrl}/tick_${String(tick).padStart(5, "0")}_front.jpg`;
    try {
      const res = await fetch(url);
      if (!res.ok) return;
      const bitmap = await createImageBitmap(await res.blob());
      frameStore.put(
        { type: "frame", t: performance.now() / 1000, run_id: runId, camera: "front", kind: "rgb", tick, width: bitmap.width, height: bitmap.height, bytes: 0 },
        bitmap,
      );
    } catch {
      /* a missing frame is not worth an error: the canvas keeps the last one */
    }
  }

  restart() {
    this.close();
    void this.connect();
  }

  subscribe() {
    /* nothing to subscribe to in a replay */
  }

  close() {
    this.stopped = true;
    for (const id of this.timers) window.clearTimeout(id);
    this.timers = [];
  }
}

export interface LiveOptions {
  cameras: CameraName[];
  depth: boolean;
  bodies: boolean;
  onEvent: (event: WorkerEvent) => void;
  onHello: (hello: Record<string, unknown>) => void;
  onState: (state: ConnectionState, detail?: string) => void;
}

export class LiveConnection {
  private ws: WebSocket | null = null;
  private opts: LiveOptions;
  private closedByUs = false;
  private retry = 0;
  private decoding = 0;

  constructor(opts: LiveOptions) {
    this.opts = opts;
  }

  connect() {
    this.closedByUs = false;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const params = new URLSearchParams({ cameras: this.opts.cameras.join(","), depth: this.opts.depth ? "1" : "0", bodies: this.opts.bodies ? "1" : "0" });
    const ws = new WebSocket(`${proto}://${location.host}/ws/live?${params}`);
    ws.binaryType = "arraybuffer";
    this.ws = ws;
    this.opts.onState("connecting");
    ws.onopen = () => {
      this.retry = 0;
      this.opts.onState("open");
    };
    ws.onclose = () => {
      this.opts.onState("closed", this.closedByUs ? "closed" : "the stream dropped, reconnecting");
      if (!this.closedByUs) {
        this.retry = Math.min(this.retry + 1, 6);
        setTimeout(() => this.connect(), 250 * 2 ** (this.retry - 1));
      }
    };
    ws.onerror = () => this.opts.onState("closed", "the stream could not be opened");
    ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const parsed = JSON.parse(ev.data);
        if (parsed.type === "hello") this.opts.onHello(parsed);
        else this.opts.onEvent(parsed as WorkerEvent);
        return;
      }
      // Frames: decode off the main thread and drop any that pile up, so the newest one always wins.
      if (this.decoding > 2) return;
      const decoded = decodeEnvelope(ev.data as ArrayBuffer);
      if (!decoded) return;
      this.decoding += 1;
      createImageBitmap(decoded.jpeg)
        .then((bitmap) => frameStore.put(decoded.meta, bitmap))
        .catch(() => undefined)
        .finally(() => {
          this.decoding -= 1;
        });
    };
  }

  subscribe(cameras: CameraName[], depth: boolean, bodies: boolean) {
    this.opts.cameras = cameras;
    this.opts.depth = depth;
    this.opts.bodies = bodies;
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ cameras, depth, bodies }));
  }

  close() {
    this.closedByUs = true;
    this.ws?.close();
  }
}
