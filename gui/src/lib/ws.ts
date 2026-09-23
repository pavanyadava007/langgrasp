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
