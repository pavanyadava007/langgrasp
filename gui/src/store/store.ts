// One store for everything the UI shows, fed by the live event stream and the REST calls.
// Camera frames deliberately do NOT live here (see lib/ws.ts): they would re-render the tree 10 times a second.

import { create } from "zustand";
import { api, ApiError } from "../lib/api";
import { frameStore, LiveConnection, type ConnectionState } from "../lib/ws";
import { tickStore } from "../lib/ticks";
import {
  STAGES,
  type CameraName,
  type ControllerName,
  type ExampleCommand,
  type GateRequestEvent,
  type OutcomeEvent,
  type RunConfig,
  type SafetyEvent,
  type Scenario,
  type StageFinishedEvent,
  type StageName,
  type StageStatus,
  type SystemInfo,
  type TickEvent,
  type WorkerEvent,
} from "../lib/types";

export interface StageView {
  status: StageStatus;
  latency_ms: number | null;
  latency_kind: string;
  message: string | null;
  payload: Record<string, unknown>;
  images: Record<string, string>;
}

const emptyStage = (): StageView => ({ status: "pending", latency_ms: null, latency_kind: "none", message: null, payload: {}, images: {} });
const emptyStages = (): Record<StageName, StageView> => Object.fromEntries(STAGES.map((s) => [s, emptyStage()])) as Record<StageName, StageView>;

export type ViewName = "live" | "inspector" | "results" | "batch" | "safety";

export interface OverlayLayers {
  candidates: boolean;
  winner: boolean;
  mask: boolean;
  points: boolean;
  grasp: boolean;
  waypoints: boolean;
  geofence: boolean;
  groundTruth: boolean;
}

export interface Notice {
  id: number;
  level: "info" | "warn" | "error";
  message: string;
  at: number;
}

interface State {
  // connection and system
  connection: ConnectionState;
  connectionDetail: string;
  system: SystemInfo | null;
  banner: string;

  // scene
  scene: Scenario | null;
  examples: ExampleCommand[];
  sceneForm: { seed: number; stratum: "seen" | "unseen" | "langvar"; lighting: "" | "nominal" | "degraded"; n_distractors: number | ""; fixed_goal: boolean };

  // command and controller
  command: string;
  controller: ControllerName;
  config: RunConfig;

  // the current run
  runId: string | null;
  running: boolean;
  stages: Record<StageName, StageView>;
  lastTick: TickEvent | null;
  tickCount: number;
  outcome: OutcomeEvent | null;
  gate: GateRequestEvent | null;
  safety: SafetyEvent | null;
  estopLatencyMs: number | null;
  paused: boolean;

  // ui
  view: ViewName;
  theme: "dark" | "light";
  camera: CameraName;
  showDepth: boolean;
  layers: OverlayLayers;
  drawerStage: StageName | null;
  notices: Notice[];
  announce: string;
  shortcutsOpen: boolean;

  // actions
  init: () => Promise<void>;
  applyEvent: (event: WorkerEvent) => void;
  setState: (patch: Partial<State>) => void;
  setConfig: (patch: Partial<RunConfig>) => void;
  setSceneForm: (patch: Partial<State["sceneForm"]>) => void;
  setLayer: (key: keyof OverlayLayers, value: boolean) => void;
  setCamera: (camera: CameraName, depth: boolean) => void;
  setTheme: (theme: "dark" | "light") => void;
  setView: (view: ViewName) => void;
  openDrawer: (stage: StageName | null) => void;
  notify: (level: Notice["level"], message: string) => void;
  dismiss: (id: number) => void;
  newScene: () => Promise<void>;
  replay: () => Promise<void>;
  startRun: (source?: string, sttLatency?: number | null) => Promise<void>;
  estop: () => Promise<void>;
  reset: () => Promise<void>;
  togglePause: () => Promise<void>;
  stepOnce: () => Promise<void>;
  confirmGate: (decision: "confirm" | "reject") => Promise<void>;
  refreshSystem: () => Promise<void>;
}

let live: LiveConnection | null = null;
let noticeId = 1;

const storedTheme = (): "dark" | "light" => (localStorage.getItem("langgrasp.theme") === "light" ? "light" : "dark");

export const useStore = create<State>((set, get) => ({
  connection: "connecting",
  connectionDetail: "",
  system: null,
  banner: "Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware",

  scene: null,
  examples: [],
  sceneForm: { seed: 5000, stratum: "seen", lighting: "", n_distractors: "", fixed_goal: false },

  command: "",
  controller: "pipeline",
  config: { use_color_check: true, use_yolo_mask: true, depth_noise: true, seg_backend: "pt", require_human_confirm: false, safety_mode: "monitor", speed: 1.0 },

  runId: null,
  running: false,
  stages: emptyStages(),
  lastTick: null,
  tickCount: 0,
  outcome: null,
  gate: null,
  safety: null,
  estopLatencyMs: null,
  paused: false,

  view: "live",
  theme: storedTheme(),
  camera: "front",
  showDepth: false,
  layers: { candidates: true, winner: true, mask: true, points: false, grasp: true, waypoints: false, geofence: false, groundTruth: false },
  drawerStage: null,
  notices: [],
  announce: "",
  shortcutsOpen: false,

  setState: (patch) => set(patch),
  setConfig: (patch) => set({ config: { ...get().config, ...patch } }),
  setSceneForm: (patch) => set({ sceneForm: { ...get().sceneForm, ...patch } }),
  setLayer: (key, value) => set({ layers: { ...get().layers, [key]: value } }),
  setView: (view) => set({ view }),
  openDrawer: (stage) => set({ drawerStage: stage }),
  notify: (level, message) => set({ notices: [...get().notices.slice(-4), { id: noticeId++, level, message, at: Date.now() }], announce: message }),
  dismiss: (id) => set({ notices: get().notices.filter((n) => n.id !== id) }),

  setTheme: (theme) => {
    localStorage.setItem("langgrasp.theme", theme);
    document.documentElement.dataset.theme = theme;
    set({ theme });
  },

  setCamera: (camera, depth) => {
    set({ camera, showDepth: depth });
    const cams: CameraName[] = camera === "front" ? ["front"] : ["front", camera];
    live?.subscribe(cams, depth, get().layers.geofence);
  },

  async init() {
    document.documentElement.dataset.theme = get().theme;
    await get().refreshSystem();
    try {
      const { scene, examples } = await api.scene();
      set({ examples });
      if (scene) set({ scene, command: scene.command, sceneForm: { ...get().sceneForm, seed: scene.seed, stratum: scene.stratum } });
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
    live = new LiveConnection({
      cameras: ["front"],
      depth: false,
      bodies: false,
      onEvent: (event) => get().applyEvent(event),
      onHello: (hello) => {
        if (hello.banner) set({ banner: String(hello.banner) });
        if (hello.scene) set({ scene: hello.scene as Scenario });
      },
      onState: (state, detail) => set({ connection: state, connectionDetail: detail ?? "" }),
    });
    live.connect();
  },

  async refreshSystem() {
    try {
      const system = await api.system();
      set({ system, banner: system.banner, safety: system.safety ?? get().safety });
    } catch (e) {
      set({ connection: "closed", connectionDetail: e instanceof ApiError ? e.message : String(e) });
    }
  },

  applyEvent(event) {
    switch (event.type) {
      case "system": {
        if (event.scene) set({ scene: event.scene });
        if (event.note === "models warm" || event.note?.includes(":")) void get().refreshSystem();
        break;
      }
      case "stage_started": {
        const stages = { ...get().stages, [event.stage]: { ...get().stages[event.stage], status: "running" as StageStatus } };
        set({ stages, runId: event.run_id ?? get().runId, running: true });
        break;
      }
      case "stage_finished": {
        const e = event as StageFinishedEvent;
        const prev = get().stages[e.stage];
        set({
          stages: { ...get().stages, [e.stage]: { status: e.status, latency_ms: e.latency_ms, latency_kind: e.latency_kind, message: e.message, payload: e.payload, images: { ...prev.images, ...e.images } } },
          announce: e.status === "ok" ? "" : `${e.stage}: ${e.status}${e.message ? `, ${e.message}` : ""}`,
        });
        break;
      }
      case "tick": {
        tickStore.push(event as TickEvent);
        set({ lastTick: event as TickEvent, tickCount: get().tickCount + 1 });
        break;
      }
      case "outcome": {
        set({ outcome: event as OutcomeEvent, running: false, gate: null, announce: `Outcome: ${(event as OutcomeEvent).message}` });
        break;
      }
      case "gate_request": {
        set({ gate: event as GateRequestEvent });
        break;
      }
      case "safety": {
        const e = event as SafetyEvent;
        set({ safety: e, estopLatencyMs: e.estop_latency_ms ?? get().estopLatencyMs });
        if (e.state === "ESTOP") set({ announce: `E-stop: ${e.reason}`, running: false });
        break;
      }
      case "log": {
        if (event.level !== "info") get().notify(event.level, event.message);
        break;
      }
      default:
        break;
    }
  },

  async newScene() {
    const f = get().sceneForm;
    try {
      const { scene } = await api.newScene({
        seed: f.seed,
        stratum: f.stratum,
        lighting: f.lighting === "" ? null : f.lighting,
        n_distractors: f.n_distractors === "" ? null : Number(f.n_distractors),
        fixed_goal: f.fixed_goal,
      });
      frameStore.clear();
      tickStore.clear();
      set({ scene, command: scene.command, stages: emptyStages(), outcome: null, lastTick: null, tickCount: 0, gate: null, runId: null, drawerStage: null });
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },

  async replay() {
    await get().newScene();
    await get().startRun("chip");
  },

  async startRun(source = "typed", sttLatency = null) {
    const { command, controller, config } = get();
    tickStore.clear();
    set({ stages: emptyStages(), outcome: null, tickCount: 0, lastTick: null, gate: null, running: true });
    try {
      const { run_id } = await api.run({ command: command || null, controller, config, source, stt_latency_ms: sttLatency });
      set({ runId: run_id });
    } catch (e) {
      set({ running: false });
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },

  async estop() {
    try {
      const r = await api.estop();
      set({ estopLatencyMs: r.estop_latency_ms ?? get().estopLatencyMs, running: false });
      get().notify("warn", `E-stop latched${r.estop_latency_ms ? ` (${r.estop_latency_ms.toFixed(1)} ms from click to the arm being held)` : ""}.`);
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },

  async reset() {
    try {
      const r = await api.reset();
      get().notify("info", `E-stop reset. State is now ${r.state}.`);
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },

  async togglePause() {
    const next = !get().paused;
    try {
      await api.pause({ paused: next });
      set({ paused: next });
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },

  async stepOnce() {
    try {
      await api.pause({ step: true });
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },

  async confirmGate(decision) {
    try {
      await api.confirm(decision);
      set({ gate: null });
    } catch (e) {
      get().notify("error", e instanceof ApiError ? e.message : String(e));
    }
  },
}));

export const stageOrder = STAGES;
