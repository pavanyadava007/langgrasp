// Mirror of langgrasp/gui/trace.py. Keep the two in step: the Python side is the source of truth.

export const STAGES = ["command", "parse", "capture", "grounding", "select", "gate", "segment", "fuse", "execute"] as const;
export type StageName = (typeof STAGES)[number];
export type StageStatus = "running" | "ok" | "warn" | "fail" | "skipped" | "pending";
export type SafetyStateName = "RUN" | "REDUCED_SPEED" | "HOLD" | "ESTOP";
export type CameraName = "front" | "wrist" | "side";
export type FrameKind = "rgb" | "depth";
export type LatencyKind = "compute" | "wall_paced" | "none";
export type ControllerName = "pipeline" | "oracle" | "act" | "ppo_reach" | "ppo_lift";

export interface BaseEvent {
  type: string;
  t: number;
  run_id: string | null;
}

export interface StageStartedEvent extends BaseEvent {
  type: "stage_started";
  stage: StageName;
  payload: Record<string, unknown>;
}

export interface StageFinishedEvent extends BaseEvent {
  type: "stage_finished";
  stage: StageName;
  status: Exclude<StageStatus, "pending">;
  latency_ms: number | null;
  latency_kind: LatencyKind;
  message: string | null;
  payload: Record<string, unknown>;
  images: Record<string, string>;
}

export interface FrameMetaEvent extends BaseEvent {
  type: "frame";
  camera: CameraName;
  kind: FrameKind;
  tick: number | null;
  width: number;
  height: number;
  bytes: number;
}

export interface SafetyEvent extends BaseEvent {
  type: "safety";
  state: SafetyStateName;
  reason: string;
  mode: "monitor" | "enforce";
  gate: Record<string, unknown> | null;
  watchdog: Record<string, number | null>;
  clips: { limit?: number; velocity?: number };
  would_clip: { limit?: number; velocity?: number };
  estop_latency_ms: number | null;
  estop_latency_note: string | null;
}

export interface TickEvent extends BaseEvent {
  type: "tick";
  tick: number;
  phase: string;
  q: number[];
  q_target: number[];
  jaw: number;
  tcp: number[];
  sim_t: number | null;
  bodies: Record<string, number[]> | null;
}

export interface OutcomeEvent extends BaseEvent {
  type: "outcome";
  grounding_correct: boolean | null;
  grasped: boolean;
  lifted: boolean;
  placed: boolean;
  aborted: string | null;
  scored_by: "ground_truth" | "none";
  message: string;
  latency_ms: Record<string, number>;
  sim_compute_ms: number | null;
  wall_ms: number | null;
}

export interface GateRequestEvent extends BaseEvent {
  type: "gate_request";
  box: number[] | null;
  score: number | null;
  top2: number[];
  threshold: number | null;
  ambiguous: boolean;
  needs_confirmation: boolean;
  timeout_s: number | null;
}

export interface SystemEvent extends BaseEvent {
  type: "system";
  models: Record<string, ModelState>;
  gpu: string | null;
  hardware_label: string | null;
  scene: Scenario | null;
  note: string | null;
}

export interface LogEvent extends BaseEvent {
  type: "log";
  level: "info" | "warn" | "error";
  message: string;
  detail: Record<string, unknown> | null;
}

export interface ReplyEvent extends BaseEvent {
  type: "reply";
  reply_to: string;
  ok: boolean;
  data: Record<string, unknown>;
  error: string | null;
}

export interface JobProgressEvent extends BaseEvent {
  type: "job_progress";
  job_id: string;
  state: "queued" | "running" | "done" | "cancelled" | "error";
  done: number;
  total: number;
  out_path: string | null;
  last: Record<string, unknown> | null;
  message: string | null;
}

export type WorkerEvent =
  | StageStartedEvent
  | StageFinishedEvent
  | FrameMetaEvent
  | SafetyEvent
  | TickEvent
  | OutcomeEvent
  | GateRequestEvent
  | SystemEvent
  | LogEvent
  | ReplyEvent
  | JobProgressEvent;

export interface ModelState {
  state: "absent" | "loading" | "warm" | "missing" | "error";
  load_ms: number | null;
  detail: string | null;
}

export interface PlacedObject {
  name: string;
  kind: string;
  color: string;
  pos: [number, number];
  yaw: number;
}

export interface Scenario {
  stratum: "seen" | "unseen" | "langvar";
  objects: PlacedObject[];
  target: string;
  command: string;
  lang_variant: string;
  lighting: "nominal" | "degraded";
  seed: number;
  fixed_goal?: boolean;
}

export interface SystemInfo {
  banner: string;
  hardware_label: string | null;
  gpu: string | null;
  worker: { alive: boolean; busy: boolean; pid: number | null };
  models: Record<string, ModelState>;
  versions: Record<string, string | null>;
  python: string;
  engines: { path: string; present: boolean; size_mb: number | null; mtime: number | null }[];
  stages: { name: StageName; title: string; help: string }[];
  cameras: CameraName[];
  safety: SafetyEvent | null;
  safety_note: string;
  hub: Record<string, unknown>;
  results_dir: string;
}

export interface ExampleCommand {
  stratum: string;
  command: string;
  why: string;
}

// Payload shapes worth naming, because the overlays and the drawer read them.
export interface Candidate {
  box: [number, number, number, number];
  score: number;
  label: string;
  color_frac: number;
  area_px: number;
}

export interface GraspPose {
  center: [number, number, number];
  psi: number;
  width: number;
  long_axis: number[];
  aspect: number;
  n: number;
  elongated: boolean;
  fallback: boolean;
  grasp_z?: number;
  median_z?: number;
  max_z?: number;
}

export interface RunConfig {
  use_color_check: boolean;
  use_yolo_mask: boolean;
  depth_noise: boolean;
  seg_backend: string;
  require_human_confirm: boolean;
  safety_mode: "monitor" | "enforce";
  speed: number;
}
