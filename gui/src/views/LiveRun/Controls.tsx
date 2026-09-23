import { Button, Field, NumberInput, Select, Term, Toggle } from "../../components/ui";
import { useStore } from "../../store/store";
import type { ControllerName } from "../../lib/types";

// Scene, controller and ablations. Two rules here: a backend whose file is missing is offered but marked, and a
// controller that cannot run this command is disabled with the reason rather than hidden.

const SEG_BACKENDS: { value: string; label: string; file: string }[] = [
  { value: "pt", label: "PyTorch", file: "checkpoints/yolo11n-seg-langgrasp.pt" },
  { value: "pt-fp16", label: "PyTorch FP16", file: "checkpoints/yolo11n-seg-langgrasp.pt" },
  { value: "onnx", label: "ONNX Runtime CUDA", file: "checkpoints/yolo11n-seg-langgrasp.onnx" },
  { value: "engine", label: "TensorRT FP16", file: "checkpoints/yolo11n-seg-langgrasp-fp16.engine" },
  { value: "engine-int8", label: "TensorRT INT8", file: "checkpoints/yolo11n-seg-langgrasp-int8.engine" },
];

export function Controls() {
  const form = useStore((s) => s.sceneForm);
  const setSceneForm = useStore((s) => s.setSceneForm);
  const newScene = useStore((s) => s.newScene);
  const replay = useStore((s) => s.replay);
  const controller = useStore((s) => s.controller);
  const setState = useStore((s) => s.setState);
  const config = useStore((s) => s.config);
  const setConfig = useStore((s) => s.setConfig);
  const system = useStore((s) => s.system);
  const engines = system?.engines ?? [];
  const command = useStore((s) => s.command);
  const running = useStore((s) => s.running);

  const present = new Map(engines.map((e) => [e.path, e.present]));
  const fixedGoalCommand = /red\s+cube/i.test(command);

  const controllers: { value: ControllerName; label: string; disabled?: boolean; why?: string }[] = [
    { value: "pipeline", label: "Modular pipeline" },
    { value: "oracle", label: "Oracle (ground-truth pose)", why: "Skips perception entirely and takes the grasp pose from the simulator. The executor's upper bound." },
    {
      value: "act",
      label: "ACT (fixed goal)",
      disabled: !fixedGoalCommand,
      why: fixedGoalCommand
        ? "Not wired into the worker yet."
        : "ACT takes images and joint state, not language, so it only has one goal: the red cube. Write a command for the red cube to enable it.",
    },
    { value: "ppo_reach", label: "PPO reach", disabled: true, why: "PPO runs on its own state-based environment with no camera in the loop. Not wired into the worker yet." },
    { value: "ppo_lift", label: "PPO lift", disabled: true, why: "As above." },
  ];

  return (
    <div className="space-y-4">
      <div>
        <h2 className="mb-2 text-sm font-semibold">Scene</h2>
        <Field label="Seed" htmlFor="seed" hint="The scene and its command are a pure function of this number.">
          <NumberInput id="seed" value={form.seed} onChange={(v) => setSceneForm({ seed: v === "" ? 0 : v })} min={0} />
        </Field>
        <Field label="Stratum" htmlFor="stratum">
          <Select
            id="stratum"
            value={form.stratum}
            onChange={(v) => setSceneForm({ stratum: v as "seen" | "unseen" | "langvar" })}
            options={[
              { value: "seen", label: "seen: kinds and colours from training" },
              { value: "unseen", label: "unseen: novel colour or shape" },
              { value: "langvar", label: "language variation: synonym, attribute, spatial" },
            ]}
          />
        </Field>
        <Field label="Lighting" htmlFor="lighting">
          <Select
            id="lighting"
            value={form.lighting}
            onChange={(v) => setSceneForm({ lighting: v as "" | "nominal" | "degraded" })}
            options={[
              { value: "", label: "from the seed (25% degraded)" },
              { value: "nominal", label: "nominal" },
              { value: "degraded", label: "degraded: 25 to 45% brightness" },
            ]}
          />
        </Field>
        <Field label="Distractors" htmlFor="distractors" hint="Blank means whatever the seed chose (1 or 2).">
          <NumberInput id="distractors" value={form.n_distractors} onChange={(v) => setSceneForm({ n_distractors: v })} min={0} max={3} placeholder="from the seed" />
        </Field>
        <Toggle checked={form.fixed_goal} onChange={(v) => setSceneForm({ fixed_goal: v })} label="Fixed goal: red cube" hint="The 100 scenes ACT and the pipeline share" />
        <div className="flex gap-2">
          <Button full onClick={newScene} disabled={running}>
            New scene
          </Button>
          <Button full onClick={replay} disabled={running} title="Rebuild this seed and run it again (keyboard: R)">
            Replay
          </Button>
        </div>
      </div>

      <div>
        <h2 className="mb-2 text-sm font-semibold">Controller</h2>
        <div role="radiogroup" aria-label="Controller" className="space-y-1">
          {controllers.map((c) => (
            <div key={c.value}>
              <label className={`flex items-start gap-2 text-sm ${c.disabled ? "text-fg-muted" : ""}`}>
                <input
                  type="radio"
                  name="controller"
                  className="mt-1 accent-[color:var(--accent)]"
                  checked={controller === c.value}
                  disabled={c.disabled}
                  onChange={() => setState({ controller: c.value })}
                />
                <span>
                  {c.label}
                  {c.why && <span className="block text-xs text-fg-muted">{c.why}</span>}
                </span>
              </label>
            </div>
          ))}
        </div>
      </div>

      <div>
        <h2 className="mb-2 text-sm font-semibold">
          <Term term="ablation">Ablations</Term>
        </h2>
        <Toggle
          checked={config.use_color_check}
          onChange={(v) => setConfig({ use_color_check: v })}
          label={<Term term="colour check">HSV colour check</Term>}
          hint="Off: Grounding DINO ranking only"
        />
        <Toggle checked={config.use_yolo_mask} onChange={(v) => setConfig({ use_yolo_mask: v })} label="YOLO11-seg mask" hint="Off: the box becomes the mask, table pixels included" />
        <Toggle checked={config.depth_noise} onChange={(v) => setConfig({ depth_noise: v })} label="Synthetic depth noise" hint="Off: perfect rendered depth" />
        <Field label="Segmenter backend" htmlFor="backend">
          <Select
            id="backend"
            value={config.seg_backend}
            onChange={(v) => setConfig({ seg_backend: v })}
            options={SEG_BACKENDS.map((b) => ({
              value: b.value,
              label: present.get(b.file) === false ? `${b.label} (file missing)` : b.label,
              disabled: present.get(b.file) === false,
            }))}
          />
        </Field>
        <Field label="Speed" htmlFor="speed" hint="1x paces the executor to its real 10 Hz. Max removes the pacing; physics is identical either way.">
          <Select
            id="speed"
            value={String(config.speed)}
            onChange={(v) => setConfig({ speed: Number(v) })}
            options={[
              { value: "0.5", label: "0.5x" },
              { value: "1", label: "1x, real time" },
              { value: "2", label: "2x" },
              { value: "0", label: "as fast as the simulator runs" },
            ]}
          />
        </Field>
        <Field label="Safety mode" htmlFor="mode">
          <Select
            id="mode"
            value={config.safety_mode}
            onChange={(v) => setConfig({ safety_mode: v as "monitor" | "enforce" })}
            options={[
              { value: "monitor", label: "observe: report clips, keep the evaluated trajectory" },
              { value: "enforce", label: "enforce: apply clips (changes the motion)" },
            ]}
          />
        </Field>
        <Toggle
          checked={config.require_human_confirm}
          onChange={(v) => setConfig({ require_human_confirm: v })}
          label="Require human confirmation"
          hint="FMEA H6: nothing moves until an operator confirms the grounding"
        />
      </div>
    </div>
  );
}
