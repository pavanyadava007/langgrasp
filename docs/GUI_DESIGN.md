# LangGrasp GUI: design document (Phase 0)

Status: draft for review, 2026-09-23. Nothing in this document has been built yet.

This is the design for a live web GUI on top of the existing LangGrasp stack. The stack is simulation only
(MuJoCo, SO-ARM100 model, NVIDIA L4 on an x86 cloud VM, no display). The GUI must let three kinds of people
understand and operate it without ever changing what the pipeline does or what the numbers say.

## 0. Ground rules the design is built around

| Rule | Consequence for the design |
|---|---|
| Pipeline behaviour and measured numbers do not change | Instrumentation is additive: optional hook object, optional tracer listener, the existing `record_fn` on the controller. The default (hooks off) code path is byte-for-byte what runs today. A test runs 10 seeds hooks-on vs hooks-off and asserts identical `CommandResult.to_dict()` output. |
| No hardcoded metric | Every number on screen is read from `results/*.json` at request time or arrives as a live event. A missing file or key renders the literal text "not run". Every widget shows its source file and the file's modification time. |
| Persistent banner | "Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware" is a fixed element in the app shell on every view, never dismissable. The GPU name in it comes from `/api/system`, not from a string constant. |
| Headless VM | FastAPI serves the built frontend and the API on one port; the user reaches it with `ssh -L 8000:localhost:8000`. No X display, no browser on the VM except headless Chromium for Playwright. |
| One owner of the sim and models | A single worker process owns `LangGraspEnv`, Grounding DINO, the segmenters, faster-whisper and the `SafetyMonitor`. The API process never imports MuJoCo or torch. Commands go in through one queue, events come out through another. |
| Ground truth never feeds the live pipeline | The label image and `env.grasp_point` are only read by explicitly named "ground truth" code paths: the outcome scorer (already how `_box_hits_object` works), the oracle controller (the user picks it on purpose), and the "ground truth" overlay layer, which is off by default and drawn in a distinct dashed style with a "GT" tag. |

Live latency shown in the GUI is real, but the GPU on this VM is shared with an unrelated Ollama server
(4.7 GB resident at the time of writing). Live numbers are therefore labelled "live, shared GPU" and are
never merged into the "clean bench" figures from `results/pipeline_latency_l4.json`.

## 1. Personas and their jobs

### P1 Recruiter or interviewer (NEURA Robotics)

Has 60 seconds, maybe five minutes. Does not know MuJoCo, Grounding DINO or Wilson intervals.
Wants to see: what the system does, one pick actually happening, whether the results are real.
Fails if: the page is a wall of controls, a number looks made up, or nothing moves.

Jobs: (a) read the one-line description and the banner, (b) press one obvious button and watch a pick with
the decision path visible, (c) glance at the results table and see confidence intervals and sample sizes.

### P2 Engineer (the author)

Knows the code. Wants to find out why seed 7012 failed, replay it with a different ablation, and compare
methods without editing scripts.

Jobs: (a) run any seed with any toggle, (b) step through a finished run tick by tick with joint angles,
fingertip trajectory and stage images, (c) launch a protocol subset and get the JSON in `results/`,
(d) jump from a red cell in a batch grid to that seed in Live Run.

### P3 Safety reviewer

Thinks in hazards, gates and states. Wants to see the monitor state at all times, what the gate decided and
why, which FMEA rows are implemented, tested or hardware-only, and to prove the e-stop works.

Jobs: (a) always-visible state RUN / REDUCED_SPEED / HOLD / ESTOP with the reason, (b) gate value vs
threshold for the current command, (c) press E-STOP and see the arm stop and the latency logged,
(d) read the FMEA table with links to the tests.

## 2. User flows

### F1 The 60-second demo (P1)

1. Open `http://localhost:8000`. The Live Run view loads with the front camera streaming an idle scene,
   the banner, and the stepper in "pending" state. Model status in the header reads "models warm".
2. Click an example chip, for instance "pick the blue screwdriver". The command bar fills.
3. Press Enter (or the Run button). Within one frame the stepper shows "Parse: ok", then "Grounding:
   running" with an indeterminate bar labelled "about 275 ms on this GPU (clean bench)".
4. Candidates appear as boxes on the video with scores and colour fractions. The winner turns solid.
5. The mask and the grasp marker appear. The arm moves in real time (paced at 10 Hz).
6. The outcome card shows "Grounding correct (ground truth), grasped, placed". Total about 8 s.
7. The user hovers "grounding correct" and reads the tooltip: scored against the simulator's label map,
   which the pipeline never sees.

### F2 Debug a failure (P2)

1. In Results, the per-object table shows screwdriver 39/41. Click the row: a list of the failing seeds.
2. Click seed 7012. Live Run loads that scene (stratum from the seed range) with its protocol command.
3. Press R (replay). Watch the stepper. Depth fusion shows "warn: 34 points after table removal".
4. Open the depth fusion drawer: input mask image, point count before and after the table filter, PCA
   axes, chosen grasp centre, the fallback flag.
5. Switch to Pipeline Inspector. Scrub to the "descend" phase. The fingertip trajectory plot shows the
   TCP 12 mm off the ground-truth centre (GT layer switched on by the user). Joint plot shows Wrist_Roll
   saturating.
6. Toggle "YOLO mask: off", replay. Compare: outcome card shows the box-only mask gave a wider grasp.

### F3 Prove the e-stop (P3)

1. Start a run. While the arm is in "transport", press E.
2. The safety rail flips to ESTOP (red, with the octagon icon and the word ESTOP), reason "operator
   e-stop (GUI)". The arm holds. The stepper marks Execute as "fail: estop".
3. The rail shows "e-stop latency: click to hold 41 ms (measured this run)". The value is measured from
   the browser click timestamp to the worker's first held tick, corrected for the WebSocket round trip.
4. Press Reset. A confirmation asks "Reset the latched e-stop? The arm can move again." Confirm.
   State returns to RUN. Nothing moves until a new command is issued.

### F4 Voice command (P1 and P2)

1. Click the microphone. Browser asks for permission. Recording indicator shows a level meter.
2. Click again to stop. The clip goes to `/api/stt`. The transcript appears in the command bar, editable,
   with "transcribed in 194 ms (faster-whisper base, this GPU)".
3. Nothing runs until the user presses Enter. This is the H6 mitigation: STT output is never auto-executed.

### F5 Batch evaluation (P2)

1. Batch Evaluate: pick "modular", strata seen + unseen, n = 10 per stratum, ablation "no colour check".
2. Output path is proposed as `results/modular_nocolor_protocol.json`. It exists, so the field turns
   amber: "exists (2026-09-23 10:14). Overwrite?" with the alternatives "write to
   results/gui/<timestamp>_modular_nocolor_protocol.json" (default) or "overwrite" (needs a second click).
3. Start. Progress bar, scene grid fills cell by cell (green placed, amber grasped only, red failed, grey
   pending). The Live Run view streams the scenes as they run, labelled "batch job running".
4. Click a red cell: Live Run opens on that seed with the same config, ready to replay.

## 3. Information architecture

Left navigation, five views, keyboard 1 to 5. The banner and the model status live in the top bar. The
safety rail is part of the Live Run and Batch views (both can move the arm).

```
[Top bar]  LangGrasp · Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware   [models: warm] [theme]
[Nav]      1 Live Run   2 Pipeline Inspector   3 Results   4 Batch Evaluate   5 Safety & System
```

| View | Primary user | Answers |
|---|---|---|
| Live Run | P1, P2, P3 | What does it do, right now, on this scene? |
| Pipeline Inspector | P2 | What exactly happened, tick by tick, in a finished run? |
| Results | P1, P2 | What was measured, with what confidence, from which file? |
| Batch Evaluate | P2 | Run the protocol or a subset, and get a file in results/. |
| Safety & System | P3 | Which hazards are covered, how, tested by what; what is loaded. |

Keyboard map (global, ignored while typing in a text field except Enter and Escape):

| Key | Action |
|---|---|
| Enter | Run the command in the bar (Live Run) |
| Space | Pause / resume the executor; when paused, Space steps one tick |
| E | E-stop (no confirmation, ever) |
| R | Replay the current seed with the current toggles |
| 1 to 5 | Switch view |
| Escape | Close drawer / cancel recording |
| ? | Show the shortcut sheet |

## 4. Wireframes

### 4.1 Live Run at 1280 px and wider (three columns)

```
+---------------------------------------------------------------------------------------------------+
| LangGrasp    Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware       models: warm  [☾]|
+-----+-------------------------------------------------------------------------------+-------------+
| 1   | COMMAND                                                                       | SAFETY      |
| 2   | [ pick the blue screwdriver                              ] [🎤] [Run ⏎]      | ● RUN       |
| 3   | chips: seen: "pick the red cube" · unseen: "pick the purple can" ·            | reason: ok  |
| 4   | langvar: "grab the blue one" · "pick the yellow cube on the left"             |             |
| 5   |                                                                               | watchdogs   |
|     | +-----------------------------------------------------------+ +-------------+ | camera 0.05s|
|-----| |                                                           | | wrist       | | joints 0.01s|
|SCENE| |               front camera  (live, 12 fps)                | +-------------+ | command 3s  |
|seed | |         [box 0.74 blue 0.91]  [box 0.71 blue 0.02]        | | side        | |             |
|[5000| |                 ▣ mask   ✚ grasp  ➜ yaw  |--| width       | +-------------+ | gate        |
|strat| |                                                           | | depth       | | 0.74 ≥ 0.30 |
|seen | |                                                           | +-------------+ | ambiguity:no|
|light| +-----------------------------------------------------------+                 | [Confirm]   |
|nomin| layers: [x] candidates [x] winner [x] mask [ ] points [x] grasp [ ] waypoints | [Reject]    |
|dist | [ ] geofence [ ] ground truth (GT)          view: [2D] [3D]   speed: [1x] [max]| (human-     |
|1    |                                                                               |  confirm off)|
|[New]| PIPELINE                                                                      |             |
|[Rep]| ①cmd ②parse ③capture ④ground ⑤select ⑥gate ⑦segment ⑧fuse ⑨execute            | clipping    |
|     | ok    ok     ok 9ms  ok 281  ok     ok     ok 12ms   ok 4ms  ▶ running        | limit 0     |
|CTRL | text  pick/  640x480 3 cand  winner  0.74≥  yolo:    xyz     phase: lift      | velocity 0  |
|(•)  | blue  blue   depth   fallbk  #1     0.30   screwdr  -0.06,  tick 31/62        |             |
|Pipe | screw screw  noise:  no      colour        iou .81  -0.21,                    | [ E-STOP ]  |
|( )  |       driver on                                     0.014                     | [ Reset ]   |
|Oracl|                                                                               | (latched)   |
|( )  | OUTCOME (scored against ground truth, which the pipeline never sees)          |             |
|ACT ⓘ| grounding: correct   grasp: lifted   place: pending                            | last e-stop |
|( )  |                                                                               | latency:    |
|PPO  |                                                                               | not run     |
|ABLAT|                                                                               |             |
|[x]col                                                                              |             |
|[x]yol|                                                                              |             |
|[x]noi|                                                                              |             |
|seg:  |                                                                              |             |
|[pt v]|                                                                              |             |
+-----+-------------------------------------------------------------------------------+-------------+
```

Left column (controls) is 280 px, right rail (safety) is 260 px, the centre takes the rest. The centre
scrolls; the two side columns are sticky. Stage cells are buttons; the active one has a visible focus ring
and an `aria-current`. Clicking a stage opens a drawer over the centre column with inputs, outputs and
images for that stage.

### 4.2 Live Run at 1024 px

The safety rail collapses into a fixed bar at the bottom of the viewport: state pill, reason, gate value,
E-STOP and Reset. The controls column becomes a collapsible panel behind a "Scene & controls" button in
the command bar. The stepper wraps to two rows. Nothing is hidden that concerns safety.

```
+-------------------------------------------------------------------+
| banner                                                             |
| [≡ scene & controls]  [ command ............ ] [🎤] [Run]           |
| +---------------------------------------------------------------+ |
| |                      front camera                              | |
| +---------------------------------------------------------------+ |
| ① ② ③ ④ ⑤        ⑥ ⑦ ⑧ ⑨                                          |
| outcome                                                            |
+-------------------------------------------------------------------+
| ● RUN  ok  gate 0.74 ≥ 0.30  [E-STOP] [Reset]                       |
+-------------------------------------------------------------------+
```

### 4.3 Stage detail drawer (example: grounding)

```
+--------------------------------------------------------------- [x] +
| ④ Grounding   ok   281 ms (live, shared GPU)   clean bench: 274 ms  |
| query sent: "blue screwdriver"   fallback to "object": no           |
| threshold box 0.25 / text 0.20                                      |
| candidates (3)                                                      |
|  # | label        | score | colour frac | area px | picked         |
|  1 | screwdriver  | 0.74  | 0.91        | 1310    | yes            |
|  2 | screwdriver  | 0.71  | 0.02        | 1288    | colour filtered|
|  3 | cube         | 0.33  | 0.00        |  980    | colour filtered|
| [image: input RGB with all three boxes]                             |
| why this one: colour filter kept 1 of 3; no spatial word; top score |
+---------------------------------------------------------------------+
```

### 4.4 Pipeline Inspector

```
+-----+------------------------------------------------------------------------------+
| run | runs/gui/2026-09-23T10-41-07_5000.jsonl   seed 5000 seen   "pick the blue cube"|
| list| outcome: placed   config: colour on, yolo on, noise on, seg pt, gate 0.30     |
|     +------------------------------------------------------------------------------+
| ... | [front frame at tick 31 with overlays]      | joints (rad) vs target          |
| ... |                                             | 5 lines + 5 dashed              |
| ... |                                             |---------------------------------|
|     |                                             | fingertip z, xy distance to GT  |
|     +---------------------------------------------+---------------------------------+
|     | ◀ ▶  [=========o-----------------]  tick 31 / 62   phase: lift   t = 3.1 s     |
|     +------------------------------------------------------------------------------+
|     | latency waterfall: parse|capture|grounding ████████|segment|fuse|execute ████ |
|     | stage images: [rgb] [candidates] [mask] [depth] [points]                      |
+-----+------------------------------------------------------------------------------+
```

### 4.5 Results dashboard

Each widget is a card with a title, the chart or table, and a footer "source: results/x.json ·
2026-09-23 10:14". A card whose file is missing shows the title and "not run" only.

```
+---------------------------------------------------------------------------------+
| Protocol (place rate, Wilson 95 %)         | Ablation deltas vs modular          |
| method × stratum, horizontal CI bars       | no colour / no yolo / no noise      |
| source: oracle_protocol.json, modular_*    | source: modular_*_protocol.json     |
+--------------------------------------------+-------------------------------------+
| Per object kind      | Per lighting        | Language variants                   |
+----------------------+---------------------+-------------------------------------+
| Fixed goal (red cube, 100 scenes): oracle / modular / ACT attempts 1..3          |
+---------------------------------------------------------------------------------+
| YOLO mAP + latency by backend            | Latency budget (stacked, clean bench)|
+------------------------------------------+--------------------------------------+
| PPO sim-to-sim gap (reach, lift)         | Depth fusion error table             |
+------------------------------------------+--------------------------------------+
| STT latency by model size                | ROS 2 smoke                          |
+---------------------------------------------------------------------------------+
```

### 4.6 Batch Evaluate

```
| method [modular v]  strata [x]seen [x]unseen [ ]langvar   n per stratum [10]      |
| ablation [none v]  segmenter [pt v]   gate threshold: 0.30 (from eval_modular.py) |
| output: results/modular_protocol.json  ⚠ exists (10:14)  (•) write timestamped copy|
|                                                          ( ) overwrite (confirm)  |
| [Start]                                                                            |
| progress ████████░░░░░░░ 12 / 20   placed 11   grounding ok 12   elapsed 1:34      |
| grid: seen  [■■■■■■■■■□]  unseen [■■□□□□□□□□]   ■ placed ▨ grasped ▩ failed □ pend |
| running jobs: ...  finished jobs: ... (each links to its JSON and to Results)      |
```

### 4.7 Safety & System

```
| FMEA H1..H8                                                                        |
| H | hazard | standard | mitigation | code | test | status                           |
| H1| wrong object grounding | SOTIF | conf gate + colour check | watchdog.py:gate_grounding | test_safety.py::test_low_grounding_confidence_not_allowed | tested (sim) |
| ...                                                                                |
| status legend: implemented · tested (sim) · hardware only (not exercised)          |
| System: GPU, driver, CUDA, torch, ultralytics, tensorrt, lerobot, mujoco, python   |
| Models: grounding dino tiny (warm, 1.9 s load), yolo pt/onnx/fp16/int8 (present?), |
|         faster-whisper base (warm), ACT checkpoints, PPO checkpoints               |
| Engines: checkpoints/*.engine with size and mtime; missing ones say "not built"    |
```

The FMEA table is generated from a small YAML (`langgrasp/gui/fmea.yaml`) that mirrors `docs/FMEA.md`
row by row. A unit test checks every test name in the YAML exists in the collected pytest node ids, and
every code location resolves to a real function.

## 5. Visual design

### 5.1 Tokens

CSS variables on `:root`, redefined under `[data-theme="light"]`. Tailwind reads them through its config
so utility classes stay semantic (`bg-surface`, `text-fg-muted`, `border-edge`).

```
--space-1: 4px; --space-2: 8px; --space-3: 12px; --space-4: 16px; --space-6: 24px; --space-8: 32px
--radius-s: 4px; --radius-m: 8px; --radius-l: 12px
--font-sans: Inter, system-ui; --font-mono: "JetBrains Mono", ui-monospace
--text-xs: 12px; --text-s: 13px; --text-m: 14px; --text-l: 16px; --text-xl: 20px; --text-2xl: 28px

dark (default)                          light
--bg:        #0f1216                    #f6f7f9
--surface:   #171b21                    #ffffff
--surface-2: #1f242c                    #eef0f3
--edge:      #2c333d                    #d5d9e0
--fg:        #e6e9ee                    #16191e
--fg-muted:  #9aa3b2                    #5b6472
--accent:    #4c8dff                    #2d6be4
--focus:     #ffd166                    #b5560a
status (each also has an icon and a label, never colour alone)
--ok:        #3ddc97  (check icon)      #1a8a5c
--warn:      #f5b942  (triangle icon)   #a86a00
--fail:      #ff5c5c  (x icon)          #c62828
--estop:     #ff2e2e  (octagon icon)    #b71c1c
--running:   #4c8dff  (spinner)         #2d6be4
--pending:   #6b7482  (circle icon)     #8a93a1
--gt:        #c084fc dashed (tag "GT")  #7c3aed
```

All foreground/background pairs above were chosen to meet WCAG 2.2 AA (4.5:1 for text, 3:1 for large
text and UI parts); the accessibility pass in Phase 6 verifies them with axe-core and fixes any that miss.

### 5.2 Status vocabulary

| State | Colour token | Icon | Label text |
|---|---|---|---|
| pending | pending | empty circle | "pending" |
| running | running | spinner (static ring under reduced motion) | "running" |
| ok | ok | check | "ok" |
| warn | warn | triangle | "warn: <reason>" |
| fail | fail | x | "fail: <reason>" |
| RUN | ok | filled circle | "RUN" |
| REDUCED_SPEED | warn | half circle | "REDUCED SPEED" |
| HOLD | warn | pause | "HOLD: <reason>" |
| ESTOP | estop | octagon | "E-STOP: <reason>" |

### 5.3 Accessibility and motion

- Every interactive element is a real `<button>`, `<input>`, `<select>` or has the equivalent role.
- Focus ring: 2 px `--focus` outline with 2 px offset, on every focusable element, never removed.
- `aria-live="polite"` region announces stage transitions ("Grounding finished, 3 candidates").
  `aria-live="assertive"` region announces safety state changes and e-stop.
- The video canvas has an `aria-label` describing the current overlay state in words.
- `prefers-reduced-motion`: spinners become static rings, the stepper does not animate, the 3D view does
  not auto-rotate, frame updates still happen (that is content, not decoration).
- Tooltips are keyboard reachable (focus opens them) and use `aria-describedby`. Glossary terms:
  mAP50-95, Wilson interval, SOTIF, FMEA, grounding, ablation, stratum, sim-to-sim gap, geofence,
  watchdog, TCP. Definitions are copied from the existing docs, not rewritten.
- Lighthouse accessibility target 95 or more, measured in Phase 6 with headless Chromium and reported
  as measured.

### 5.4 Latency-aware feedback

- Sending a command marks stages 1 to 3 "running" optimistically; the server's `stage_started` events
  correct the state if they differ.
- Grounding shows an indeterminate bar with the caption "typically 274 ms on this GPU when idle (clean
  bench, results/pipeline_latency_l4.json)"; the caption is built from the file, so if the file is
  missing it says "typical latency: not run".
- Results cards render skeletons until their JSON arrives.
- The WebSocket client keeps only the newest frame per camera; render happens on `requestAnimationFrame`.
  The sim never waits for a client.

### 5.5 Microcopy for errors (examples, all generated from event payloads)

- "Grounding found no candidate above 0.25 for 'blue driver'. Fell back to 'object' + colour filter."
- "Gate refused: confidence 0.27 is below the threshold 0.30. Nothing moved."
- "Gate refused: ambiguous, top two scores 0.61 and 0.58 are within 0.05. Add a spatial word such as
  'on the left'."
- "Depth fusion: 14 points left after table removal (minimum 20). No grasp. Try the YOLO mask on."
- "Grasp width 0.071 m exceeds the jaw maximum 0.060 m. No grasp."
- "E-stop latched by operator. Press Reset to allow motion."
- "ACT is disabled: it was trained on a fixed goal (red cube to tray) and takes no language input.
  Select the command 'pick the red cube' to enable it."

## 6. Architecture

### 6.1 Processes

```
browser  <-- HTTP + WebSocket -->  API process (FastAPI, uvicorn, asyncio)
                                        |  cmd_q: multiprocessing.Queue   (small dicts)
                                        |  evt_q: multiprocessing.Queue   (events; frames as JPEG bytes)
                                        |  ctrl: multiprocessing shared flags (estop, pause, step)
                                        v
                                   worker process (spawn) owns: LangGraspEnv, GroundingDINO,
                                   Segmenter per backend, SpeechToText, SafetyMonitor, ModularPipeline
```

- The worker is started with the `spawn` start method (the default on this machine is `fork`, which is
  unsafe once CUDA is initialised; the API process never touches CUDA anyway, but spawn is the safe rule).
- E-stop, pause and single-step are `multiprocessing.Value` flags, not queue messages, so the worker sees
  them on its next check without draining the queue. The check runs in the guarded step (every tick) and
  inside the pacing wait, so the worst case between click and reaction is one tick of compute (about 5 ms)
  plus the process wake-up.
- The worker emits at most one front frame per tick during motion (10 Hz when paced at real time) and one
  every 200 ms when idle. If `evt_q` holds more than 30 items the worker drops frame events (never stage or
  safety events). The API keeps a per-client "latest frame" slot and sends it on the next drain, so a slow
  client sees fewer frames, not a growing backlog.
- Models load once at worker start with warm-up; `/api/system` reports each model's load time and state
  (loading / warm / missing / error) so the header can say "models: loading Grounding DINO" honestly.

### 6.2 Non-invasive instrumentation (Phase 1 detail)

Attachment points, all additive and default-off:

| Point | Change | Behaviour when unused |
|---|---|---|
| `LatencyTracer(listener=None)` | `stage()` calls `listener("stage_started"/"stage_finished", name, ms)` if set | identical: same `samples` dict |
| `ModularPipeline(..., hooks=None)` | at each existing stage boundary in `run_command`, `perceive`, `ground`, `mask_for`, `grasp_from`: `if self.hooks: self.hooks.emit(...)` with the stage payload (intent dict, candidates, select info, mask source, grasp dict) | identical: no calls, no extra allocations |
| `PickPlaceController(record=True, record_fn=...)` | already exists; the worker passes a `record_fn` that emits `tick` events, paces to real time, and checks e-stop/pause | already the documented contract; `ModularPipeline.run_command` gets an optional `controller_factory` so the worker can build the controller with `record_fn` |
| `env.step` wrapper (worker only) | the worker wraps `env.step` exactly as `tests/test_safety.py::test_sim_pick_through_safety_monitor` does: heartbeat + `check_tcp` + `check_joint_command` around the original step | not part of the library; nothing changes for scripts |

The identical-results test: for 10 seeds across the three strata, run `ModularPipeline` with the
`OracleGrounder` (no GPU needed in CI) hooks-off and hooks-on with a recording hook, and assert
`to_dict()` equality including latency keys (values excluded), grasp centre, box, flags. A second, slower
test does the same for 3 seeds with the real Grounding DINO and is skipped when CUDA is absent.

### 6.3 Safety monitor in the live loop: monitor vs enforce

The evaluation scripts do not put `check_joint_command` in the loop (only the gate). The ROS 2 safety
node does. Applying the velocity clip in the GUI could change trajectories versus the reported protocol.
The design therefore has two modes, shown in the safety rail:

- **monitor** (default): the monitor observes every tick (heartbeats, `check_tcp`, `check_joint_command`
  computed) and the GUI shows what it would have clipped, but the arm receives the controller's target
  unchanged. HOLD and ESTOP are still enforced: in HOLD the wrapped step holds the current pose and blocks
  the controller until released; in ESTOP it holds and raises `MotionAborted`, which the worker catches
  and reports as `aborted = "safety:estop"`. This keeps the motion identical to the evaluation while the
  state machine stays authoritative.
- **enforce**: the clipped target is what the arm receives, as in the ROS 2 node. Labelled in the rail and
  in the run record; outcomes from enforce mode are never written to the protocol result files.

Measured in Phase 1 with `scripts/audit_safety_clips.py` over the 120 protocol scenes, oracle grasp poses,
results in `results/safety_clip_audit.json`:

| Quantity | Measured |
|---|---|
| Peak per-joint speed the scripted executor commands at 10 Hz | 7.8 rad/s median per trial, 21.8 rad/s worst |
| Monitor's default limit | 1.5 rad/s |
| Ticks the monitor would clip | 35.3% of 7186 ticks; every one of the 120 trials has at least one |
| Joint-limit clips | 0 |
| Placed, observing (clip reported, controller's target applied) | 120/120, the same as the oracle row in `docs/RESULTS.md` |
| Placed, enforcing (clipped target applied) | 73/120 |

So enforcing is not free: it is a different trajectory and a worse one. Monitor mode is the default and is
what the GUI uses for anything a person might compare against `results/`. Enforce mode stays available,
labelled, for showing what a rate-limited arm does. The peak comes from the first waypoint of each Cartesian
move, where the straight-line interpolation in task space implies a large joint step; the ROS 2 executor,
which re-solves from the measured pose every tick, commands smaller deltas, which is why the Docker run
recorded 51 clips and still succeeded.

One documentation defect found by this measurement: the comment next to `max_joint_vel` in
`langgrasp/safety/watchdog.py` said the scripted controller peaks well below 1.5 rad/s. It does not. The
comment now carries the measured numbers and points at the audit. No constant and no behaviour changed.

### 6.4 Human confirmation and ambiguity

`SafetyMonitor.gate_grounding` is called by the pipeline unchanged. In the worker the monitor is a thin
subclass whose `gate_grounding` first emits a `gate_request` event with the winner box, score, top-2
scores and threshold, and, when `require_human_confirm` is on, waits (up to 60 s) for `/api/confirm`
before delegating to the parent. Reject or timeout returns `(False, "rejected by operator")`. The
pipeline's own abort path handles the rest. Ambiguity (top-2 margin) stays a hard refusal exactly as
documented in FMEA H6; the GUI explains it and suggests a spatial word. Whether a human should be able to
override an ambiguous result is an open question (section 10), not a design decision taken here.

### 6.4a What the watchdogs measure in an in-process simulation

`SafetyConfig.staleness_s` gives the camera 0.5 s and the joint bus 0.2 s. On hardware those topics have
their own publishers. Here the worker samples both synchronously, and while it is inside a 275 ms grounding
call it samples nothing, so a naive reading would latch an e-stop on every command. The worker therefore
records a sensor sample at each point where it genuinely reads the simulator: every capture stage, every
frame it publishes, every idle poll and immediately before motion starts. The watchdogs then measure the
worker's own sampling gaps, which is the useful thing to measure here (they still fire if the worker stalls),
and the Safety view says so in those words rather than implying a sensor-fault detector that this
architecture cannot have.

Two consequences visible in the GUI: the worker starts in HOLD with the reason "no command yet (30 s command
watchdog)", which the first command clears, and the watchdog ages shown in the rail are ages of samples, not
of frames from an independent camera node.

### 6.4b Live runs are not bit-reproducible, and the GUI must not claim they are

Measured while writing the Phase 1 tests: two identical uninstrumented runs of Grounding DINO tiny on this
GPU return boxes that differ by about 0.002 px and scores by about 5e-4, because cuDNN may pick different
kernels. Decisions (chosen candidate, colour filtering, mask source, grasp pose) were identical across runs.
The identity test therefore demands identical decisions and a numeric wobble no larger than the GPU's own,
rather than bit equality, and the UI never claims a live run reproduces a protocol number exactly. The
oracle-grounder path is bit-exact and is the strict test.

### 6.5 Pacing and what "execute latency" means live

The executor runs about 60 ticks in about 340 ms of compute. Live, the worker paces ticks to 100 ms of
wall time so a person can watch. Pacing happens in the `record_fn` hook before `env.step`, and only
sleeps; physics and IK are untouched, so outcomes are identical to the unpaced run (asserted by the
identical-results test with pacing enabled at speed "max"). The stepper reports execute as two numbers:
"compute 338 ms" (sum of tick compute) and "wall 6.2 s (paced 1x)". Speed "max" removes the sleep.

### 6.6 Controllers

| Controller | How it runs in the worker | Constraints shown in the UI |
|---|---|---|
| Pipeline | `ModularPipeline.run_command` with the selected `PipelineConfig` and segmenter | ablation toggles apply |
| Oracle | `env.grasp_point` + `PickPlaceController`, as `scripts/eval_oracle.py` | marked "uses ground-truth pose (no perception)"; stages 4 to 8 show "skipped (oracle)" |
| ACT | `ACTRunner` closed loop as in `scripts/eval_act.py` (checkpoint selector: act_pick, act_pick_192, act_pick_v3) | enabled only when the command parses to colour red + noun cube and the stratum is the fixed-goal scene; otherwise disabled with the explanation from 5.5 |
| PPO reach / lift | `VecArmEnv(n_envs=1)` rollouts from `scripts/eval_ppo.py`; state based, no camera in the loop | its own reduced scene (arm + cube_0); rendering uses a separate `LangGraspEnv`-style renderer of the RL model; labelled "PPO uses its own state-based env (docs/RL.md); nominal or shifted dynamics selectable" |

### 6.7 Run recording

Every live run writes `runs/gui/<ISO timestamp>_<seed>/`:

```
events.jsonl      one JSON event per line (schema below), frames referenced by relative path
frames/           tick_000031_front.jpg ... and stage images (capture_rgb.jpg, candidates.jpg,
                  mask.png, depth.png, points.png)
meta.json         seed, stratum, command, controller, config, git commit, hardware label, mode
```

`runs/` is already gitignored (ACT training logs live in `runs/act`); the GUI uses the `runs/gui`
subfolder. The Inspector lists these directories through `/api/runs`.

## 7. Event schema (pydantic, `langgrasp/gui/trace.py`)

```python
class Event(BaseModel):
    t: float                 # seconds since worker start, monotonic
    run_id: str | None       # None for idle frames and system events
    type: Literal["stage_started", "stage_finished", "frame", "safety", "tick", "outcome",
                  "gate_request", "log", "job_progress", "system"]

class StageStarted(Event):   stage: StageName
class StageFinished(Event):  stage: StageName; status: Literal["ok","warn","fail","skipped"]
                             latency_ms: float; latency_kind: Literal["compute","wall_paced"]
                             payload: dict          # stage specific, see table below
                             message: str | None    # human sentence for warn/fail
class Frame(Event):          camera: Literal["front","wrist","side"]; kind: Literal["rgb","depth"]
                             tick: int | None; jpeg: bytes  (sent as a binary WS message with a 32-byte header)
class Safety(Event):         state: Literal["RUN","REDUCED_SPEED","HOLD","ESTOP"]; reason: str
                             gate: dict | None      # {score, threshold, top2, ambiguous, decision}
                             watchdog: dict         # {camera: age_s, joint_states: age_s, command: age_s}
                             clips: dict            # {limit: n, velocity: n}
                             mode: Literal["monitor","enforce"]
                             estop_latency_ms: float | None
class Tick(Event):           tick: int; phase: str; q: list[float]; q_target: list[float]; jaw: float
                             tcp: list[float]; bodies: dict[str, list[float]] | None  # name -> [x,y,z,qw,qx,qy,qz]
class Outcome(Event):        grounding_correct: bool | None; grasped: bool; lifted: bool; placed: bool
                             aborted: str | None; scored_by: Literal["ground_truth"]; message: str
class GateRequest(Event):    box: list[float]; score: float; top2: list[float]; threshold: float
                             ambiguous: bool; needs_confirmation: bool
class JobProgress(Event):    job_id: str; done: int; total: int; last: dict; out_path: str
class System(Event):         models: dict[str, {"state": str, "load_ms": float | None}]; gpu: str
```

Stage names and payloads (the 9 stages):

| # | stage | code | payload keys |
|---|---|---|---|
| 1 | command | GUI (text or STT) | text, source: typed / stt, stt_latency_ms |
| 2 | parse | `parser.parse_command` | intent dict (action, color, noun, spatial, phrase, generic, notes) |
| 3 | capture | `ModularPipeline.perceive` | camera, size, depth_noise, K, images: rgb, depth |
| 4 | grounding | `ground_with_fallback` | query, fallback, candidates [{box, score, label, color_frac, area}], thresholds |
| 5 | select | `select_target` | n_candidates, color_filtered, spatial_used, ambiguous, winner index, rule used |
| 6 | gate | `SafetyMonitor.gate_grounding` | score, threshold, top2, decision, reason, human_confirm |
| 7 | segment | `mask_for` | mask_source (oracle / yolo:<name> / box), iou, backend, timing dict, image: mask |
| 8 | fuse | `grasp_from` | n_points_raw, n_points_object, center, psi, width, long_axis, aspect, elongated, fallback, image: points |
| 9 | execute | `PickPlaceController.run` | phases with tick ranges, steps, ik_ok, max_pos_err, rot_err, compute_ms, wall_ms, paced |

Oracle runs mark 4 to 8 as `skipped` with the message "oracle controller: ground-truth grasp point".

## 8. API contract (`langgrasp/gui/api.py`)

All bodies and responses are JSON unless stated. Errors are `{"error": {"code": str, "message": str}}`
with 4xx/5xx status. The message is the same sentence the UI shows.

| Method, path | Request | Response |
|---|---|---|
| GET `/api/system` | | gpu, driver, cuda, python, versions{torch, mujoco, ultralytics, tensorrt, lerobot, transformers, faster_whisper}, models{name: state, load_ms, path}, engines[{path, present, size_mb, mtime}], worker{pid, alive, mode}, hardware_label |
| GET `/api/scene` | | current scenario dict (`Scenario.to_dict()`) plus seed, stratum, n_distractors, lighting, command, example_commands per stratum |
| POST `/api/scene` | {seed, stratum, lighting?, n_distractors?, fixed_goal?} | the new scenario; worker resets the env and streams a frame |
| POST `/api/run` | {command, controller: pipeline/oracle/act/ppo_reach/ppo_lift, config:{use_color_check, use_yolo_mask, depth_noise, seg_backend, grounding_threshold, require_human_confirm, safety_mode, speed}, act_checkpoint?, ppo_condition?} | {run_id} (202); 409 if a run or job is active; 423 if ESTOP is latched |
| POST `/api/estop` | {client_ts_ms} | {state, worker_ts, estop_latency_ms} |
| POST `/api/reset` | {confirm: true} | {state} |
| POST `/api/pause` | {paused: bool} or {step: true} | {paused, tick} |
| POST `/api/confirm` | {run_id, decision: confirm/reject} | {accepted} |
| POST `/api/stt` | multipart audio (webm/opus or wav), model_size? | {text, normalized, language, latency_ms, model_size, device} |
| GET `/api/results` | | [{file, size, mtime, approach, n_trials}] for every results/*.json that parses |
| GET `/api/results/{file}` | | the JSON file verbatim, plus header `X-Source-Mtime` |
| GET `/api/results/derived/protocol` | | the same tables the report builds (calls `langgrasp.eval.report` functions, not a copy) with file and mtime per row |
| GET `/api/runs` | | recorded live runs [{id, seed, command, outcome, mtime}] |
| GET `/api/runs/{id}` | | meta + events (frames by URL under `/runs/{id}/frames/...`) |
| POST `/api/jobs` | {kind: protocol, controller, strata, n_per_stratum, config, out_path, overwrite: bool} | {job_id} (202); 409 if the file exists and overwrite is false |
| GET `/api/jobs`, GET `/api/jobs/{id}` | | status, progress, per-trial results so far, out_path |
| DELETE `/api/jobs/{id}` | | cancels (the worker stops after the current trial) |
| GET `/api/fmea` | | rows from `langgrasp/gui/fmea.yaml` with resolved test node ids and code links |
| WS `/ws/live` | client sends {subscribe: [cameras], depth: bool, bodies: bool} | server sends JSON events as text frames and JPEGs as binary frames: `b"LGF1"` + `uint32` metadata length + metadata JSON + JPEG bytes (self describing, one message per frame; `langgrasp/gui/trace.py: pack_frame`) |

Static: `/` serves `langgrasp/gui/static/` (the Vite build); `/assets/meshes/*.stl` serves the arm meshes
for the 3D view; `/runs/...` serves recorded frames.

## 9. Frontend structure

```
gui/                      Vite + React 18 + TypeScript
  src/app/                shell, nav, banner, theme, shortcuts, aria-live regions
  src/tokens.css          design tokens (section 5.1)
  src/store/              zustand slices: connection, scene, run (stages, events), safety, results, jobs, ui
  src/ws/                 WebSocket client, binary frame decoder, latest-frame slots
  src/views/LiveRun/      CommandBar, SceneControls, ControllerSelect, Ablations, Viewport (canvas overlays),
                          Viewport3D (react-three-fiber, STL meshes, body poses from tick events),
                          Stepper, StageDrawer, OutcomeCard, SafetyRail
  src/views/Inspector/    RunList, Scrubber, JointChart, TrajectoryChart, LatencyWaterfall, StageImages
  src/views/Results/      one component per card, all fed by useResultsFile(name)
  src/views/Batch/        JobForm, JobProgress, SceneGrid
  src/views/Safety/       FmeaTable, SystemInfo
  src/components/         Button, Toggle, Select, Tooltip (glossary), StatusPill, Skeleton, Card
  e2e/                    Playwright specs
```

Build output goes to `langgrasp/gui/static/` and is committed so `make gui` works without Node on a fresh
checkout (Node is only needed to rebuild). This is a choice to confirm (section 10).

## 10. Risks and open questions

| # | Risk or question | Proposal |
|---|---|---|
| R1 | Port 8000 is already bound on this VM by another user's process (also 8001 and 8080). `make gui` on :8000 will fail here. | `make gui` reads `GUI_PORT` (default 8000) and fails fast with the message "port 8000 in use, try GUI_PORT=8010 make gui". The acceptance criterion stays ":8000 by default". Confirm this is acceptable. |
| R2 | The GPU is shared with an Ollama server. Live latencies will be worse than the clean bench and vary. | Every live latency is labelled "live, shared GPU"; the clean bench number sits next to it with its file. No live number is ever written into results/. |
| R3 | Velocity clipping in enforce mode could change trajectories. | Measured in Phase 1 and settled: it changes them a lot. See section 6.3. Monitor mode is the default. |
| R4 | Human override of an ambiguous grounding is not in the current safety design (FMEA H6 says refuse). | Not implemented unless you say so. Confirm/Reject applies to `require_human_confirm`. |
| R5 | Adding optional kwargs (`hooks`, `controller_factory`, `listener`) to `ModularPipeline`, `LatencyTracer` is an additive API change to existing modules. | Asking for approval here. Alternative is subclassing with a duplicated `run_command`, which I advise against. |
| R6 | Batch jobs run inside the worker, so the live view is busy during a job. | Accepted: the Live Run view shows "batch job running" and streams the job's scenes; E-STOP cancels the job. Running a second model set in another process would double GPU memory and violate the single-owner rule. |
| R7 | Disk is at 93 % (27 GB free). node_modules plus Playwright's Chromium need about 700 MB. | Fine, but noted. I will not install browsers for Firefox/WebKit. |
| R8 | Heavy new dependencies: fastapi, uvicorn[standard], python-multipart, pydantic v2 (check compatibility with lerobot's pins), httpx and pytest-asyncio for tests; frontend react, three, @react-three/fiber, @react-three/drei, recharts, zustand, tailwind, vite, playwright, axe-core. | These are the ones in your architecture list. I will report exact versions after install and stop if pydantic v2 conflicts with the venv. |
| R9 | ACT and PPO controllers need their own preprocessing and, for PPO, a separate reduced model; rendering the PPO scene needs a renderer on that model. | ACT in Phase 3 (fixed goal only). PPO live rollout in Phase 3 if time allows, else Phase 5, always with the "own env" label. |
| R10 | The STT mic flow needs `getUserMedia`, which browsers allow on `http://localhost` (a secure context), so the SSH tunnel works; it would not work on a plain LAN IP. | Documented in DEMO_GUIDE. |
| R11 | E-stop latency measurement across processes needs a common clock. | The worker and API both use `time.monotonic()` in the same machine; the API stamps receipt, the worker stamps the first held tick; browser-to-server time is measured separately by the WS ping and reported as a second number. |
| R12 | Committing the built frontend (`langgrasp/gui/static/`, about 1 to 2 MB) versus building at `make gui`. | Proposal: commit the build so the demo works without Node. Confirm. |
| R13 | 3D view fidelity: MuJoCo body poses are exact, but the STL meshes are visual only; collision meshes are not shown. | Labelled "visual meshes". Optional layer, off by default. |

## 11. Phase plan and deliverables

| Phase | Deliverable | Verification |
|---|---|---|
| 0 | this document | your review |
| 1 | done: `langgrasp/gui/trace.py`, `langgrasp/gui/worker.py`, additive hooks in `modular.py` and `latency.py`, `scripts/audit_safety_clips.py`; `tests/test_gui_trace.py`, `tests/test_gui_hooks_identical.py` (10 seeds oracle grounder bit-exact, 3 seeds real Grounding DINO by decision), `tests/test_gui_worker.py` (spawned worker, nine stages, e-stop, reset, monitor mode) | `make gate`: 79 existing plus 22 new tests pass; e-stop measured at 10.4 ms |
| 2 | `api.py`, WebSocket, results and runs endpoints, jobs; `tests/test_gui_api.py` with a stub worker (httpx ASGI client) | `make gate` |
| 3 | frontend shell, tokens, Live Run (stream, overlays, stepper, drawer, outcome, safety rail, e-stop, controllers, ablations) | manual on the tunnel + screenshots; e-stop latency logged |
| 4 | Inspector + replay from `runs/gui` | manual + unit tests for the scrubber reducer |
| 5 | Results (all cards from JSON), Batch Evaluate | unit tests for the JSON-to-widget mappers with a missing-file case ("not run") |
| 6 | Safety & System, STT mic flow, axe-core pass, responsive pass | Lighthouse accessibility score reported as measured |
| 7 | Playwright e2e, `make gui`, `make gui-test`, `make gate` extension, README and DEMO_GUIDE updates, screenshots | all green, numbers in results/ unchanged (diff of the files) |

Each phase ends with one commit and a short report with measured numbers and anything that could not run
on this VM.
