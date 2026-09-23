# LangGrasp GUI: design document (Phase 0)

Status: for review before any implementation. Nothing in this document changes pipeline behaviour or any
measured number; the GUI is an instrumented window onto the existing code.

## 1. Personas and jobs

| Persona | Job in one sentence | Success looks like |
|---|---|---|
| Recruiter or interviewer | Understand what the system does in 60 seconds, watch one live pick, see that the results are honest | Presses one example chip, watches the arm pick, reads the outcome card and the "simulation, not hardware" banner |
| Engineer (the author) | Find why a scene failed, stage by stage; replay a seed; compare methods and ablations | Opens the failing stage's drawer, sees candidates and scores, flips an ablation, reruns the same seed |
| Safety reviewer | See the monitor state and every gate decision; trigger and reset an e-stop; check FMEA coverage | Reads the right rail, presses E-STOP during a run and sees HOLD within 100 ms, finds each hazard's test |

## 2. User flows

Flow A, first live pick (recruiter):
1. Open http://localhost:8000 through the SSH port-forward. Live Run view loads with the last scene rendered and the banner.
2. Click the example chip "pick the blue screwdriver". The command bar fills; Enter runs it.
3. The stepper animates through the nine stages; the viewport shows the winning box, mask, grasp arrow, then the arm moving at 10 frames per second.
4. The outcome card shows grounding correct, grasp, place (labelled "scored against simulator ground truth").

Flow B, debug a failure (engineer):
1. Batch Evaluate view, run the seen stratum with 40 seeds. Grid fills; one cell is red.
2. Click the cell: Live Run opens with that seed, same command, same ablation settings.
3. Stage 4 (selection) is amber: the drawer shows two candidates with scores 0.61 and 0.58, colour fractions 0.9 and 0.1, ambiguity flag on.
4. Toggle "colour check" off, run again: the wrong object wins. Toggle on: the right one. The engineer has the story for the write-up.

Flow C, safety review:
1. Safety and System view: the FMEA table with status chips; each "tested" row links to its test.
2. Live Run: start a pick, press E during the descent. The rail shows E-STOP with the latency measured from click to hold; the arm freezes; Reset requires a click.

Flow D, speech (H6): press the mic, speak, release. The transcript appears in the command bar, editable, never executed automatically; Enter confirms.

## 3. Information architecture

Left navigation with five views, keyboard 1 to 5:

1. Live Run (default)
2. Pipeline Inspector
3. Results Dashboard
4. Batch Evaluate
5. Safety and System

Persistent chrome: top banner "Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware", theme toggle, worker status (models loading, warm, busy), shortcut help.

## 4. Wireframes

Live Run, at 1280 px and wider (three columns):

```
+------------------------------------------------------------------------------------+
| Simulation · NVIDIA L4 · x86 · not Jetson · not real hardware        [theme] [?]   |
+--------+-----------------------------------------------------+---------------------+
| nav    | [command ......................................] [mic] [Run]              |
| 1 Live | chips: pick the red cube | grab the blue one | ... on the left            |
| 2 Insp +-----------------------------------------------------+ SAFETY              |
| 3 Res  | scene: seed [5000] stratum [seen v] light [nominal] | state  RUN          |
| 4 Batch| distractors [2] [New scene] [Replay seed]           | camera 0.08 s ok    |
| 5 Safe | controller: (o) Pipeline ( ) Oracle ( ) ACT ( ) PPO | joints 0.02 s ok    |
|        | ablations: [x] colour [x] YOLO [x] noise  seg [TRT] | gate 0.81 >= 0.30   |
|        +-----------------------------------------------------+ ambiguity: no       |
|        | viewport  [front][wrist][side][depth][3D]           | clips this run: 3   |
|        | layers: [x]cands [x]box [x]mask [ ]cloud [x]grasp   |                     |
|        |                                                     |  [  E-STOP  ]       |
|        |         640 x 480 live stream with overlays        |  [ Reset ]          |
|        |                                                     |                     |
|        +-----------------------------------------------------+ FMEA quick view     |
|        | 1 cmd  2 parse  3 ground  4 select  5 seg  6 fuse   | H1 gate  H5 fence   |
|        |  ok 0   ok 0.1   ok 274    ok 1     ok 5   ok 3 ms  | H6 confirm H8 limit |
|        | 7 safety  8 exec  9 servos          [outcome card]  |                     |
+--------+-----------------------------------------------------+---------------------+
```

At 1024 px the safety rail becomes a collapsible drawer on the right edge with the state always visible as a chip in the banner; the controls column collapses into an accordion above the viewport.

Pipeline Inspector:

```
| run: 2026-09-23_1012_seed5000  [load run v]                                        |
| scrubber |----o--------------------------------------------| tick 23 / 61  [> ] |
| viewport (front frame at tick 23)   | joint angles vs targets (6 lines, 10 Hz)   |
|                                     | fingertip x,y,z over time                    |
| stage waterfall: parse 0.1 | capture 9 | grounding 274 | seg 5 | fuse 3 | exec 313 |
```

Results Dashboard: a grid of cards, each headed by the JSON file name and its modification time; a card with no file shows "not run".

## 5. Event schema (pydantic, `langgrasp/gui/trace.py`)

| Event | Fields | Emitted by |
|---|---|---|
| stage_started | run_id, stage (1 to 9), t_ms | pipeline hooks |
| stage_finished | run_id, stage, latency_ms, status (ok, warn, fail), payload (stage specific, JSON safe) | pipeline hooks |
| frame | run_id, camera, kind (rgb, depth, label), tick, jpeg (bytes, base64 on the socket) | worker |
| tick | run_id, t, q (6), q_target (6), tcp (3), jaw | executor hook |
| safety | run_id, state, reason, values (gate score, threshold, clips) | safety monitor hook |
| outcome | run_id, grounding_correct, grasped, lifted, placed, aborted, scored_by ("simulator ground truth") | worker |
| job_progress | job_id, done, total, last_result | batch runner |

Stage payloads: 2 parser fields; 3 all candidates (box, score); 4 colour fractions, spatial rule used, chosen box, ambiguity; 5 masks (run-length encoded) and the matched one; 6 grasp centre, yaw, width, point count; 7 gate decision; 8 waypoints; 9 per-tick data arrives as tick events.

Every run is appended to `runs/<timestamp>_<seed>.jsonl` (one event per line, frames stored as JPEG files next to it) so the Inspector can replay without the worker.

## 6. Instrumentation without behaviour change

`ModularPipeline`, `PickPlaceController` and `SafetyMonitor` get an optional `on_event` callback (default `None`). With no callback the code path is byte-identical. A new test runs 10 protocol seeds with the callback on and off and asserts identical outcome flags, grasp centres and latencies within noise; the existing 79 tests stay untouched.

## 7. API contract

| Method and path | Body or query | Returns |
|---|---|---|
| GET /api/system | | GPU, driver, package versions, engine files present, model load status, worker state |
| POST /api/scene | seed, stratum, lighting, distractors, fixed_goal | scene description, first frames |
| POST /api/run | command, controller, ablations, segmenter | run_id (events follow on the socket) |
| POST /api/estop | | state, measured latency to hold |
| POST /api/reset | | state |
| POST /api/confirm | run_id, accept | continues or aborts a run paused on ambiguity |
| POST /api/stt | multipart audio (webm or wav) | transcript, model, latency_ms; never executes |
| GET /api/results | | list of results files with mtime |
| GET /api/results/{file} | | the JSON, unmodified |
| POST /api/jobs | protocol subset, n per stratum, ablations, output name | job_id; refuses to overwrite an existing file unless `confirm=true` |
| GET /api/jobs/{id} | | progress and per-scene grid |
| WS /ws/live | | the event stream; frames coalesced so the newest wins |

## 8. Concurrency model

One worker process owns the MuJoCo environment, Grounding DINO, the segmenters, the ACT and PPO policies and the safety monitor. It reads commands from a multiprocessing queue and writes events to another. The API process never touches the models. Frames are pushed at 10 to 15 per second for the front camera; if the socket falls behind, older frames are dropped, never the simulation ticks. Batch jobs run in the same worker between live runs (a live run pre-empts at scene boundaries) so the models load once.

## 9. Design tokens and accessibility

CSS variables for colour (background, surface, text, accent, status ok/warn/fail/estop), 4 px spacing scale, radius 4/8/12, a 6-step type scale. Dark default, light theme, prefers-reduced-motion honoured (stepper animation becomes a fade). Status is never colour alone: every status chip has an icon and a word. All interactive elements reachable by keyboard with a visible focus ring; stage and safety changes announced through aria-live regions; contrast checked with axe-core in the e2e run.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Worker crash (CUDA error) takes the sim down | supervisor restarts the worker; UI shows "worker restarting"; runs are recorded so nothing is lost |
| Grounding latency (275 ms) feels like a freeze | explicit progress state on stage 3 with the measured typical time |
| Browser audio format vs faster-whisper | server converts with ffmpeg (imageio-ffmpeg already installed) |
| Frontend toolchain (Node) not on the VM | check; if absent, install Node 20 via nvm in the venv scope or fall back to a no-build setup (Preact + htm from a pinned CDN copy vendored into the repo) and say so |
| E-stop latency target under 100 ms | the e-stop sets a flag the worker checks every physics tick (2 ms), measured and logged per press |
| Numbers drifting from results/ | the dashboard reads files at request time; a re-run of the protocol from the UI writes a new file name unless overwrite is confirmed |

## 11. Out of scope for this build

Real hardware drivers, multi-user access, authentication (the port-forward is the access control), mobile layouts below 1024 px.
