# LangGrasp failure-mode analysis (ISO 26262 / ISO 21448 style)

Scope: the language-guided pick-and-place stack in this repository, running on a simulated SO-101 arm in
MuJoCo, with the safety monitor in `langgrasp/safety/watchdog.py` in the loop both in-process and as the ROS 2
`safety_node`. This is a portfolio-scale analysis by one engineer, not a certified work product: no ASIL/PL
determination, no hardware qualification, no independent assessment. Where a mitigation needs a real sensor or
a real servo it is marked **planned, hardware only** and nothing here claims it was exercised.

## 1. Framing: which standard covers what

| Standard | Question it answers | What it means for this project |
|---|---|---|
| ISO 26262 (functional safety, automotive; the cobot analogue is ISO 10218 / ISO 13849 / IEC 62061) | "What if a component **fails**?" Random hardware faults and systematic software faults in E/E systems. | Sensor dropout, stale topics, servo faults, controller crashes, joint limit violations. Mitigations are watchdogs, plausibility checks, safe states (HOLD / ESTOP), latching. Rows H3, H4, part of H2 and H8. |
| ISO 21448 SOTIF (safety of the intended functionality) | "What if nothing fails but the function is still **insufficient** for the situation?" Performance limitations and triggering conditions of perception / planning, including ML. | Wrong-object grounding, occlusion, ambiguous language, out-of-distribution policy inputs. Mitigations are confidence gating, ambiguity detection, fallback to a scripted pipeline, human confirmation, and scenario coverage in evaluation. Rows H1, H2, H6, H7. |
| ISO/PAS 8800 (safety and AI in road vehicles, 2024) | How to argue the safety of an AI/ML component specifically: data quality, OOD detection, uncertainty, monitoring, lifecycle. | Applies to the grounding model, the YOLO segmenter and the ACT / PPO policies. This repo implements the runtime side (confidence thresholds, ambiguity margin, policy fallback, stratified seen / unseen / langvar evaluation) but no data-lifecycle argument. |

The S/E/C scale below is qualitative and follows the 26262 vocabulary, applied to a desktop cobot:

* **S (severity)** S0 none, S1 light (pinch, dropped object), S2 moderate injury or fixture damage, S3 severe.
  The SO-101 is a low-torque hobby-class arm (STS3215 servos, about 0.5 kg payload); S2 is the ceiling in
  practice and is reserved for a human hand in the workspace or a fixture / camera strike.
* **E (exposure)** E1 rare, E2 occasional, E3 frequent, E4 most operating time.
* **C (controllability)** C1 easily controllable by a bystander (stop button reachable, slow motion),
  C2 normally controllable, C3 hard to control.

## 2. Hazard table H1..H8

Each row: triggering condition, hazardous behaviour, S/E/C, mitigation, where it lives in this repo, and the
test that exercises it. "Sim" means it is exercised in MuJoCo here; "hardware only" means it needs a physical
sensor or actuator and is not exercised.

### H1 Wrong-object grounding (SOTIF)

* **Triggering condition**: a distractor that shares colour or shape with the target; an unseen colour / kind
  (the `unseen` stratum); degraded lighting (`scenario.lighting == "degraded"`); a phrase the open-vocabulary
  detector maps to the wrong box.
* **Hazardous behaviour**: the arm picks and moves the wrong object, potentially a fragile or forbidden one.
* **S1 / E3 / C2** (frequent with distractors, bystander can stop, low severity).
* **Mitigation**: confidence threshold on the grounding score; colour-fraction plausibility check on the
  chosen box; the grounding result is gated before any motion is commanded.
* **Where**: `langgrasp/safety/watchdog.py: SafetyMonitor.gate_grounding` (threshold
  `SafetyConfig.grounding_conf_threshold = 0.35`); ROS: `safety_node.on_grounding` holds the arm until a
  passing result arrives. Perception side (other track): `langgrasp/perception/grounding.py: select_target`
  and `color_fraction`.
* **Test**: `tests/test_safety.py::test_low_grounding_confidence_not_allowed`;
  `tests/test_ros2_msgs.py::test_lexical_scores_and_oracle_ground` (gate blocks a zero-score fallback).
* **Status**: sim. The residual risk (a confident but wrong detection) is only reduced by the evaluation
  strata in `langgrasp/eval`, not eliminated.

### H2 Occlusion / missed detection (SOTIF, plus 26262 for a dead camera)

* **Triggering condition**: the arm or another object occludes the target in the front camera; the target is
  outside the field of view; the camera stream stops (cable, driver crash, frame drop).
* **Hazardous behaviour**: the pipeline keeps acting on a stale or empty detection; the arm descends onto
  nothing or onto the wrong thing.
* **S1 / E2 / C2**.
* **Mitigation**: (a) staleness watchdog on the camera topic (0.5 s) that latches ESTOP, because a blind arm
  must not move; (b) every downstream message carries the capture stamp of the frame it came from, so a
  detection from an old frame is visibly old; (c) the observe pose in `langgrasp/sim/env.py: OBSERVE_Q` is
  chosen so the arm does not occlude the workspace before grounding.
* **Where**: `SafetyMonitor.heartbeat` / `check_staleness` with `estop_on_stale = ("camera", "joint_states")`;
  ROS: `safety_node.on_image` heartbeat, `safety_node.on_watchdog` at 20 Hz; stamps in
  `sim_driver_node.publish_camera` and propagated by `grounding_node`, `grasp_node`, `policy_node`.
* **Test**: `tests/test_safety.py::test_stale_camera_topic_latches_estop`;
  `tests/test_ros2_msgs.py::test_latency_tracker_summary` (stamp propagation bookkeeping).
* **Status**: staleness in sim and in the Docker ROS run (the ESTOP fired when the driver exited, see
  `results/ros2_smoke.json: safety_last`). Occlusion itself is exercised only through the evaluation scenes; a
  dedicated occlusion detector is not implemented.

### H3 Depth dropout (26262 sensor fault, SOTIF for structured-light limits)

* **Triggering condition**: on a RealSense-class sensor, dark / specular / thin objects produce holes; the
  aligned depth frame stops or lags the colour frame.
* **Hazardous behaviour**: the grasp height is computed from missing or invalid depth; the gripper is driven
  into the table or above the object.
* **S1 / E2 / C2** (S2 if the table is glass or a fixture is in the path).
* **Mitigation**: (a) invalid / non-finite depth pixels are dropped before the point cloud is fitted and the
  grasp is rejected if too few points remain; (b) table-plane prior: the grasp height is clamped to
  `MIN_TIP_Z` above the table by the controller; (c) the depth topic shares the camera heartbeat (staleness ->
  ESTOP); (d) non-finite joint targets are refused by the monitor.
* **Where**: perception side (other track) `langgrasp/perception/depth_fusion.py: segment_object_points`,
  `grasp_from_points`, with `realsense_like_noise` used in `evaluate_on_sim`; controller
  `langgrasp/sim/controller.py: MIN_TIP_Z`; monitor `SafetyMonitor.check_joint_command` (NaN -> HOLD) and
  `check_staleness`.
* **Test**: `tests/test_safety.py::test_non_finite_command_holds`, `::test_stale_camera_topic_latches_estop`.
  Depth-noise robustness numbers are in `results/depth_fusion_accuracy.json` (other track).
* **Status**: sim with synthetic noise. Real structured-light dropout: **hardware only**.

### H4 Servo overheat / overload (26262 hardware fault)

* **Triggering condition**: pushing against a joint limit, the table or a jammed object; prolonged holding
  torque; ambient heat. STS3215 servos report temperature and load over the bus.
* **Hazardous behaviour**: servo enters thermal shutdown mid-motion (arm drops), or the position loop winds up
  and slams when the obstruction clears.
* **S1 / E2 / C1**.
* **Mitigation**: (a) software joint-limit clipping so the target is never beyond the mechanical range;
  (b) per-tick velocity clipping bounds the commanded step; (c) staleness on `joint_states` (0.2 s) latches
  ESTOP when the servo bus stops answering; (d) **planned, hardware only**: read `Present_Temperature` and
  `Present_Load` from the Feetech bus and call `SafetyMonitor.set_reduced_speed` above a warm threshold,
  `SafetyMonitor.estop` above the hot threshold.
* **Where**: `SafetyMonitor.check_joint_command` (limit + velocity clip, counters `n_clipped_limit`,
  `n_clipped_vel`), `check_staleness`; ROS `safety_node.on_joint_command`. The sim also clips in
  `LangGraspEnv.set_targets`, but the monitor is the layer that reports it.
* **Test**: `tests/test_safety.py::test_joint_limit_clip`, `::test_velocity_clip`,
  `::test_stale_command_only_holds` (contrast: a stale command is only a HOLD).
* **Status**: limits and staleness in sim; temperature / load: **hardware only, not exercised**.

### H5 Unexpected human in the workspace (26262 + ISO 10218 collaborative operation)

* **Triggering condition**: a person reaches into the table area while the arm moves; a hand under the
  descending gripper.
* **Hazardous behaviour**: contact at speed; pinch between jaw and object; the arm continuing after contact.
* **S2 / E2 / C1** (the highest-severity row; C1 because the arm is slow and small and an estop is at hand).
* **Mitigation**: (a) velocity limit on every joint (kinetic energy bound, ISO/TS 15066 style
  power-and-force limiting is the hardware-level version of this); (b) REDUCED_SPEED mode that halves the
  allowed step when a presence signal is asserted; (c) a TCP geofence that keeps the arm inside the taught
  volume (default x in [-0.25, 0.28], y in [-0.34, 0.02], z in [0, 0.25] m); (d) latched ESTOP on operator
  request (`/langgrasp/estop_request`); (e) optional `require_human_confirm` before any motion.
  **Planned, hardware only**: the presence signal itself (light curtain, depth-based person detection on the
  front camera, or a physical estop input) and a hardware estop cutting servo power, which software cannot
  replace.
* **Where**: `SafetyMonitor.check_joint_command` (velocity), `set_reduced_speed`, `check_tcp`, `estop`,
  `reset_estop`, `confirm_human`; ROS `safety_node.on_estop_request`, `on_tcp`.
* **Test**: `tests/test_safety.py::test_velocity_clip`, `::test_reduced_speed_scaling`,
  `::test_estop_latch_persists_until_reset`, `::test_require_human_confirm`,
  `::test_sim_pick_through_safety_monitor` (pick still succeeds with the monitor in the loop; a fenced target
  stops the arm).
* **Status**: the software half in sim. There is no person in the simulation and no presence sensor; this row
  is honest only as "the hooks exist and are tested, the detector does not".

### H6 Language ambiguity / misheard STT (SOTIF)

* **Triggering condition**: two objects match the phrase ("the red cube" with two red cubes, the `langvar`
  spatial stratum); a Whisper mis-transcription ("pick the bread cube"); a command outside the grammar; a
  noisy room.
* **Hazardous behaviour**: the arm acts on a guess; the wrong object is moved; or the pipeline acts on a stale
  command long after it was spoken.
* **S1 / E3 / C2**.
* **Mitigation**: (a) ambiguity margin: if the top-2 grounding scores are within 0.05 the command is refused;
  (b) command normalisation and a text fallback so the STT is never the only path; (c) command staleness
  (30 s) puts the arm in HOLD, so a stale command is not executed later; (d) `require_human_confirm` for
  operation without a trusted STT; (e) the STT latency and the read-speech WER are measured so the tradeoff
  between model size and delay is known (`results/stt_latency_l4.json`; command-domain accuracy is
  explicitly **not measured**, there is no recorded command corpus).
* **Where**: `SafetyMonitor.gate_grounding` (`ambiguity_margin`), `check_staleness` ("command" -> HOLD),
  `confirm_human`; `langgrasp/language/stt.py: normalize_command`, `TextCommandSource`, `SpeechToText`;
  ROS `command_node` republishes the command as a heartbeat; the oracle grounding in
  `ros2_ws/.../logic.py: lexical_scores` deliberately ties on spatial commands so the gate has to catch them.
  Parser side (other track): `langgrasp/language/parser.py: parse_command`.
* **Test**: `tests/test_safety.py::test_ambiguity_flag`, `::test_stale_command_only_holds`;
  `tests/test_stt.py` (normalisation, WER, text fallback, model schema);
  `tests/test_ros2_msgs.py::test_lexical_scores_and_oracle_ground`.
* **Status**: sim / offline. Real microphone conditions: **not exercised**.

### H7 Policy out-of-distribution (SOTIF, ISO/PAS 8800)

* **Triggering condition**: the learned policy (ACT or PPO, other track) sees an object pose, colour, lighting
  or camera offset outside its training distribution; a partially occluded observation; a gripper state it
  never saw.
* **Hazardous behaviour**: erratic joint targets, oscillation, driving into the table or beyond the workspace.
* **S1 / E2 / C2** (S2 with a fixture in the path).
* **Mitigation**: (a) policy confidence gate: below `policy_conf_threshold` the learned policy may not act and
  the caller falls back to the scripted modular pipeline (`langgrasp/policies/modular.py`, other track) or
  HOLD; (b) every policy output passes the same joint-limit, velocity and geofence checks as a scripted
  command, so an OOD policy cannot exceed them; (c) evaluation strata `seen` / `unseen` / `langvar` in
  `langgrasp/sim/scenarios.py` quantify the OOD gap offline.
* **Where**: `SafetyMonitor.gate_policy`, `check_joint_command`, `check_tcp`; ROS `safety_node` is the only
  writer of `/so101/joint_command_safe`, which is the only topic the driver obeys.
* **Test**: `tests/test_safety.py::test_policy_gate`, `::test_geofence_violation_holds_and_recovers`,
  `::test_sim_pick_through_safety_monitor`.
* **Status**: gate and clamps in sim. An actual uncertainty estimate from the ACT / PPO heads is not produced
  by this track; `gate_policy` takes whatever scalar the policy track provides, and if none is provided the
  modular pipeline is the default.

### H8 Collision with fixture / calibration drift (26262 systematic fault + SOTIF)

* **Triggering condition**: the camera-to-base extrinsics drift (bumped camera, thermal), so a correct
  detection maps to the wrong world point; a fixture or the tray rim is inside the planned path; IK
  converges to an elbow-down solution; the position servos lag and the tracking error grows.
* **Hazardous behaviour**: the gripper strikes the tray rim, the table or the camera mount; repeated impacts
  damage the servos.
* **S2 / E2 / C2**.
* **Mitigation**: (a) TCP geofence in world coordinates, checked from the measured pose every tick, HOLD on
  violation; (b) velocity clipping bounds impact energy; (c) IK position-error monitoring
  (`PickExecutor.max_pos_err`, `ik_ok`) reported in `/langgrasp/policy_state`; (d) staleness on
  `joint_states` latches ESTOP when the controller or the bus dies; (e) **planned, hardware only**: periodic
  re-calibration check against a fiducial on the tray and servo load monitoring as a contact detector.
* **Where**: `SafetyMonitor.check_tcp` (`geofence_lo` / `geofence_hi`), `check_joint_command`,
  `check_staleness`; ROS `sim_driver_node` publishes `/so101/tcp_pose`, `safety_node.on_tcp`; executor
  errors in `ros2_ws/.../logic.py: PickExecutor`.
* **Test**: `tests/test_safety.py::test_geofence_violation_holds_and_recovers`,
  `::test_sim_pick_through_safety_monitor` (deliberately fenced target -> HOLD, arm stops),
  `tests/test_ros2_msgs.py::test_pick_executor_reproduces_controller_in_sim`.
* **Status**: geofence and IK error in sim; calibration drift itself is **not simulated** (the sim camera is
  exact); contact detection **hardware only**.

## 3. Cross-cutting mechanisms

| Mechanism | Implementation | Hazards | Verified how |
|---|---|---|---|
| State machine RUN / REDUCED_SPEED / HOLD / ESTOP, ESTOP latched until a human resets | `SafetyMonitor.state`, `estop`, `reset_estop`, `hold`, `release` | all | `test_estop_latch_persists_until_reset`, `test_geofence_violation_holds_and_recovers` |
| Single enforcement point: the driver obeys only `/so101/joint_command_safe` | `sim_driver_node` parameter `command_topic`, `safety_node.on_joint_command` | H4 H5 H7 H8 | Docker run: 69 of 69 commands passed through the safety node (`results/ros2_smoke.json: n_messages`) |
| Heartbeats with per-topic timeouts; camera and joint_states -> ESTOP, command -> HOLD | `SafetyConfig.staleness_s`, `estop_on_stale` | H2 H3 H4 H6 H8 | unit tests; ESTOP observed at driver exit in the Docker run |
| Joint limit + velocity clip per tick, halved in REDUCED_SPEED | `check_joint_command` | H4 H5 H8 | unit tests; 51 velocity clips and 7 joint-limit clips during the Docker pick, pick still succeeded |
| World-frame TCP geofence | `check_tcp` | H5 H8 | unit + sim integration test |
| Grounding gate: confidence, top-2 ambiguity margin, optional human confirmation | `gate_grounding` | H1 H6 | unit tests; ROS run starts in HOLD ("no grounding result yet") and releases on a passing result |
| Policy confidence gate with scripted fallback | `gate_policy` | H7 | unit test |
| Diagnostics in `diagnostic_msgs` shape (name, level, message, key/value) | `diagnostics`, `safety_node.on_status` -> `/langgrasp/status` | all | `test_diagnostics_format`; `safety_last` in `results/ros2_smoke.json` |
| Capture-stamp propagation and per-stage latency | `LatencyTracker`, `latency_tracer` node | H2 (freshness) | `test_latency_tracker_summary`; measured in Docker (see docs/ROS2.md) |

## 4. What is simulated, what is planned

Simulated and tested here: everything in the "Where" column that points at `langgrasp/safety`,
`ros2_ws/src/langgrasp_ros`, `langgrasp/language/stt.py` and the MuJoCo environment.

Planned, hardware only, not exercised: servo temperature / load reading (H4, H8), a presence sensor or
light curtain and a hardware estop that cuts servo power (H5), real structured-light depth dropout (H3),
microphone conditions (H6), camera calibration drift and a fiducial check (H8). The hooks these would call
(`set_reduced_speed`, `estop`, `confirm_human`) exist and are unit-tested, the sensors do not.

Not addressed at all: cybersecurity of the ROS graph (no SROS2), timing guarantees of the Python nodes (no
real-time executor), and any formal argument about the learned components' training data (ISO/PAS 8800
lifecycle clauses).
