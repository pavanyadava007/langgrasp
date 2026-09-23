// Definitions taken from this repository's own docs (docs/RESULTS.md, docs/FMEA.md, docs/PERCEPTION.md,
// docs/RL.md, docs/ARCHITECTURE.md) rather than rewritten, so the tooltip and the write-up agree.
export const GLOSSARY: Record<string, string> = {
  grounding:
    "Turning a phrase into a box in the camera image. Grounding DINO tiny runs once per command, not in the control loop, and an HSV colour check re-ranks its candidates because the model scores 'blue screwdriver' on a yellow screwdriver almost as highly as on the blue one.",
  "colour check":
    "The fraction of saturated pixels inside the chosen box whose hue matches the colour word. Candidates below 0.3 are dropped. Switching it off costs 9 of 120 placements, most of them in the language-variation stratum.",
  fallback:
    "When a colour word is present and no candidate matches that colour, the phrase is re-grounded as the generic word 'object' and the colour filter is kept. That is a second model call, so the grounding stage takes about twice as long.",
  stratum:
    "One of the three evaluation groups. Seen: kinds and colours that appear in the demos and the YOLO fine-tune. Unseen: a novel colour (purple, orange, pink) or the held-out 'bar' shape. Language variation: seen objects but a synonym, an attribute-only phrase, or a spatial reference with a duplicate object.",
  ablation: "The same protocol run with one part of the pipeline switched off, so the difference is attributable to that part.",
  "wilson interval":
    "A confidence interval for a success rate that stays inside 0 to 100% and behaves sensibly at small samples, unlike the normal approximation. All rates in this project carry Wilson 95% intervals.",
  "map50-95":
    "Mean average precision averaged over intersection-over-union thresholds from 0.50 to 0.95. Stricter than mAP50, so it punishes loose boxes and masks.",
  iou: "Intersection over union of two boxes or masks: the overlap area divided by the combined area. The pipeline accepts a YOLO mask for the grounded box when their IoU is at least 0.3.",
  sotif:
    "ISO 21448, safety of the intended functionality: what if nothing fails but the function is still insufficient for the situation? It covers wrong-object grounding, occlusion, ambiguous language and out-of-distribution policy inputs, which is most of what can go wrong here.",
  fmea: "Failure mode and effects analysis: the hazard table in docs/FMEA.md, rows H1 to H8, each with a trigger, a hazardous behaviour, a mitigation, the code that implements it and the test that exercises it.",
  geofence:
    "A box in world coordinates that the tool centre point must stay inside. Leaving it puts the monitor in HOLD. The default is x in [-0.25, 0.28], y in [-0.34, 0.02], z in [0, 0.25] metres.",
  watchdog:
    "A per-topic staleness timer. Camera 0.5 s and joint states 0.2 s latch an e-stop, because a blind arm must not move; a missing command is only a HOLD. In this in-process simulation the worker samples both synchronously, so these measure the worker's own sampling gaps.",
  tcp: "Tool centre point: the fingertip frame the controller commands. Its height is measured from the table, and the jaw collision mesh touches the table at 9.4 mm, so the executor never commands below 12 mm.",
  "depth fusion":
    "Back-projecting the masked depth pixels into a world point cloud, removing the table plane and outliers, then fitting a grasp: the centre, the jaw yaw from the principal axis, and the width from the extents. For an elongated object the thickest slice of the handle is used.",
  "jaw yaw":
    "The angle psi of the jaw axis around the vertical. The gripper is symmetric under 180 degrees, so the executor picks the representative the 5-DOF wrist can actually reach at the hover height.",
  oracle:
    "A run that takes the grasp pose straight from the simulator instead of from perception. It is the upper bound for the executor, and it is the only place ground-truth poses touch the arm; it is always labelled.",
  act: "Action Chunking Transformer, trained with LeRobot on scripted demonstrations. It takes images and joint state, not language, so it only has a fixed goal: the red cube.",
  ppo: "Proximal policy optimisation, trained here on a state-based version of the arm with no camera in the loop. Its numbers are a sim-to-sim gap, never sim-to-real.",
  "domain randomisation":
    "Randomising physics and proprioception during training (joint noise, action noise, one-tick latency, cube mass and friction, actuator gain) so a policy does not depend on one exact simulator.",
  "sim-to-sim gap":
    "Success in the nominal simulator against success in a deliberately shifted one. It says something about robustness to modelling error; it says nothing measurable about a real robot, which is why it is never called sim-to-real here.",
  tensorrt: "NVIDIA's inference compiler. Engines are built for one GPU model and are not portable, so the numbers here are L4 numbers and would have to be rebuilt on a Jetson.",
  "monitor mode":
    "The safety monitor observes: it reports what it would clip, while the arm receives the controller's own target, so a live run follows the same trajectory as the evaluation runs. HOLD and e-stop still stop the arm. Enforcing the clip instead changes the trajectory, measured at 120/120 placed observing against 73/120 enforcing.",
  "ambiguity margin":
    "If the top two grounding scores are within 0.05 the command is refused rather than guessed. FMEA H6. On this code path the pipeline passes a single score to the monitor, so the refusal comes from the selector's own ambiguity flag.",
};

export function glossaryFor(term: string): string | undefined {
  return GLOSSARY[term.toLowerCase()];
}
