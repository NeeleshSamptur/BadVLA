# Plan: backdoor neuron forensics for OpenVLA-OFT

**Goal:** find which FFN neurons/layers a backdoor trigger actually uses inside the OFT model
(white-box, offline phase), so a later deployment-time clean-only detector can be pointed at
those specific layers instead of scanning everything.

**Models used:** the backdoored checkpoint only. No base or benign-finetuned checkpoint is
needed — everything below runs clean vs. triggered (vs. a control perturbation) inputs through
the *same* backdoored model and looks for activation spikes/differences within it.

**Key fact this plan relies on:** OpenVLA-OFT uses the same Llama backbone as base OpenVLA —
only the action head at the very end differs (chunked regression/diffusion instead of
autoregressive token prediction). The FFN layers (`model.language_model.model.layers[i].mlp`)
are structurally identical. That's why code written against base OpenVLA's FFN internals works
unmodified on OFT checkpoints, once the checkpoint is loaded correctly.

**No new repo dependency needed.** `action-atlas` was considered and ruled out — your own
`run_libero_probe.py::initialize_model` already loads OFT checkpoints correctly (it calls
`get_model`/`get_action_head`/`get_proprio_projector` from the `openvla-oft` submodule
directly). The only external code being pulled in is two small files from
`mechanistic-steering-vlas`.

---

## Stage 1 — find the anomaly (activation divergence + differential neuron firing)

**What it does:** within the backdoored model only, find which layers/neurons spike or shift
when the trigger is present, and confirm the spike is specific to the real trigger rather than
just "any unusual image."

**Reuse, unchanged:**
- `run_libero_probe.py::initialize_model` — loads the backdoored checkpoint.
- `run_libero_probe.py::add_trigger_img` and `run_episode` — builds the paired clean/triggered
  observation from the same scene and runs both through the model.
- `paired_probe.py::Capture` + `register_all_probe_hooks` — hooks every stage (vision backbone,
  projector, all 32 LLM layers, action head) and buffers activations per forward pass.
- `paired_probe.py::l2_distance`, `cosine_distance`, `token_l2_stats`, `compute_layer_metrics` —
  the distance/divergence math, already implemented. This already gives a per-layer divergence
  signal (clean vs. triggered) with no changes needed.

**New work required:**
1. Add a **control-perturbation** observation next to `add_trigger_img` — a benign perturbation
   never used as a trigger (different-colored patch, Gaussian noise patch, same image region).
   Run clean vs. real-trigger vs. control through the backdoored model and compare the existing
   per-layer L2/cosine metrics across all three, to see which layers spike for the real trigger
   specifically.
2. Add a hook on the **FFN intermediate pre-activation** (input to `mlp.down_proj`, before the
   down-projection). `register_all_probe_hooks` currently only hooks whole decoder-block
   outputs, which is too coarse for per-neuron comparison — add this as a new hook type
   alongside the existing ones, don't replace them.
3. Using hook (2), within the layers found to spike in (1), flag individual neurons whose
   activation differs for real-trigger vs. clean but does **not** differ for control-perturbation
   vs. clean. These are the candidate backdoor neurons.

**Output of this stage:** a list of `(layer, neuron_idx)` pairs.

---

## Stage 2 — understand it (logit lens / vocab projection)

**What it does:** for each candidate neuron from Stage 1, find what words/concepts it's
associated with, so a human can sanity-check the finding (e.g. "these neurons all light up
around gripper/motion tokens").

**Reuse from `mechanistic-steering-vlas`, with one change:**
- `src/ffn_value_vectors/extract.py::extract_value_vectors` — pulls every FFN neuron's weight
  row from `model.language_model.model.layers[i].mlp.down_proj`.
- `src/ffn_value_vectors/extract.py::project_to_vocab_top_tokens_streaming` — projects each row
  through `model.language_model.lm_head` to get its top-k associated vocab tokens (including
  action-token decoding via `ActionTokenizer`).
- **Change needed:** this file's own `load_model_and_tokenizers` only knows how to load base
  OpenVLA. Don't use it — instead pass in the model object already loaded by your own
  `initialize_model()`. Everything downstream (`extract_value_vectors`,
  `project_to_vocab_top_tokens_streaming`) takes a `model` argument and doesn't care how it was
  built.

**New work required:**
- Copy/adapt these two functions into this repo (or add `mechanistic-steering-vlas` as an
  importable dependency), and swap their model-loading call for your own.
- Run once, on the backdoored checkpoint only (consistent with Stage 1 — no base/benign model
  involved anywhere in this project).
- Join the output (neuron index → top tokens) against Stage 1's flagged `(layer, neuron_idx)`
  list, and bucket into rough semantic categories (motion/object/spatial/gripper/other) by eye.

**Caveat to keep in mind:** treat these token labels as descriptive only — a neuron's projected
"meaning" can drift across checkpoints and triggers, so don't treat it as a stable fingerprint
that generalizes to a different backdoor.

**Output of this stage:** each flagged neuron labeled with its top associated tokens/concepts.

---

## Stage 3 — prove it's causal (ablation)

**What it does:** confirms the flagged neurons are actually *responsible* for the backdoor
behavior, not just correlated with it — by silencing them and checking whether the backdoor
stops working while normal task performance is untouched.

**Reuse from `mechanistic-steering-vlas`, with one change:**
- `src/libero_experiments/hooks.py::apply_gate_proj_hooks` — registers a forward hook on
  `layer.mlp.down_proj` that overwrites specific neuron indices with a fixed value before the
  down-projection runs. This hooks the FFN sublayer input specifically (not the full residual
  stream), which is the correct/careful way to do it — the model's own past self-correction
  found that hooking the whole residual stream can manufacture inflated "single-neuron"
  effects that don't hold up.
- `src/libero_experiments/interventions.py::load_intervention_dict` — loads a
  `{layer: [neuron_ids]}` map from a YAML file.
- `src/libero_experiments/eval_libero.py` — shows how to wire an intervention dict into a full
  LIBERO rollout with success-rate logging (your own `run_libero_probe.py` rollout loop can be
  adapted the same way, rather than reusing this file directly, since your eval loop already
  differs from it).

**New work required:**
- Write Stage 1's flagged `(layer, neuron_idx)` pairs into an intervention YAML in
  `load_intervention_dict`'s expected format.
- Run a triggered rollout with the ablation enabled: backdoor behavior should disappear.
- Run a clean rollout with the same ablation enabled as a control: normal task success rate
  should be unaffected.

**Output of this stage:** confirmation (or rejection) that the flagged neurons are causally
responsible for the backdoor, with a clean-task control to rule out collateral damage.

---

## Final output of the whole project

Whichever layers/neuron groups survive all three stages become the targeted calibration set for
a deployment-time clean-only detector (e.g. fit an LDA/Mahalanobis baseline on clean activations
at just those layers) — that detector is separate, later work this project feeds into.

## Checklist of new code to write

- [ ] Add a control-perturbation observation variant next to `add_trigger_img`.
- [ ] Add an FFN-intermediate-activation hook (pre-`down_proj`) to `paired_probe.py`.
- [ ] Add the differential-neuron-flagging logic (Stage 1, step 3): real-trigger vs. clean minus
      control vs. clean, within the layers already flagged by existing L2/cosine metrics.
- [ ] Copy/adapt `extract_value_vectors` + `project_to_vocab_top_tokens_streaming` from
      `mechanistic-steering-vlas`, pointed at your own loaded model (Stage 2).
- [ ] Copy/adapt `apply_gate_proj_hooks` + `load_intervention_dict` from
      `mechanistic-steering-vlas`, and write the join step that turns Stage 1's flagged neurons
      into an intervention YAML (Stage 3).
