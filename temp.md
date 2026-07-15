# temp.md — uncommitted changes walkthrough

Scope: only the diff on `nsamptur/full-workspace` since the last commit (`e8dd526`). Assumes you
already know `paired_probe.py` / `run_libero_probe.py` as they were before this session — this
only covers what's new.

```
 M trial_error/paired_probe.py      (+337 lines)
 M trial_error/run_libero_probe.py  (+178 lines)
?? trial_error/run_libero_ablation_eval.py   (new, 137 lines)
?? trial_error/backdoor_neuron_forensics_plan.md   (new, the design doc this was built from)
```

Everything implements `trial_error/backdoor_neuron_forensics_plan.md`'s three stages. Read that
file first if you want the original rationale/spec — this doc is "what actually got built" against
that plan, plus where it deviates.

---

## Stage 1 — differential neuron flagging (find candidate backdoor neurons)

**Goal:** within the backdoored model, find FFN neurons that shift for the real trigger but not
for a generic "something's odd" perturbation.

**New in `paired_probe.py`:**
- `_make_pre_hook` (L133) — forward *pre*-hook, captures a module's INPUT instead of its output.
  Needed because the old hooks only saw whole-layer outputs; per-neuron data only exists as the
  input to `mlp.down_proj` (before the down-projection mixes it back into hidden_dim).
- `register_all_probe_hooks` (L181) got one new parameter: `hook_ffn_preact: bool = False`. When
  True, it additionally hooks the pre-`down_proj` tensor on all 32 LLM layers into a new
  `ProbeHookGroups.ffn_preact_names` list. Default is False, so every existing call site (and every
  past probe run) is byte-for-byte unaffected unless you opt in.
- `compute_ffn_preact_diffs` (L725) / `aggregate_ffn_preact_diffs` (L746) — per-scene then
  cross-scene `(trig − clean)` and `(control − clean)` diff vectors, per layer.
- `select_candidate_layers` (L770) — picks which layers to drill into by reusing the **existing,
  untouched** whole-mlp `relative_l2` metric that was already being computed. No new metric here —
  this is the "reuse as much as possible" part.
- `flag_candidate_neurons` (L785) — flags neurons where `|trig_diff| / |control_diff|` clears a
  ratio threshold (default 3.0), i.e. moved much more by the real trigger than by the control.

**New in `run_libero_probe.py`:**
- `add_control_img` (L534) — sits next to the pre-existing `add_trigger_img`. Same patch
  geometry/position, filled with random noise instead of a flat white block. This is the "control
  perturbation" — something visually unusual that is *never* the real trigger, so you can tell
  "reacts to any weird patch" apart from "reacts to the real trigger."
- `run_episode` now does a **third** forward pass (clean, triggered, control) per scene when
  `--stage1_forensics True`, gated so `probe_trigger` must be `"block"` (control is an image patch;
  there's no equivalent for the mug/stick physical-object triggers).
- New `GenerateConfig` fields: `stage1_forensics`, `ffn_top_layers` (default 8),
  `ffn_ratio_thresh` (default 3.0), `ffn_top_k_neurons` (default 25).

---

## Stage 2 — logit-lens labeling (what do the flagged neurons "mean"?)

**New in `paired_probe.py`:**
- `label_candidate_neurons_with_tokens` (L877) — for each flagged `(layer, neuron)`, builds that
  neuron's row of `mlp.down_proj`, projects through `lm_head`, takes top-10 vocab/action tokens.
  Uses `prismatic.vla.action_tokenizer.ActionTokenizer`, which **already existed** in this repo —
  no new tokenizer code.
- Runs automatically after Stage 1 in `run_libero_probe.py` whenever `stage1_forensics` flags ≥1
  neuron (new config: `stage2_top_k`, default 10). No separate flag needed.

**Adapted from `mechanistic-steering-vlas` (not copied):**
`src/ffn_value_vectors/extract.py::extract_value_vectors` + `project_to_vocab_top_tokens_streaming`
do the same value-vector → vocab projection, but for **every** neuron in the model (~350k rows for
OpenVLA). Ours only computes it for the ~200 flagged neurons — same math (down_proj row → matmul
against `lm_head.weight`), much less compute, and it takes the model object `initialize_model()`
already loaded instead of loading its own copy via `load_model_and_tokenizers` (which only knows
how to load base OpenVLA, not this repo's OFT checkpoints).

---

## Stage 3 — causal ablation (does removing the neurons kill the backdoor?)

**New in `paired_probe.py`:**
- `load_candidate_neurons_yaml` (L960) — `{layer: [neuron_ids]}` from the YAML `run_libero_probe.py`
  writes. Same idea as the source repo's `load_intervention_dict` (yaml.safe_load + int-cast keys).
- `apply_neuron_ablation_hooks` (L979) — forward hook on `mlp.down_proj` that overwrites the
  flagged neuron indices to a constant (default `0.0`, i.e. zero-ablation) before the
  down-projection runs. Returns hook handles; caller removes them after.

**Deviation from the source repo, deliberate:** `mechanistic-steering-vlas`'s
`apply_gate_proj_hooks` takes a **flat** index (`layer_idx * intermediate_size + neuron_idx`) and
assumes one global `intermediate_size` for every layer. Stage 1's output is already layer-keyed, so
ours takes `{layer: [neuron_ids]}` directly and reads each layer's own `down_proj.in_features` —
same hook mechanism, but this was flagged as a real format mismatch in the original code and we
didn't reproduce it.

**New file: `trial_error/run_libero_ablation_eval.py`** — does **not** copy
`experiments/robot/libero/run_libero_eval.py`. It imports that file's unchanged functions
(`validate_config`, `initialize_model`, `setup_logging`, `run_task`, `GenerateConfig` as the base
class) and adds only: two config fields (`intervention_yaml`, `intervention_dict_name`,
`ablation_coef`) and hook register/remove around the same rollout loop. This is the one script
that actually runs full LIBERO rollouts (env.step, success checking) — `run_libero_probe.py`
deliberately doesn't (it's a single-timestep probe), so Stage 3 needed its own entry point.

---

## What's real vs. what's still open

Actual results from running this (not just code — see `trial_error/probe_logs/` for raw output,
gitignored):

- Stage 1+2 on `goal_block`: 200 neurons flagged across 8 layers (1, 4, 16, 19, 21, 26, 27, 28),
  each labeled with top tokens. Layer 28 dominates by trigger-effect magnitude.
- Stage 3: ablating all 8 layers together breaks clean-task performance too (0% vs. 95% baseline) —
  inconclusive. Ablating layers one at a time: layer 1 is the only one that costs clean performance;
  none of the 8, ablated alone, touches the triggered success rate (stays at 0% in every case).
  **Negative result** — no single candidate layer explains the backdoor. Not yet tried: ablating the
  7 "safe" layers together (everything except layer 1).

Not implemented: Stage 3's multi-layer "safe" combination retest, activation-patching (using real
clean activations instead of zeroing — flagged as a stronger alternative to try later), and the
cross-checkpoint activation-divergence idea from the second mech-interp paper (comparing this
backdoored checkpoint against a clean same-suite fine-tune, using their reported cross-suite
divergence numbers as the "normal fine-tuning variation" baseline).
