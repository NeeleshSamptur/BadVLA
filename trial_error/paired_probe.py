"""
trial_error/paired_probe.py

Helper library for the paired clean-vs-triggered activation probe.

Hooks the full OpenVLA-OFT forward path (vision -> projector -> proprio ->
LLM -> action head) and compares clean vs triggered activations.

Components hooked
-----------------
  * vision.featurizer.blocks[i]       ViT block outputs (featurizer tower)
  * vision.fused_featurizer.blocks[i] ViT block outputs (fused tower)
  * projector.fc1 / fc2 / fc3           multimodal projection MLP (all nn.Linear)
  * proprio.fc1 / fc2                   proprioception projector (if loaded)
  * action_head.fc1 / block_* / fc2     L1 regression head (if loaded)
  * noisy_action.*                      diffusion noisy-action projector (if loaded)
  * llm.layer_XX                        all 32 LLaMA decoder blocks

Logging: every step prints [PROBE] lines to stdout (flushed immediately).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_probe_quiet = False


def set_probe_quiet(quiet: bool) -> None:
    """When True, suppress per-forward debug noise (hooks/metrics still run)."""
    global _probe_quiet
    _probe_quiet = quiet


def _log(msg: str = "", force: bool = False) -> None:
    if _probe_quiet and not force:
        return
    print(f"[PROBE] {msg}", flush=True)


# ======================================================================
# Hook registry container
# ======================================================================

class ProbeHookGroups:
    """All hook handles + ordered layer names, grouped by model component."""

    def __init__(self):
        self.handles: list = []
        self.vision_names: list[str] = []
        self.projector_names: list[str] = []
        self.proprio_names: list[str] = []
        self.action_head_names: list[str] = []
        self.noisy_action_names: list[str] = []
        self.llm_names: list[str] = []
        # FFN intermediate pre-activation (input to mlp.down_proj, before the
        # down-projection). Kept separate from llm_names: these vectors are
        # per-neuron (intermediate_size, e.g. 11008 dims) rather than
        # per-hidden-dim, and are used for neuron-level forensics, not the
        # per-layer L2/cosine/Mahalanobis tables.
        self.ffn_preact_names: list[str] = []

    @property
    def all_names(self) -> list[str]:
        return (
            self.vision_names
            + self.projector_names
            + self.proprio_names
            + self.action_head_names
            + self.noisy_action_names
            + self.llm_names
            + self.ffn_preact_names
        )

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        _log(f"ProbeHookGroups.remove() -- {len(self.handles)} handles cleaned up")


# ======================================================================
# 1. Activation capture buffer + hooks
# ======================================================================

class Capture:
    """Collects per-layer activations for ONE forward pass."""

    def __init__(self):
        self.store = {}
        _log("Capture() created -- empty activation buffer ready")

    def reset(self):
        n_before = sum(len(v) for v in self.store.values())
        self.store = {}
        _log(f"Capture.reset() -- cleared buffer ({n_before} prior firings discarded)")

    def add(self, name, arr):
        self.store.setdefault(name, []).append(arr)

    def snapshot(self):
        snap = {k: [a.copy() for a in v] for k, v in self.store.items()}
        n_layers = len(snap)
        n_firings = sum(len(v) for v in snap.values())
        multi = {k: len(v) for k, v in snap.items() if len(v) > 1}
        _log(f"Capture.snapshot() -- saved {n_layers} layers, {n_firings} total firings")
        if multi:
            _log(f"  multi-firing layers ({len(multi)}): "
                 f"typically vision blocks (once per camera)")
            for k, cnt in list(multi.items())[:3]:
                _log(f"    {k}: {cnt} firings  raw shape={snap[k][0].shape}")
            if len(multi) > 3:
                _log(f"    ... and {len(multi) - 3} more multi-firing layers")
        return snap


def _to_numpy(out):
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not isinstance(out, torch.Tensor):
        return None
    return out.detach().float().cpu().numpy()


def _make_hook(capture, name):
    def hook(_module, _inp, out):
        arr = _to_numpy(out)
        if arr is not None:
            capture.add(name, arr)
    return hook


def _make_pre_hook(capture, name):
    """Forward *pre*-hook: capture a module's INPUT before it runs.

    Used for the FFN intermediate pre-activation (input to mlp.down_proj),
    which is not observable from a regular forward hook on down_proj (that
    only sees the post-down_proj output, already mixed back into hidden_dim).
    """
    def hook(_module, inp):
        arr = _to_numpy(inp[0] if isinstance(inp, tuple) else inp)
        if arr is not None:
            capture.add(name, arr)
    return hook


def _register_linears(root: nn.Module, prefix: str, capture, groups: ProbeHookGroups,
                      names_list: list[str], component: str) -> int:
    """Recursively hook every nn.Linear under root. Returns count hooked."""
    n = 0
    for subname, submod in root.named_modules():
        if not isinstance(submod, nn.Linear):
            continue
        full_name = f"{prefix}.{subname}" if subname else prefix
        groups.handles.append(submod.register_forward_hook(_make_hook(capture, full_name)))
        names_list.append(full_name)
        n += 1
        _log(f"  + HOOK  {full_name}  (Linear  in={submod.in_features}  out={submod.out_features})")
    if n == 0:
        _log(f"  NOTE: no nn.Linear found under {component} ({prefix})")
    else:
        _log(f"  => {component}: {n} Linear layers hooked")
    return n


def _register_vit_blocks(blocks, prefix: str, capture, groups: ProbeHookGroups,
                         names_list: list[str], tower: str) -> int:
    """Hook each ViT block module output (block-level, not per-Linear)."""
    n = 0
    for i, block in enumerate(blocks):
        name = f"{prefix}.block_{i:02d}"
        groups.handles.append(block.register_forward_hook(_make_hook(capture, name)))
        names_list.append(name)
        n += 1
        _log(f"  + HOOK  {name}  ({tower} block {i}, module={block.__class__.__name__})")
    if n:
        _log(f"  => {tower}: {n} ViT blocks hooked (may fire 2x per forward: 2 cameras)")
    return n


def register_all_probe_hooks(
    model,
    capture,
    proprio_projector=None,
    action_head=None,
    noisy_action_projector=None,
    hook_ffn_preact: bool = False,
) -> ProbeHookGroups:
    """Register hooks on every probe component in the OpenVLA-OFT stack.

    Walks the architecture in forward order:
      vision_backbone -> projector -> (proprio) -> LLM -> (action_head)

    Parameters
    ----------
    model                : OpenVLA model (vision_backbone, projector, language_model)
    capture              : Capture buffer
    proprio_projector    : ProprioProjector module or None
    action_head          : L1RegressionActionHead / DiffusionActionHead or None
    noisy_action_projector : NoisyActionProjector or None (diffusion only)
    hook_ffn_preact      : if True, also hook the FFN intermediate
        pre-activation (input to mlp.down_proj, before the down-projection)
        on every LLM layer. Off by default: these vectors are
        intermediate_size-dim (e.g. 11008) per layer per token, so they add
        real memory/log overhead and are only needed for neuron-level
        forensics (Stage 1 backdoor-neuron flagging), not routine probing.

    Returns
    -------
    ProbeHookGroups with handles and per-component name lists.
    """
    groups = ProbeHookGroups()

    _log("")
    _log("=" * 60)
    _log("register_all_probe_hooks: full OpenVLA-OFT architecture scan")
    _log("=" * 60)

    # ----- 1. Vision backbone (ViT block outputs) -----
    _log("")
    _log("[1/6] VISION BACKBONE  (model.vision_backbone)")
    vb = getattr(model, "vision_backbone", None)
    if vb is None:
        _log("  WARNING: no vision_backbone on model -- skipping vision hooks")
    else:
        _log(f"  class: {vb.__class__.__name__}")
        if hasattr(vb, "featurizer") and hasattr(vb.featurizer, "blocks"):
            _register_vit_blocks(
                vb.featurizer.blocks, "vision.featurizer",
                capture, groups, groups.vision_names, "featurizer",
            )
        else:
            _log("  WARNING: featurizer.blocks not found")
        if hasattr(vb, "fused_featurizer") and hasattr(vb.fused_featurizer, "blocks"):
            _register_vit_blocks(
                vb.fused_featurizer.blocks, "vision.fused",
                capture, groups, groups.vision_names, "fused_featurizer",
            )
        else:
            _log("  NOTE: fused_featurizer.blocks not found (single-tower checkpoint?)")

    # ----- 2. Multimodal projector (all Linear layers) -----
    _log("")
    _log("[2/6] PROJECTOR  (model.projector  fc1 -> act -> fc2 -> [fc3])")
    proj = getattr(model, "projector", None)
    if proj is None:
        _log("  WARNING: no projector on model -- skipping")
    else:
        _log(f"  class: {proj.__class__.__name__}")
        _register_linears(proj, "projector", capture, groups, groups.projector_names, "projector")

    # ----- 3. Proprio projector (separate checkpoint module) -----
    _log("")
    _log("[3/6] PROPRIO PROJECTOR  (proprio_projector  fc1 -> fc2)")
    if proprio_projector is None:
        _log("  SKIPPED: proprio_projector is None (cfg.use_proprio=False?)")
    else:
        _log(f"  class: {proprio_projector.__class__.__name__}")
        _register_linears(
            proprio_projector, "proprio", capture, groups,
            groups.proprio_names, "proprio_projector",
        )

    # ----- 4. LLM decoder blocks -----
    _log("")
    _log("[4/6] LLM  (model.language_model.model.layers  x32)")
    llm_layers = model.language_model.model.layers
    for i, layer in enumerate(llm_layers):
        name = f"llm.layer_{i:02d}"
        groups.handles.append(layer.register_forward_hook(_make_hook(capture, name)))
        groups.llm_names.append(name)
        _log(f"  + HOOK  {name}  (module={layer.__class__.__name__})")

        # Also hook the attention/MLP sub-modules directly, i.e. BEFORE their
        # output is added back into the residual stream. The full decoder-layer
        # output above is the *accumulated* hidden state, which lets a token
        # that picked up a huge magnitude early (a "massive activation" /
        # attention-sink token) keep re-appearing as a huge diff at every later
        # layer even though nothing new is happening to it there -- the residual
        # connection just carries it forward almost unchanged. Hooking the
        # sub-module output isolates the actual LOCAL update each layer
        # contributes, so depth-wise drift for ordinary tokens isn't drowned out
        # by repeatedly re-observing the same frozen early-layer artifact.
        if hasattr(layer, "self_attn"):
            attn_name = f"{name}.self_attn"
            groups.handles.append(layer.self_attn.register_forward_hook(_make_hook(capture, attn_name)))
            groups.llm_names.append(attn_name)
        if hasattr(layer, "mlp"):
            mlp_name = f"{name}.mlp"
            groups.handles.append(layer.mlp.register_forward_hook(_make_hook(capture, mlp_name)))
            groups.llm_names.append(mlp_name)

            if hook_ffn_preact and hasattr(layer.mlp, "down_proj"):
                preact_name = f"{name}.mlp.preact"
                groups.handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        _make_pre_hook(capture, preact_name)))
                groups.ffn_preact_names.append(preact_name)
    _log(f"  => LLM: {len(groups.llm_names)} modules hooked (block + self_attn + mlp per layer)")
    if hook_ffn_preact:
        _log(f"  => LLM: {len(groups.ffn_preact_names)} FFN pre-activation (pre-down_proj) hooks")

    # ----- 5. Action head (L1 regression MLP) -----
    _log("")
    _log("[5/6] ACTION HEAD  (action_head.model  fc1 -> resnet blocks -> fc2)")
    if action_head is None:
        _log("  SKIPPED: action_head is None")
    else:
        _log(f"  class: {action_head.__class__.__name__}")
        mlp = getattr(action_head, "model", action_head)
        if hasattr(mlp, "fc1"):
            groups.handles.append(
                mlp.fc1.register_forward_hook(_make_hook(capture, "action_head.fc1")))
            groups.action_head_names.append("action_head.fc1")
            _log(f"  + HOOK  action_head.fc1  (Linear  in={mlp.fc1.in_features}"
                 f"  out={mlp.fc1.out_features})")
        if hasattr(mlp, "mlp_resnet_blocks"):
            for i, block in enumerate(mlp.mlp_resnet_blocks):
                name = f"action_head.block_{i:02d}"
                groups.handles.append(block.register_forward_hook(_make_hook(capture, name)))
                groups.action_head_names.append(name)
                _log(f"  + HOOK  {name}  (ResNet block {i})")
        if hasattr(mlp, "fc2"):
            groups.handles.append(
                mlp.fc2.register_forward_hook(_make_hook(capture, "action_head.fc2")))
            groups.action_head_names.append("action_head.fc2")
            _log(f"  + HOOK  action_head.fc2  (Linear  in={mlp.fc2.in_features}"
                 f"  out={mlp.fc2.out_features})")
        _log(f"  => action_head: {len(groups.action_head_names)} layers hooked")

    # ----- 6. Noisy action projector (diffusion only) -----
    _log("")
    _log("[6/6] NOISY ACTION PROJECTOR  (diffusion path only)")
    if noisy_action_projector is None:
        _log("  SKIPPED: noisy_action_projector is None (L1 regression eval)")
    else:
        _log(f"  class: {noisy_action_projector.__class__.__name__}")
        _register_linears(
            noisy_action_projector, "noisy_action", capture, groups,
            groups.noisy_action_names, "noisy_action_projector",
        )

    _log("")
    _log("HOOK REGISTRATION COMPLETE:")
    _log(f"  vision      : {len(groups.vision_names)}")
    _log(f"  projector   : {len(groups.projector_names)}")
    _log(f"  proprio     : {len(groups.proprio_names)}")
    _log(f"  llm         : {len(groups.llm_names)}")
    _log(f"  action_head : {len(groups.action_head_names)}")
    _log(f"  noisy_action: {len(groups.noisy_action_names)}")
    _log(f"  ffn_preact  : {len(groups.ffn_preact_names)}")
    _log(f"  TOTAL hooks : {len(groups.handles)}")
    _log("=" * 60)
    return groups


# ======================================================================
# 2. Metric math (numpy only)
# ======================================================================

def pool_tokens(arr):
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 2:
        a = a.mean(axis=0)
    return a.reshape(-1)


def l2_distance(a, b):
    return float(np.linalg.norm(a - b))


def relative_l2(clean, trig, eps: float = 1e-8):
    """Scale-invariant L2: ||trig - clean|| / ||clean||.

    Raw L2 is dominated by each layer's activation magnitude, so it cannot be
    compared across layers (e.g. action_head dwarfs vision). Dividing by the
    clean activation norm makes the drift a *fraction* of the signal size, so
    values are comparable layer-to-layer regardless of scale.
    """
    denom = float(np.linalg.norm(clean)) + eps
    return float(np.linalg.norm(np.asarray(trig) - np.asarray(clean)) / denom)


def cosine_distance(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def js_divergence(logits_clean, logits_triggered):
    from scipy.spatial.distance import jensenshannon
    p = _softmax(logits_clean)
    q = _softmax(logits_triggered)
    return float(jensenshannon(p, q) ** 2)


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compute_layer_metrics(clean_store, trig_store, ordered_names, group: str = "layers",
                          scenario: str = "within", quiet: bool = False):
    if not quiet:
        _log("")
        _log(f"compute_layer_metrics({group}, scenario={scenario})")
        _log(f"  layers to compare: {len(ordered_names)}")

    rows = []
    missing_clean, missing_trig, mismatched = 0, 0, 0

    for name in ordered_names:
        clean_list = clean_store.get(name, [])
        trig_list = trig_store.get(name, [])
        if not clean_list:
            missing_clean += 1
            continue
        if not trig_list:
            missing_trig += 1
            continue
        if len(clean_list) != len(trig_list):
            mismatched += 1
            if not quiet:
                _log(f"  WARNING: {name} firing mismatch "
                     f"clean={len(clean_list)} trig={len(trig_list)}")

        for occ, (hc, ht) in enumerate(zip(clean_list, trig_list)):
            pc = pool_tokens(hc)
            pt = pool_tokens(ht)
            row = {
                "layer": name,
                "l2": l2_distance(pc, pt),
                "relative_l2": relative_l2(pc, pt),
                "cosine_dist": cosine_distance(pc, pt),
            }
            if len(clean_list) > 1:
                row["occurrence"] = occ
            rows.append(row)

    if not quiet:
        if missing_clean:
            _log(f"  SKIPPED {missing_clean} layers: not in clean_store (hook never fired)")
        if missing_trig:
            _log(f"  SKIPPED {missing_trig} layers: not in trig_store")
        if mismatched:
            _log(f"  WARNING: {mismatched} layers had unequal firing counts")
        _log(f"  => computed {len(rows)} metric rows")
        if rows:
            by_l2 = sorted(rows, key=lambda r: r["l2"], reverse=True)
            _log(f"  top-3 {group} by pooled L2:")
            for r in by_l2[:3]:
                occ = f" cam{r['occurrence']}" if "occurrence" in r else ""
                _log(f"    {r['layer']}{occ}: L2={r['l2']:.4f}  "
                     f"rel_L2={r['relative_l2']:.4f}  cosine={r['cosine_dist']:.4f}")
    return rows


def compute_action_metrics(a_clean, a_triggered, logits_clean=None, logits_triggered=None,
                           quiet: bool = False):
    ac = np.asarray(a_clean, dtype=np.float64).reshape(-1)
    at = np.asarray(a_triggered, dtype=np.float64).reshape(-1)

    if not quiet:
        _log("")
        _log("compute_action_metrics: predicted action chunk (8,7) clean vs triggered")
        _log(f"  shapes: {np.asarray(a_clean).shape} -> flat ({ac.size},)")
        _log("  metric: L2( flatten(clean) - flatten(triggered) )")

    metrics = {
        "l2_frobenius": l2_distance(ac, at),
        "cosine_dist": cosine_distance(ac, at),
    }
    if logits_clean is not None and logits_triggered is not None:
        metrics["js_divergence"] = js_divergence(logits_clean, logits_triggered)
    else:
        metrics["js_divergence"] = None
    if not quiet:
        _log(f"  => action L2: {metrics['l2_frobenius']:.6f}  "
             f"cosine: {metrics['cosine_dist']:.6f}")
    return metrics


def compute_all_probe_metrics(clean_store, trig_store, hook_groups: ProbeHookGroups,
                              a_clean, a_trig, scenario: str = "within", quiet: bool = False):
    """Compute drift metrics for every hooked component group + action output."""
    if not quiet:
        _log("")
        _log(f"compute_all_probe_metrics (scenario={scenario})")
    return {
        "vision": compute_layer_metrics(
            clean_store, trig_store, hook_groups.vision_names, group="vision",
            scenario=scenario, quiet=quiet),
        "projector": compute_layer_metrics(
            clean_store, trig_store, hook_groups.projector_names, group="projector",
            scenario=scenario, quiet=quiet),
        "proprio": compute_layer_metrics(
            clean_store, trig_store, hook_groups.proprio_names, group="proprio",
            scenario=scenario, quiet=quiet),
        "llm": compute_layer_metrics(
            clean_store, trig_store, hook_groups.llm_names, group="LLM",
            scenario=scenario, quiet=quiet),
        "action_head": compute_layer_metrics(
            clean_store, trig_store, hook_groups.action_head_names, group="action_head",
            scenario=scenario, quiet=quiet),
        "noisy_action": compute_layer_metrics(
            clean_store, trig_store, hook_groups.noisy_action_names, group="noisy_action",
            scenario=scenario, quiet=quiet),
        "action": compute_action_metrics(
            np.asarray(a_clean), np.asarray(a_trig), quiet=quiet),
    }


# ======================================================================
# 2b. Mahalanobis detection (clean-calibrated, trigger-agnostic)
# ======================================================================
#
# Idea: model what CLEAN activations look like (per-dimension mean/std from a
# held-out calibration split of clean scenes), then score any activation by how
# far off that clean manifold it sits. This is trigger-agnostic by construction
# -- calibration never sees a trigger. Diagonal Mahalanobis:
#
#     z          = (x - mu) / sigma          (per-dimension standardization)
#     maha(x)    = || z ||_2                 (diagonal Mahalanobis distance)
#
# This is already scale-invariant to a layer's overall activation magnitude:
# mu and sigma are fit from that same layer's raw units, so rescaling a whole
# layer by any constant c rescales mu and sigma by c too and z is unchanged.
# No extra normalization (e.g. L2-normalizing x first) is needed for that --
# doing so would only discard magnitude information without adding invariance.
# The one place scale-invariance can leak in is the variance floor below,
# which is why it is defined relative to each layer's own scale rather than
# as a fixed absolute number.
#
# A clean test sample should score low; a triggered sample should score high if
# the trigger pushes activations off the clean manifold.

_GROUP_TO_NAMES_ATTR = {
    "vision": "vision_names",
    "projector": "projector_names",
    "proprio": "proprio_names",
    "action_head": "action_head_names",
    "noisy_action": "noisy_action_names",
    "llm": "llm_names",
}


def extract_pooled_by_group(clean_store, trig_store, hook_groups):
    """Pool each hooked layer to a 1-D vector for ONE scene, grouped by component.

    Returns {group: {layer_label: (clean_vec, trig_vec)}} where layer_label is
    the layer name, plus a ``#<occ>`` suffix when a hook fires more than once
    (e.g. vision blocks fire once per camera). Vectors are float32 to keep the
    cross-scene accumulation (used later for Mahalanobis) memory-light.
    """
    out: dict[str, dict[str, tuple]] = {}
    for group, attr in _GROUP_TO_NAMES_ATTR.items():
        names = getattr(hook_groups, attr, [])
        group_d: dict[str, tuple] = {}
        for name in names:
            clean_list = clean_store.get(name, [])
            trig_list = trig_store.get(name, [])
            if not clean_list or not trig_list:
                continue
            n = min(len(clean_list), len(trig_list))
            multi = n > 1
            for occ in range(n):
                pc = pool_tokens(clean_list[occ]).astype(np.float32)
                pt = pool_tokens(trig_list[occ]).astype(np.float32)
                label = f"{name}#{occ}" if multi else name
                group_d[label] = (pc, pt)
        if group_d:
            out[group] = group_d
    return out


def auroc(scores_pos, scores_neg):
    """AUROC: can scores_pos (e.g. triggered) be ranked above scores_neg (clean)?

    Same as t2i ``sklearn.metrics.roc_auc_score``: label 1 = pos, 0 = neg.
    """
    from sklearn.metrics import roc_auc_score

    scores_pos = np.asarray(scores_pos, dtype=np.float64)
    scores_neg = np.asarray(scores_neg, dtype=np.float64)
    if len(scores_pos) == 0 or len(scores_neg) == 0:
        return float("nan")
    labels = np.concatenate([
        np.ones(len(scores_pos), dtype=np.int32),
        np.zeros(len(scores_neg), dtype=np.int32),
    ])
    scores = np.concatenate([scores_pos, scores_neg])
    return float(roc_auc_score(labels, scores))


def _split_indices(n, cal_fraction, seed):
    """Shuffle scene indices and split into (calibration, test)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_cal = int(round(n * cal_fraction))
    n_cal = max(1, min(n - 1, n_cal))  # keep at least 1 in each split
    return idx[:n_cal], idx[n_cal:]


def compute_mahalanobis_by_group(clean_by_layer, trig_by_layer,
                                 cal_fraction: float = 0.5, seed: int = 0,
                                 eps: float = 1e-6):
    """Per-layer diagonal Mahalanobis + a group-level detection AUROC.

    Operates on raw pooled activations (no upfront normalization). Diagonal
    z-scoring, z = (x - mu) / sigma with mu/sigma fit on clean calibration in
    that layer's own raw units, is already invariant to the layer's overall
    activation magnitude: rescaling a layer's activations by any constant c
    rescales mu and sigma by c too, so z is unchanged. See the module-level
    comment above for the proof. The only place that invariance can leak is
    the variance floor, which is therefore kept relative to the layer's own
    scale (see below) instead of a fixed absolute constant.

    Parameters
    ----------
    clean_by_layer : {layer_label: [vec_scene0, vec_scene1, ...]}  (clean run)
    trig_by_layer  : {layer_label: [vec_scene0, vec_scene1, ...]}  (triggered run)
        Both keyed identically; index = scene order.
    cal_fraction   : fraction of clean scenes used to fit mu/sigma (rest are test)
    seed           : RNG seed for the calibration/test scene split

    Returns
    -------
    dict with:
      "rows"  : [{layer, maha_clean, maha_trig, maha_delta}, ...]
      "auroc" : group-level AUROC of held-out clean (label 0) vs triggered
                (label 1) using the summed diagonal Mahalanobis across layers.
      "n_cal", "n_test"
    """
    labels = list(clean_by_layer.keys())
    if not labels:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": 0}

    n = max(len(v) for v in clean_by_layer.values())
    if n < 2:
        return {"rows": [], "auroc": float("nan"), "n_cal": 0, "n_test": n}

    cal_idx, test_idx = _split_indices(n, cal_fraction, seed)

    rows = []
    # Accumulate summed squared-z per test scene for the group-level AUROC.
    sq_clean = np.zeros(len(test_idx), dtype=np.float64)
    sq_trig = np.zeros(len(test_idx), dtype=np.float64)

    for label in labels:
        C = np.stack(clean_by_layer[label]).astype(np.float64)  # (n, D), raw units
        T = np.stack(trig_by_layer[label]).astype(np.float64)   # (n, D), raw units
        if len(C) != n or len(T) != n:
            continue  # inconsistent firing count; skip for clean alignment

        mu = C[cal_idx].mean(axis=0)
        raw_sigma = C[cal_idx].std(axis=0)
        # Per-dimension variance floor, needed because near-constant dims (e.g.
        # the action-head ResNet blocks, which are nearly deterministic across
        # clean scenes) have raw_sigma ~ 0 and would blow z = (x-mu)/sigma up to
        # huge values for any tiny drift. The floor only ever affects dims whose
        # real std is below it; dims with genuine variance keep their own std.
        #
        # Scale the floor by the layer's OWN typical variability: the median of
        # the non-zero per-dim stds. This is data-driven for any layer that has
        # real variance somewhere, and -- critically -- scales linearly with
        # that layer's own raw magnitude, so it does not break the scale-
        # invariance of z (see module comment above): a layer with huge raw
        # activations (action_head) and one with tiny raw activations (proprio)
        # each get a floor sized to their own units, not a shared absolute one.
        # Only when a layer is *fully* deterministic (no dim varies -> median
        # undefined) do we fall back to the signal magnitude (RMS of the clean
        # mean), since there is then no variance to borrow a scale from. `eps`
        # is used only as a literal division-by-zero guard for the fully-zero
        # edge case, never as the dominant floor. Fit uses clean calibration
        # only (trigger-agnostic).
        nonzero = raw_sigma[raw_sigma > eps]
        if nonzero.size > 0:
            scale = float(np.median(nonzero))
        else:
            scale = float(np.sqrt(np.mean(mu ** 2)))  # fully-deterministic fallback
        sigma_floor = max(1e-2 * scale, eps)
        sigma = np.maximum(raw_sigma, sigma_floor)

        zc = (C[test_idx] - mu) / sigma  # (n_test, D)
        zt = (T[test_idx] - mu) / sigma
        maha_clean = np.linalg.norm(zc, axis=1)  # (n_test,)
        maha_trig = np.linalg.norm(zt, axis=1)

        sq_clean += (zc ** 2).sum(axis=1)
        sq_trig += (zt ** 2).sum(axis=1)

        rows.append({
            "layer": label,
            "maha_clean": float(maha_clean.mean()),
            "maha_trig": float(maha_trig.mean()),
            "maha_delta": float(maha_trig.mean() - maha_clean.mean()),
        })

    group_auroc = auroc(np.sqrt(sq_trig), np.sqrt(sq_clean))
    return {
        "rows": rows,
        "auroc": group_auroc,
        "n_cal": len(cal_idx),
        "n_test": len(test_idx),
    }


# ======================================================================
# 2c. FFN neuron-level forensics (Stage 1: differential neuron flagging)
# ======================================================================
#
# Goal: within layers already flagged as divergent by the block/mlp-level
# metrics above, find INDIVIDUAL neurons (dims of the FFN intermediate
# activation, i.e. the input to mlp.down_proj) that shift for the real
# trigger but not for a benign control perturbation. Requires
# register_all_probe_hooks(..., hook_ffn_preact=True) so ffn_preact_names is
# populated, and a control (non-trigger) forward pass captured the same way
# as clean/triggered.

def compute_ffn_preact_diffs(clean_store, trig_store, control_store, ffn_preact_names):
    """Per-layer, per-neuron diff vectors for ONE scene.

    Returns {layer_name: (trig_diff, control_diff)} where each is a
    (intermediate_size,) float64 vector: pooled(trig or control) - pooled(clean).
    Layers missing from any of the three stores are skipped.
    """
    out = {}
    for name in ffn_preact_names:
        c = clean_store.get(name, [])
        t = trig_store.get(name, [])
        k = control_store.get(name, []) if control_store is not None else []
        if not c or not t or not k:
            continue
        pc = pool_tokens(c[0])
        pt = pool_tokens(t[0])
        pk = pool_tokens(k[0])
        out[name] = (pt - pc, pk - pc)
    return out


def aggregate_ffn_preact_diffs(per_scene_diffs):
    """Mean trig/control diff vectors across scenes, per layer.

    Parameters
    ----------
    per_scene_diffs : list of {layer_name: (trig_diff, control_diff)}, one
        dict per scene (as returned by compute_ffn_preact_diffs).

    Returns
    -------
    {layer_name: (mean_trig_diff, mean_control_diff)}
    """
    acc: dict = {}
    for scene in per_scene_diffs:
        for name, (td, cd) in scene.items():
            acc.setdefault(name, {"trig": [], "control": []})
            acc[name]["trig"].append(td)
            acc[name]["control"].append(cd)
    return {
        name: (np.mean(v["trig"], axis=0), np.mean(v["control"], axis=0))
        for name, v in acc.items()
    }


def select_candidate_layers(llm_agg_rows, top_n: int = 8):
    """Pick the FFN layers to drill into, reusing the existing whole-mlp L2
    metric as the coarse divergence signal (Stage 1, part 1).

    llm_agg_rows are the aggregated rows in agg["llm"] (from
    compute_layer_metrics against hook_groups.llm_names), which already
    include one row per f"llm.layer_{i:02d}.mlp" (the whole-FFN-output hook).
    We rank those rows by relative_l2 and return the corresponding
    ffn_preact layer names (f"llm.layer_{i:02d}.mlp.preact") for the top_n.
    """
    mlp_rows = [r for r in llm_agg_rows if r["layer"].endswith(".mlp")]
    mlp_rows.sort(key=lambda r: r["relative_l2"], reverse=True)
    return [r["layer"] + ".preact" for r in mlp_rows[:top_n]]


def flag_candidate_neurons(agg_diffs_by_layer, candidate_layers,
                           top_k: int = 25, ratio_thresh: float = 3.0, eps: float = 1e-8):
    """Flag neurons whose activation shifts for the real trigger but not for
    the control perturbation (Stage 1, part 3).

    Parameters
    ----------
    agg_diffs_by_layer : {layer_name: (mean_trig_diff, mean_control_diff)}
        as returned by aggregate_ffn_preact_diffs.
    candidate_layers   : layer names to inspect (e.g. from select_candidate_layers).
    top_k              : max neurons to keep per layer.
    ratio_thresh       : a neuron must have |trig_diff| / (|control_diff| + eps)
                         above this to be flagged -- i.e. the trigger moves it
                         much more than the control perturbation does.

    Returns
    -------
    list of {"layer": str, "neuron_idx": int, "trig_effect": float,
             "control_effect": float, "ratio": float}, sorted by trig_effect
    descending, across all candidate_layers.
    """
    rows = []
    for layer in candidate_layers:
        if layer not in agg_diffs_by_layer:
            _log(f"  flag_candidate_neurons: {layer} not in ffn_preact diffs -- skipping")
            continue
        trig_diff, control_diff = agg_diffs_by_layer[layer]
        trig_abs = np.abs(trig_diff)
        control_abs = np.abs(control_diff)
        ratio = trig_abs / (control_abs + eps)

        order = np.argsort(-trig_abs)
        picked = 0
        for idx in order:
            if ratio[idx] <= ratio_thresh:
                continue
            rows.append({
                "layer": layer,
                "neuron_idx": int(idx),
                "trig_effect": float(trig_abs[idx]),
                "control_effect": float(control_abs[idx]),
                "ratio": float(ratio[idx]),
            })
            picked += 1
            if picked >= top_k:
                break
        _log(f"  flag_candidate_neurons: {layer} -- {picked} neuron(s) "
             f"pass ratio_thresh={ratio_thresh} (out of {trig_abs.size})", force=True)

    rows.sort(key=lambda r: r["trig_effect"], reverse=True)
    return rows


def format_candidate_neurons_table(rows, title="Candidate backdoor neurons (Stage 1):"):
    if not rows:
        return [title, "  (none flagged)"]
    has_tokens = any("top_tokens" in r for r in rows)
    header = f"{'Layer':<22} | {'Neuron':>7} | {'Trig eff.':>10} | {'Ctrl eff.':>10} | {'Ratio':>8}"
    if has_tokens:
        header += " | Top tokens (logit lens, Stage 2)"
    lines = [title, header, "-" * len(header)]
    for r in rows:
        line = (
            f"{r['layer']:<22} | {r['neuron_idx']:>7} | {r['trig_effect']:>10.4f} | "
            f"{r['control_effect']:>10.4f} | {r['ratio']:>8.2f}"
        )
        if has_tokens:
            line += f" | {', '.join(r.get('top_tokens', []))}"
        lines.append(line)
    return lines


# ======================================================================
# 2d. Stage 2 -- logit lens / vocab projection for flagged neurons
# ======================================================================
#
# Adapted from mechanistic-steering-vlas's
# src/ffn_value_vectors/extract.py::extract_value_vectors +
# project_to_vocab_top_tokens_streaming. That code sweeps every FFN neuron
# in the model (~32 layers x 11008 = ~350k rows) through the LM head, which
# is unnecessary and memory-heavy when we only need labels for the few
# hundred neurons Stage 1 flagged. This computes the same thing --
# down_proj's per-neuron "value vector" projected through lm_head, then
# top-k vocab tokens -- but only for the flagged (layer, neuron_idx) pairs.

def _ffn_layer_num(layer_name: str) -> int:
    """"llm.layer_05.mlp.preact" -> 5"""
    rest = layer_name[len("llm.layer_"):]
    num, _, _ = rest.partition(".")
    return int(num)


def label_candidate_neurons_with_tokens(model, processor, candidate_rows,
                                        top_k: int = 10, action_bins: int = 256,
                                        action_min: float = -1.0, action_max: float = 1.0):
    """Stage 2: label each Stage-1-flagged neuron with its top-k vocab tokens.

    Mutates and returns candidate_rows, adding "top_tokens" (list[str]) and
    "top_token_ids" (list[int]) to each row. Requires the model already
    loaded by initialize_model() (no separate model load, unlike the
    original mechanistic-steering-vlas script).

    Caveat (per the forensics plan): treat these labels as descriptive only
    -- a neuron's projected "meaning" can drift across checkpoints/triggers,
    not a stable fingerprint.
    """
    if not candidate_rows:
        return candidate_rows

    from prismatic.vla.action_tokenizer import ActionTokenizer

    decoder_layers = model.language_model.model.layers
    tokenizer = processor.tokenizer
    vocab_size = tokenizer.vocab_size
    action_token_start = vocab_size - action_bins
    action_tokenizer = ActionTokenizer(
        tokenizer, bins=action_bins, min_action=action_min, max_action=action_max)

    by_layer: dict = {}
    for i, row in enumerate(candidate_rows):
        by_layer.setdefault(_ffn_layer_num(row["layer"]), []).append(i)

    lm_head_weight = model.language_model.lm_head.weight.detach()

    with torch.no_grad():
        for layer_num, row_idxs in by_layer.items():
            down_proj = decoder_layers[layer_num].mlp.down_proj
            p_dtype, p_device = down_proj.weight.dtype, down_proj.weight.device
            neuron_ids = [candidate_rows[i]["neuron_idx"] for i in row_idxs]

            one_hot = torch.zeros(
                len(neuron_ids), down_proj.in_features, dtype=p_dtype, device=p_device)
            for r, nid in enumerate(neuron_ids):
                one_hot[r, nid] = 1.0
            value_vecs = down_proj(one_hot)  # (n_neurons, hidden_dim)

            embedding_matrix = lm_head_weight.to(device=p_device, dtype=torch.float32)
            logits = value_vecs.float() @ embedding_matrix.T  # (n_neurons, vocab)
            top_ids = torch.topk(logits, k=top_k, dim=1).indices.cpu()

            for r, row_i in enumerate(row_idxs):
                token_strs = []
                for tid in top_ids[r].tolist():
                    if tid >= action_token_start:
                        action_val = action_tokenizer.decode_token_ids_to_actions(
                            np.array([tid]))[0]
                        token_strs.append(f"[action: {action_val:.3f}]")
                    else:
                        token_strs.append(repr(tokenizer.decode([tid])))
                candidate_rows[row_i]["top_tokens"] = token_strs
                candidate_rows[row_i]["top_token_ids"] = top_ids[r].tolist()

    _log(f"label_candidate_neurons_with_tokens: labeled {len(candidate_rows)} "
         f"neuron(s) across {len(by_layer)} layer(s)", force=True)
    return candidate_rows


# ======================================================================
# 2e. Stage 3 -- causal ablation (prove the flagged neurons are responsible)
# ======================================================================
#
# Adapted from mechanistic-steering-vlas's
# src/libero_experiments/hooks.py::apply_gate_proj_hooks +
# src/libero_experiments/interventions.py::load_intervention_dict. Both are
# reused conceptually unchanged (same forward-hook-on-down_proj trick, same
# yaml.safe_load + int-cast-keys loading), with one adaptation: the original
# hooks.py takes a FLAT index (layer_idx * intermediate_size + neuron_idx)
# and assumes one global intermediate_size for every layer, recovering
# (layer, neuron) via //  and %. Our own Stage 1 output (and
# write_candidate_neurons_yaml) is already keyed by layer, so this version
# takes {layer_idx: [neuron_idx, ...]} directly and reads each layer's own
# down_proj.in_features -- avoids the flat-index/layer-dict mismatch in the
# original repo (flagged when Stage 1/2 were implemented) instead of
# reproducing it.

def load_candidate_neurons_yaml(path, dict_name: str = "stage1_candidates"):
    """{layer_idx: [neuron_idx, ...]} from a YAML file written by
    write_candidate_neurons_yaml (run_libero_probe.py). Same idea as
    mechanistic-steering-vlas's load_intervention_dict: yaml.safe_load, then
    cast string keys to int.
    """
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        all_dicts = yaml.safe_load(f) or {}

    if dict_name not in all_dicts:
        available = ", ".join(sorted(str(k) for k in all_dicts.keys()))
        raise KeyError(f"Unknown intervention dict {dict_name!r}. Available: {available}")

    raw = all_dicts[dict_name] or {}
    return {int(k): [int(n) for n in (v or [])] for k, v in raw.items()}


def apply_neuron_ablation_hooks(model, layer_to_neurons: dict, coef: float = 0.0):
    """Silence specific FFN neurons: overwrite the down_proj input at the
    flagged neuron indices (to `coef`) before the down-projection runs, on
    every layer in layer_to_neurons.

    coef=0.0 (default) zero-ablates -- the standard "does the effect
    disappear" causal test. Returns the list of hook handles; caller must
    call .remove() on each when done (see run_libero_ablation_eval.py for
    the register/run/remove pattern around a full LIBERO rollout).
    """
    print("\nNeurons selected for ablation:\n")
    for layer, neurons in layer_to_neurons.items():
        print(f"  Layer {layer}: {len(neurons)} neuron(s)")
    print()

    def _make_hook(neuron_ids, coef_val):
        def hook_fn(module, inp, out):
            modified_input = inp[0]
            modified_input[..., neuron_ids] = coef_val
            return F.linear(modified_input, module.weight, module.bias)
        return hook_fn

    decoder_layers = model.language_model.model.layers
    handles = []
    for layer_idx, neuron_ids in layer_to_neurons.items():
        if not neuron_ids:
            continue
        down_proj = decoder_layers[layer_idx].mlp.down_proj
        handles.append(down_proj.register_forward_hook(_make_hook(neuron_ids, coef)))
    return handles


# ======================================================================
# 2f. Classic logit lens -- project REAL activations (not fixed neuron
# weights) through lm_head, clean vs triggered, per layer.
# ======================================================================
#
# Unlike Stage 2 (which projects a neuron's fixed value vector, no input
# involved), this projects the actual pooled hidden state from a real
# forward pass -- the classic logit-lens technique. Question: does the
# model's "currently predicted token" visibly diverge between clean and
# triggered inputs, and at which layer does that divergence first appear?

def logit_lens_activation_divergence(model, clean_store, trig_store, layer_names, top_k: int = 10):
    """For each layer name (whole-block outputs, e.g. hook_groups.llm_names
    entries like "llm.layer_05"), project the real pooled clean and
    triggered hidden states through lm_head and compare top-k tokens.

    Returns a list of {layer, clean_top_ids, trig_top_ids, overlap} rows,
    where overlap = number of tokens shared between the two top-k lists
    (top_k = identical top-k sets, 0 = completely disjoint).
    """
    lm_head_weight = model.language_model.lm_head.weight.detach()
    device, dtype = lm_head_weight.device, torch.float32
    embedding_matrix = lm_head_weight.to(dtype=dtype)

    rows = []
    with torch.no_grad():
        for name in layer_names:
            c = clean_store.get(name, [])
            t = trig_store.get(name, [])
            if not c or not t:
                continue
            pc = torch.tensor(pool_tokens(c[0]), dtype=dtype, device=device)
            pt = torch.tensor(pool_tokens(t[0]), dtype=dtype, device=device)

            clean_logits = pc @ embedding_matrix.T
            trig_logits = pt @ embedding_matrix.T
            clean_top = torch.topk(clean_logits, top_k).indices.tolist()
            trig_top = torch.topk(trig_logits, top_k).indices.tolist()
            overlap = len(set(clean_top) & set(trig_top))

            rows.append({
                "layer": name,
                "clean_top_ids": clean_top,
                "trig_top_ids": trig_top,
                "overlap": overlap,
                "top_k": top_k,
            })
    return rows


def decode_token_ids(model, processor, token_ids, action_bins: int = 256,
                     action_min: float = -1.0, action_max: float = 1.0):
    """Decode a list of vocab token IDs to readable strings, using the same
    action-bin convention as label_candidate_neurons_with_tokens."""
    from prismatic.vla.action_tokenizer import ActionTokenizer

    tokenizer = processor.tokenizer
    vocab_size = tokenizer.vocab_size
    action_token_start = vocab_size - action_bins
    action_tokenizer = ActionTokenizer(
        tokenizer, bins=action_bins, min_action=action_min, max_action=action_max)

    strs = []
    for tid in token_ids:
        if tid >= action_token_start:
            action_val = action_tokenizer.decode_token_ids_to_actions(np.array([tid]))[0]
            strs.append(f"[action: {action_val:.3f}]")
        else:
            strs.append(repr(tokenizer.decode([tid])))
    return strs


def format_logit_lens_divergence_table(rows, title="Classic logit lens: clean vs. triggered (real activations)"):
    if not rows:
        return [title, "  (no data)"]
    lines = [
        title,
        f"{'Layer':<15} | {'Overlap (of top_k)':>18}",
        f"{'-'*15}-+-{'-'*18}",
    ]
    for r in sorted(rows, key=lambda r: r["overlap"]):
        lines.append(f"{r['layer']:<15} | {r['overlap']:>18}")
    return lines


# ======================================================================
# 2g. Combined real-activation projection for the flagged-neuron GROUP
# ======================================================================
#
# Neither Stage 2 (one-hot value vectors, no real input) nor the classic
# logit lens above (the WHOLE pooled hidden state) asks this question:
# "using their REAL captured activations for one real scene, what do just
# the ~200 flagged neurons -- acting together, at their actual firing
# strengths -- push the model toward?" This masks every other neuron to
# zero, keeps only the flagged ones at their real pooled value, runs the
# result through that layer's own down_proj, then lm_head.

def mask_to_flagged_neurons(pooled_vec, neuron_ids):
    """Zero out every position except neuron_ids. pooled_vec: (intermediate,)."""
    masked = np.zeros_like(pooled_vec)
    masked[neuron_ids] = pooled_vec[neuron_ids]
    return masked


def project_masked_activation_to_vocab(model, layer_num, masked_vec, top_k: int = 10):
    """Run a masked (flagged-neurons-only) real activation vector through
    that layer's down_proj, then lm_head. Returns top-k token IDs."""
    down_proj = model.language_model.model.layers[layer_num].mlp.down_proj
    p_dtype, p_device = down_proj.weight.dtype, down_proj.weight.device
    x = torch.tensor(masked_vec, dtype=p_dtype, device=p_device).unsqueeze(0)
    with torch.no_grad():
        out = down_proj(x)
        lm_head_weight = model.language_model.lm_head.weight.detach().to(dtype=torch.float32)
        logits = out.float() @ lm_head_weight.T
        top_ids = torch.topk(logits[0], top_k).indices.tolist()
    return top_ids


# ======================================================================
# 3. Reporting
# ======================================================================

def _table_row_label(row):
    occ = f"#cam{row['occurrence']}" if "occurrence" in row else ""
    label = f"{row['layer']}{occ}"
    # llm.layer_05 / llm.layer_05.self_attn / llm.layer_05.mlp
    if label.startswith("llm.layer_"):
        rest = label[len("llm.layer_"):]
        num, _, suffix = rest.partition(".")
        try:
            short = f"{int(num):>2}"
        except ValueError:
            return label
        return f"{short}.{suffix}" if suffix else short
    return label


def _table_fmt_float(value):
    if value != value:  # NaN
        return "N/A"
    return f"{value:.4f}"


def _format_table(rows, title):
    """Layer drift table: pooled L2 / Rel L2 / cosine."""
    col_layer = "Layer"
    col_l2 = "L2 (pooled)"
    col_rel = "Rel L2"
    col_cos = "Cosine"
    labels = [_table_row_label(r) for r in rows]
    l2_vals = [_table_fmt_float(r["l2"]) for r in rows]
    rel_vals = [_table_fmt_float(r.get("relative_l2", float("nan"))) for r in rows]
    cos_vals = [_table_fmt_float(r["cosine_dist"]) for r in rows]

    w_layer = max(len(col_layer), max((len(l) for l in labels), default=0))
    w_l2 = max(len(col_l2), max((len(v) for v in l2_vals), default=0))
    w_rel = max(len(col_rel), max((len(v) for v in rel_vals), default=0))
    w_cos = max(len(col_cos), max((len(v) for v in cos_vals), default=0))

    sep = f"{'-' * w_layer}-+-{'-' * w_l2}-+-{'-' * w_rel}-+-{'-' * w_cos}"
    lines = [
        title,
        (f"{col_layer:<{w_layer}} | {col_l2:>{w_l2}} | {col_rel:>{w_rel}} | "
         f"{col_cos:>{w_cos}}"),
        sep,
    ]
    for label, l2, rel, cos in zip(labels, l2_vals, rel_vals, cos_vals):
        lines.append(
            f"{label:<{w_layer}} | {l2:>{w_l2}} | {rel:>{w_rel}} | {cos:>{w_cos}}"
        )
    return lines


def _format_maha_table(maha_result, title):
    """Format a Mahalanobis result dict as a 4-column ASCII table."""
    rows = maha_result.get("rows", [])
    if not rows:
        return [title, "  (no data)"]

    col_layer = "Layer"
    col_clean = "Maha(clean)"
    col_trig  = "Maha(trig)"
    col_delta = "Delta"

    labels      = [r["layer"] for r in rows]
    clean_vals  = [_table_fmt_float(r["maha_clean"]) for r in rows]
    trig_vals   = [_table_fmt_float(r["maha_trig"])  for r in rows]
    delta_vals  = [_table_fmt_float(r["maha_delta"]) for r in rows]

    w_layer = max(len(col_layer), max((len(l) for l in labels), default=0))
    w_cl    = max(len(col_clean), max((len(v) for v in clean_vals), default=0))
    w_tr    = max(len(col_trig),  max((len(v) for v in trig_vals),  default=0))
    w_de    = max(len(col_delta), max((len(v) for v in delta_vals), default=0))

    sep = f"{'-'*w_layer}-+-{'-'*w_cl}-+-{'-'*w_tr}-+-{'-'*w_de}"
    lines = [
        title,
        f"{col_layer:<{w_layer}} | {col_clean:>{w_cl}} | {col_trig:>{w_tr}} | {col_delta:>{w_de}}",
        sep,
    ]
    for lab, cl, tr, de in zip(labels, clean_vals, trig_vals, delta_vals):
        lines.append(f"{lab:<{w_layer}} | {cl:>{w_cl}} | {tr:>{w_tr}} | {de:>{w_de}}")

    n_cal  = maha_result.get("n_cal", "?")
    n_test = maha_result.get("n_test", "?")
    auc    = maha_result.get("auroc", float("nan"))
    lines.append(f"  cal={n_cal} test={n_test}  group-AUROC={_table_fmt_float(auc)}")
    return lines


def format_summary(results: dict):
    """Build human-readable summary table for within-sample probe results."""
    lines = []
    if "within" in results:
        lines.append("=== WITHIN-SAMPLE (same scene: clean vs triggered) ===")
        lines.append(format_summary_section(results["within"]))
    text = "\n".join(lines) if lines else "(no comparison modes enabled)"
    _log(f"format_summary: built table ({len(lines)} lines)")
    return text


def format_summary_section(metrics: dict):
    """Format one metrics bundle (within or cross aggregated)."""
    lines = []
    if metrics.get("llm"):
        lines.extend(_format_table(metrics["llm"], "LLM decoder blocks (full sequence mean):"))
    lines.append("")
    act = metrics.get("action", {})
    lines.append(f"Action L2 (Frobenius): {act.get('l2_frobenius', 0):.4f}")
    lines.append(f"Action cosine distance: {act.get('cosine_dist', 0):.4f}")
    js = act.get("js_divergence")
    lines.append(f"Action JS divergence : "
                 f"{'N/A (continuous L1 head)' if js is None else f'{js:.4f}'}")

    for key, title in (
        ("projector", "Projector layers:"),
        ("proprio", "Proprio projector layers:"),
        ("action_head", "Action head layers:"),
        ("vision", "Vision ViT blocks:"),
        ("noisy_action", "Noisy action projector:"),
    ):
        if metrics.get(key):
            lines.append("")
            lines.extend(_format_table(metrics[key], title))

    # Mahalanobis section (only present when run with --cal_fraction > 0)
    if metrics.get("mahalanobis"):
        lines.append("")
        lines.append("=== Mahalanobis Detection (diagonal, clean-calibrated) ===")
        lines.append("  Drift tables above: pooled L2 / Rel L2 / cosine (mean over tokens).")
        lines.append("  Mahalanobis uses pooled, clean-calibrated z-scores for detection AUROC.")
        lines.append("  AUROC: P(Maha(trig) > Maha(clean)) on held-out test scenes.")
        for grp_key, grp_title in (
            ("llm",         "LLM decoder blocks:"),
            ("projector",   "Projector layers:"),
            ("proprio",     "Proprio projector:"),
            ("action_head", "Action head layers:"),
            ("vision",      "Vision ViT blocks:"),
            ("noisy_action","Noisy action projector:"),
        ):
            maha = metrics["mahalanobis"].get(grp_key)
            if maha and maha.get("rows"):
                lines.append("")
                lines.extend(_format_maha_table(maha, grp_title))
        # Summary AUROC line across all groups
        aurocs = {
            g: metrics["mahalanobis"][g]["auroc"]
            for g in metrics["mahalanobis"]
            if metrics["mahalanobis"].get(g)
            and metrics["mahalanobis"][g].get("auroc") == metrics["mahalanobis"][g].get("auroc")
        }
        if aurocs:
            lines.append("")
            lines.append("  Group-level detection AUROC summary:")
            for g, auc in aurocs.items():
                lines.append(f"    {g:<15}: {_table_fmt_float(auc)}")

    return "\n".join(lines)
