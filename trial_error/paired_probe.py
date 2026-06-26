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

    @property
    def all_names(self) -> list[str]:
        return (
            self.vision_names
            + self.projector_names
            + self.proprio_names
            + self.action_head_names
            + self.noisy_action_names
            + self.llm_names
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
    _log(f"  => LLM: {len(groups.llm_names)} decoder blocks hooked")

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
            row = {"layer": name, "l2": l2_distance(pc, pt),
                   "cosine_dist": cosine_distance(pc, pt)}
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
            _log(f"  top-3 {group} by L2 drift:")
            for r in by_l2[:3]:
                occ = f" cam{r['occurrence']}" if "occurrence" in r else ""
                _log(f"    {r['layer']}{occ}: L2={r['l2']:.4f}  cosine={r['cosine_dist']:.4f}")
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
# 3. Reporting
# ======================================================================

def _table_row_label(row):
    occ = f"#cam{row['occurrence']}" if "occurrence" in row else ""
    label = f"{row['layer']}{occ}"
    if label.startswith("llm.layer_"):
        return f"{int(label.split('_')[-1]):>2}"
    return label


def _table_fmt_float(value):
    if value != value:  # NaN
        return "N/A"
    return f"{value:.4f}"


def _format_table(rows, title):
    col_layer, col_l2, col_cos = "Layer", "L2 Distance", "Cosine Distance"
    labels = [_table_row_label(r) for r in rows]
    l2_vals = [_table_fmt_float(r["l2"]) for r in rows]
    cos_vals = [_table_fmt_float(r["cosine_dist"]) for r in rows]

    w_layer = max(len(col_layer), max((len(l) for l in labels), default=0))
    w_l2 = max(len(col_l2), max((len(v) for v in l2_vals), default=0))
    w_cos = max(len(col_cos), max((len(v) for v in cos_vals), default=0))

    sep = f"{'-' * w_layer}-+-{'-' * w_l2}-+-{'-' * w_cos}"
    lines = [
        title,
        f"{col_layer:<{w_layer}} | {col_l2:>{w_l2}} | {col_cos:>{w_cos}}",
        sep,
    ]
    for label, l2, cos in zip(labels, l2_vals, cos_vals):
        lines.append(f"{label:<{w_layer}} | {l2:>{w_l2}} | {cos:>{w_cos}}")
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
    return "\n".join(lines)
