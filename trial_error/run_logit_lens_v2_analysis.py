"""
trial_error/run_logit_lens_v2_analysis.py

Classic logit lens (real pooled activation -> lm_head -> top-k vocab),
applied to the SAME 4-condition data already collected in
trial_error/probe_logs/tsne_activations_v2_{block,mug}.npz
(clean, trig, control, object -- 20 scenes each, all 32 whole-block layers).

For each checkpoint and each layer, computes top-k overlap (out of top_k=10)
between clean and each of {trig, control, object}, per scene, then averages
across scenes. This is the same overlap metric used to build the first
logit-lens artifact (paired_probe.py::logit_lens_activation_divergence),
just run here directly on the cached pooled vectors instead of re-doing
forward passes, and extended to all 3 non-clean conditions instead of only
"triggered".

Only needs each checkpoint's own lm_head weight matrix, so the full model is
loaded once per checkpoint (no environment / forward passes needed here).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # BadVLA's own experiments/ must win over the openvla-oft editable install
sys.path.append("/home/grads/nsamptur/vla_bkd_def/LIBERO")  # broken editable install, MAPPING is empty
from trial_error.run_libero_probe import GenerateConfig, initialize_model

CHECKPOINTS = {
    "block": "/home/grads/nsamptur/vla_bkd_def/BadVLA/vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt",
    "mug": "/home/grads/nsamptur/vla_bkd_def/BadVLA/vla-scripts/goal_mug/trigger_sec/goal_mug_stage2_30000_chkpt",
}
CONDITIONS = ["trig", "control", "object"]
TOP_K = 10


def overlap_curve(lm_head_weight, data, other_cond, n_layers=32, top_k=TOP_K):
    """Per-layer mean top-k overlap between clean and `other_cond`, averaged
    over all scenes present for that layer."""
    device, dtype = lm_head_weight.device, torch.float32
    W = lm_head_weight.to(dtype=dtype)

    means = []
    for l in range(n_layers):
        ck, ok = f"clean_L{l:02d}", f"{other_cond}_L{l:02d}"
        if ck not in data or ok not in data:
            means.append(float("nan"))
            continue
        C = torch.tensor(data[ck], dtype=dtype, device=device)  # (n_scenes, 4096)
        O = torch.tensor(data[ok], dtype=dtype, device=device)
        n = min(C.shape[0], O.shape[0])
        with torch.no_grad():
            clean_logits = C[:n] @ W.T  # (n, vocab)
            other_logits = O[:n] @ W.T
            clean_top = torch.topk(clean_logits, top_k, dim=1).indices  # (n, top_k)
            other_top = torch.topk(other_logits, top_k, dim=1).indices
        overlaps = []
        for i in range(n):
            overlaps.append(len(set(clean_top[i].tolist()) & set(other_top[i].tolist())))
        means.append({"mean": float(np.mean(overlaps)), "min": float(np.min(overlaps)), "max": float(np.max(overlaps))})
    return means


def main():
    results = {}
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"[logit-lens-v2] loading checkpoint for {ckpt_name} lm_head ...")
        cfg = GenerateConfig(
            pretrained_checkpoint=ckpt_path,
            probe_trigger=ckpt_name,
            task_suite_name="libero_goal",
            num_trials_per_task=1,
        )
        model, *_ = initialize_model(cfg)
        lm_head_weight = model.language_model.lm_head.weight.detach()

        data = np.load(f"trial_error/probe_logs/tsne_activations_v2_{ckpt_name}.npz")
        curves = {}
        for cond in CONDITIONS:
            curves[cond] = overlap_curve(lm_head_weight, data, cond)
            print(f"  {ckpt_name} clean-vs-{cond}: {curves[cond]}")
        results[ckpt_name] = curves

        del model
        torch.cuda.empty_cache()

    out_path = Path("trial_error/probe_logs/logit_lens_v2_overlap.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
