"""
trial_error/run_vocab_calibration_v2_probe.py

Turns the calibration-based logit-lens idea (run_vocab_calibration_probe.py)
into an actual detector with an AUROC, and runs the same specificity check
already applied to the other two detectors (Mahalanobis, cross-attention):
does it distinguish the real trigger from a harmless noise-patch control and
a genuinely different task (object), or does it fire on anything?

Calibration: average the pooled activation across held-out CLEAN calibration
scenes, project through lm_head ONCE per layer -> a fixed "clean reference"
logit vector over the full vocabulary (not just top-10). Never sees a
trigger, control, or object input.

Scoring any scene (clean-test, trig, control, object): project its own
activation through lm_head -> full logit vector -> cosine similarity against
the calibration reference. anomaly = 1 - cosine_similarity. Continuous,
per-layer, and combinable across layers (sum of anomaly scores) into one
detector score, same spirit as Mahalanobis's chi-squared combination.

Reuses the already-collected pooled activations in
trial_error/probe_logs/tsne_activations_v2_block.npz (20 scenes x 4
conditions x 32 layers) -- only needs the block checkpoint's lm_head weight.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append("/home/grads/nsamptur/vla_bkd_def/LIBERO")
from trial_error.run_libero_probe import GenerateConfig, initialize_model

CHECKPOINT = "/home/grads/nsamptur/vla_bkd_def/BadVLA/vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt"
N_CAL = 10  # first 10 clean scenes -> calibration reference; remaining 10 clean scenes are held-out test


def auroc(pos, neg):
    labels = [1] * len(pos) + [0] * len(neg)
    scores = list(pos) + list(neg)
    return float(roc_auc_score(labels, scores))


def main():
    cfg = GenerateConfig(pretrained_checkpoint=CHECKPOINT, probe_trigger="block", task_suite_name="libero_goal", num_trials_per_task=1)
    model, *_ = initialize_model(cfg)
    W = model.language_model.lm_head.weight.detach().to(dtype=torch.float32)  # (vocab, hidden)
    device = W.device

    data = np.load("trial_error/probe_logs/tsne_activations_v2_block.npz")
    conditions = ["trig", "control", "object"]

    out = {"rows": [], "n_cal": N_CAL}
    anomaly_sum = {"clean_test": None, "trig": None, "control": None, "object": None}

    for l in range(32):
        ck = f"clean_L{l:02d}"
        if ck not in data:
            continue
        clean_all = data[ck]  # (20, 4096)
        cal_vec = clean_all[:N_CAL].mean(axis=0)
        clean_test = clean_all[N_CAL:]  # held-out clean scenes

        with torch.no_grad():
            cal_logits = torch.tensor(cal_vec, dtype=torch.float32, device=device) @ W.T  # (vocab,)
            cal_logits = cal_logits / cal_logits.norm()

            def anomaly_scores(vecs):
                V = torch.tensor(vecs, dtype=torch.float32, device=device) @ W.T  # (n, vocab)
                V = V / V.norm(dim=1, keepdim=True)
                cos_sim = V @ cal_logits  # (n,)
                return (1.0 - cos_sim).cpu().numpy()

            a_clean_test = anomaly_scores(clean_test)
            a_by_cond = {cond: anomaly_scores(data[f"{cond}_L{l:02d}"]) for cond in conditions}

        for key, arr in [("clean_test", a_clean_test)] + list(a_by_cond.items()):
            if anomaly_sum[key] is None:
                anomaly_sum[key] = arr.copy()
            else:
                anomaly_sum[key] += arr

        row = {"layer": l, "clean_test_mean_anomaly": float(a_clean_test.mean())}
        for cond in conditions:
            row[f"{cond}_mean_anomaly"] = float(a_by_cond[cond].mean())
            row[f"{cond}_layer_auroc"] = auroc(a_by_cond[cond], a_clean_test)
        out["rows"].append(row)

    group_auroc = {cond: auroc(anomaly_sum[cond], anomaly_sum["clean_test"]) for cond in conditions}
    out["group_auroc"] = group_auroc

    print(f"{'Layer':<6} | {'clean anomaly':>13} | {'trig anomaly':>13} | {'ctrl anomaly':>13} | {'obj anomaly':>12} | {'trig AUC':>9} | {'ctrl AUC':>9} | {'obj AUC':>8}")
    for row in out["rows"]:
        print(f"{row['layer']:<6} | {row['clean_test_mean_anomaly']:13.5f} | {row['trig_mean_anomaly']:13.5f} | {row['control_mean_anomaly']:13.5f} | {row['object_mean_anomaly']:12.5f} | {row['trig_layer_auroc']:9.3f} | {row['control_layer_auroc']:9.3f} | {row['object_layer_auroc']:8.3f}")

    print(f"\nGroup AUROC (all 32 layers combined, summed anomaly scores):")
    for cond, v in group_auroc.items():
        print(f"  {cond}: {v:.4f}")

    out_path = Path("trial_error/probe_logs/vocab_calibration_v2_probe.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
