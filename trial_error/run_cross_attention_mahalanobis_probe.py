"""
trial_error/run_cross_attention_mahalanobis_probe.py

Fixes the one real weakness of run_cross_attention_probe.py: that script only
scored two hand-picked patch regions (the known trigger location + one control
region), which requires knowing where a trigger is before you can look for it.

This script instead computes, for EVERY vision-patch key position (all 512 =
256 patches x 2 cameras), the attention share it receives from the action-query
rows -- then clean-calibrates a per-patch, per-layer mean/std (mu/sigma) from
held-out clean scenes ONLY (same diagonal Mahalanobis recipe as
paired_probe.py::compute_mahalanobis_by_group, just applied to attention shares
per patch instead of activation values per hidden dim). Any scene (clean or
triggered) can then be scored patch-by-patch: which patches z-score above a
threshold, and does that flag concentrate on the trigger's real location or
stay diffuse (consistent with the earlier "global redistribution, not local
spotlight" finding)? A scene-level chi-squared-style combined score (summed
squared z across all patches, per layer) also gives a detection AUROC that
needs zero prior knowledge of trigger geometry.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append("/home/grads/nsamptur/vla_bkd_def/LIBERO")

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark
from sklearn.metrics import roc_auc_score

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import get_image_resize_size
from experiments.robot.openvla_utils import get_processor, get_action_head, get_proprio_projector

from trial_error.run_libero_probe import GenerateConfig, add_trigger_img, _warmup_and_prepare_observation, check_unnorm_key
from trial_error.paired_probe import set_probe_quiet
from trial_error.run_cross_attention_probe import load_model_eager, predict_action_with_attention, NUM_PATCHES_PER_IMAGE, TRIGGER_PATCH_IDX

N_PATCHES_TOTAL = 2 * NUM_PATCHES_PER_IMAGE  # 512: full_image patches [0:256], wrist_image patches [256:512]


def per_patch_attention_vector(attn_layer, query_start, query_end):
    """attn_layer: (1, heads, seq, seq). Returns a 512-dim vector: mean attention
    weight (over heads and action-query rows) landing on each of the 512 vision
    patches (positions 1..512, i.e. full_image then wrist_image, per the model's
    known token layout)."""
    sub = attn_layer[0, :, query_start:query_end, 1:1 + N_PATCHES_TOTAL]  # (heads, n_query, 512)
    return sub.mean(dim=(0, 1)).float().cpu().numpy()  # (512,)


def fit_mu_sigma(C, eps=1e-6):
    mu = C.mean(axis=0)
    raw_sigma = C.std(axis=0)
    nonzero = raw_sigma[raw_sigma > eps]
    scale = float(np.median(nonzero)) if nonzero.size > 0 else float(np.sqrt(np.mean(mu ** 2)))
    sigma_floor = max(1e-2 * scale, eps)
    sigma = np.maximum(raw_sigma, sigma_floor)
    return mu, sigma


def auroc(pos, neg):
    labels = [1] * len(pos) + [0] * len(neg)
    scores = list(pos) + list(neg)
    return float(roc_auc_score(labels, scores))


@draccus.wrap()
def run_probe(cfg: GenerateConfig) -> None:
    assert cfg.probe_trigger == "block", "this probe only supports the block trigger (known patch geometry, for validation)"
    N_EPISODES_PER_TASK = 3  # 10 libero_goal tasks x 3 = up to 30 scenes

    cfg.task_suite_name = "libero_goal"
    processor = get_processor(cfg)
    vla = load_model_eager(cfg)
    check_unnorm_key(cfg, vla)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)

    task_suite = benchmark.get_benchmark_dict()["libero_goal"]()
    num_tasks = task_suite.n_tasks
    n_layers = vla.config.text_config.num_hidden_layers

    set_probe_quiet(True)
    # clean_vecs[layer] / trig_vecs[layer] = list of 512-dim per-patch attention vectors, one per scene
    clean_vecs = {l: [] for l in range(n_layers)}
    trig_vecs = {l: [] for l in range(n_layers)}
    n_scenes = 0

    try:
        for task_id in tqdm.tqdm(range(num_tasks)):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)
            for episode_idx in range(min(N_EPISODES_PER_TASK, len(initial_states))):
                obs = _warmup_and_prepare_observation(env, cfg, initial_states[episode_idx], resize_size)
                clean_obs = copy.deepcopy(obs)
                trig_obs = copy.deepcopy(obs)
                trig_obs["full_image"] = add_trigger_img(trig_obs["full_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)
                trig_obs["wrist_image"] = add_trigger_img(trig_obs["wrist_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)

                c_attn, qs, qe = predict_action_with_attention(cfg, vla, processor, clean_obs, task_description, action_head, proprio_projector)
                t_attn, qs2, qe2 = predict_action_with_attention(cfg, vla, processor, trig_obs, task_description, action_head, proprio_projector)
                assert qs == qs2 and qe == qe2

                for l in range(n_layers):
                    clean_vecs[l].append(per_patch_attention_vector(c_attn[l], qs, qe))
                    trig_vecs[l].append(per_patch_attention_vector(t_attn[l], qs, qe))

                del c_attn, t_attn
                torch.cuda.empty_cache()
                n_scenes += 1
    finally:
        set_probe_quiet(False)

    print(f"\n[cross-attn-maha-probe] {n_scenes} scenes")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n_scenes)
    n_cal = n_scenes // 2
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

    import json
    out = {"n_scenes": n_scenes, "n_cal": len(cal_idx), "n_test": len(test_idx), "rows": []}
    sq_clean_test = np.zeros(len(test_idx))
    sq_trig = np.zeros(n_scenes)
    flag_counts = np.zeros(N_PATCHES_TOTAL)  # how often each patch is flagged (|z|>3) in triggered scenes, summed over layers/scenes

    for l in range(n_layers):
        C = np.stack(clean_vecs[l])  # (n_scenes, 512)
        T = np.stack(trig_vecs[l])
        mu, sigma = fit_mu_sigma(C[cal_idx])

        z_clean_test = (C[test_idx] - mu) / sigma  # (n_test, 512)
        z_trig = (T - mu) / sigma  # (n_scenes, 512) -- score ALL triggered scenes (calibration only used clean)

        sq_clean_test += (z_clean_test ** 2).sum(axis=1)
        sq_trig += (z_trig ** 2).sum(axis=1)

        flags = (np.abs(z_trig) > 3.0)  # (n_scenes, 512) boolean
        flag_counts += flags.sum(axis=0)

        row = {
            "layer": l,
            "maha_clean_test_mean": float(np.linalg.norm(z_clean_test, axis=1).mean()),
            "maha_trig_mean": float(np.linalg.norm(z_trig, axis=1).mean()),
            "layer_auroc": auroc(np.sqrt((z_trig ** 2).sum(axis=1)), np.sqrt((z_clean_test ** 2).sum(axis=1))),
        }
        out["rows"].append(row)

    group_auroc = auroc(np.sqrt(sq_trig), np.sqrt(sq_clean_test))
    out["group_auroc_all_layers_combined"] = group_auroc

    print(f"{'Layer':<6} | {'maha(clean-test)':>16} | {'maha(trig)':>12} | {'layer AUROC':>12}")
    for row in out["rows"]:
        print(f"{row['layer']:<6} | {row['maha_clean_test_mean']:16.3f} | {row['maha_trig_mean']:12.3f} | {row['layer_auroc']:12.3f}")
    print(f"\nGroup AUROC (all 32 layers combined, chi-squared style): {group_auroc:.4f}")

    # Which patches get flagged most in triggered scenes, summed across all layers/scenes?
    top_flagged = np.argsort(-flag_counts)[:15]
    print(f"\nTop 15 most-flagged patch positions across all layers/scenes (|z|>3 in triggered runs):")
    print(f"{'patch_idx':>10} | {'camera':>12} | {'is_known_trigger_patch':>22} | {'flag_count':>10}")
    for p in top_flagged:
        camera = "full_image" if p < NUM_PATCHES_PER_IMAGE else "wrist_image"
        local_idx = p if p < NUM_PATCHES_PER_IMAGE else p - NUM_PATCHES_PER_IMAGE
        is_trig = local_idx in TRIGGER_PATCH_IDX
        print(f"{p:>10} | {camera:>12} | {str(is_trig):>22} | {int(flag_counts[p]):>10}")

    out["flag_counts"] = flag_counts.tolist()
    out["known_trigger_patch_idx_full_image"] = TRIGGER_PATCH_IDX
    out["known_trigger_patch_idx_wrist_image"] = [i + NUM_PATCHES_PER_IMAGE for i in TRIGGER_PATCH_IDX]

    out_path = Path("trial_error/probe_logs/cross_attention_mahalanobis_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_probe()
