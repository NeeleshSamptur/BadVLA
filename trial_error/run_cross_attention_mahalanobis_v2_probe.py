"""
trial_error/run_cross_attention_mahalanobis_v2_probe.py

Extends run_cross_attention_mahalanobis_probe.py with the same specificity
check already applied to the raw-activation Mahalanobis detector: does the
per-patch attention-share anomaly actually track the REAL trigger, or would
any input perturbation (a same-geometry random-noise patch) or any different
task (a genuinely different scene from libero_object, same checkpoint) trigger
it just as much?

Four conditions per scene, all clean-calibrated (mu/sigma fit on held-out
CLEAN scenes only, same diagonal-Mahalanobis recipe as before):
  clean   -- libero_goal, matching task
  trig    -- real backdoor trigger (white patch, block)
  control -- random-noise patch, same size/position, never a real trigger
  object  -- a genuinely different task's real (image, instruction) pair from
             libero_object, fed through the same goal-suite checkpoint
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
from trial_error.run_cross_attention_mahalanobis_probe import per_patch_attention_vector, fit_mu_sigma, auroc, N_PATCHES_TOTAL

CLEAN_SUITE = "libero_goal"
OBJECT_SUITE = "libero_object"


def add_control_img(image, trigger_size=0.10, trigger_position="center", seed=0):
    """Benign perturbation, same patch geometry as add_trigger_img but never a
    real trigger (uniform random RGB noise instead of a flat white block)."""
    control_image = copy.deepcopy(image)
    h, w = control_image.shape[:2]
    trigger_size = int(min(h, w) * trigger_size)
    center_x, center_y = w // 2, h // 2
    start_x, end_x = center_x - trigger_size // 2, center_x + trigger_size // 2
    start_y, end_y = center_y - trigger_size // 2, center_y + trigger_size // 2
    rng = np.random.default_rng(seed)
    patch_shape = control_image[start_y:end_y, start_x:end_x].shape
    control_image[start_y:end_y, start_x:end_x] = rng.integers(0, 256, size=patch_shape, dtype=np.uint8)
    return control_image


@draccus.wrap()
def run_probe(cfg: GenerateConfig) -> None:
    assert cfg.probe_trigger == "block", "this probe only supports the block trigger (known patch geometry, for validation)"
    N_EPISODES_PER_TASK = 3  # 10 libero_goal tasks x 3 = up to 30 scenes

    cfg.task_suite_name = CLEAN_SUITE
    processor = get_processor(cfg)
    vla = load_model_eager(cfg)
    check_unnorm_key(cfg, vla)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)

    task_suite = benchmark.get_benchmark_dict()[CLEAN_SUITE]()
    task_suite_object = benchmark.get_benchmark_dict()[OBJECT_SUITE]()
    num_tasks = task_suite.n_tasks
    num_object_tasks = task_suite_object.n_tasks
    n_layers = vla.config.text_config.num_hidden_layers

    set_probe_quiet(True)
    conditions = ["clean", "trig", "control", "object"]
    vecs = {c: {l: [] for l in range(n_layers)} for c in conditions}
    n_scenes = 0

    try:
        for task_id in tqdm.tqdm(range(num_tasks)):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)

            object_task_id = task_id % num_object_tasks
            task_obj = task_suite_object.get_task(object_task_id)
            env_obj, desc_obj = get_libero_env(task_obj, cfg.model_family, resolution=cfg.env_img_res)
            initial_states_obj = task_suite_object.get_task_init_states(object_task_id)

            for episode_idx in range(min(N_EPISODES_PER_TASK, len(initial_states))):
                obs = _warmup_and_prepare_observation(env, cfg, initial_states[episode_idx], resize_size)
                clean_obs = copy.deepcopy(obs)

                trig_obs = copy.deepcopy(obs)
                trig_obs["full_image"] = add_trigger_img(trig_obs["full_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)
                trig_obs["wrist_image"] = add_trigger_img(trig_obs["wrist_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)

                seed = task_id * 1000 + episode_idx
                control_obs = copy.deepcopy(obs)
                control_obs["full_image"] = add_control_img(control_obs["full_image"], trigger_size=0.10, trigger_position="center", seed=seed)
                control_obs["wrist_image"] = add_control_img(control_obs["wrist_image"], trigger_size=0.10, trigger_position="center", seed=seed + 1)

                obj_ep = episode_idx % len(initial_states_obj)
                object_obs = _warmup_and_prepare_observation(env_obj, cfg, initial_states_obj[obj_ep], resize_size)

                scene_obs = {"clean": (clean_obs, task_description), "trig": (trig_obs, task_description),
                             "control": (control_obs, task_description), "object": (object_obs, desc_obj)}

                attn_by_cond = {}
                qspan = None
                for cond, (o, desc) in scene_obs.items():
                    attn, qs, qe = predict_action_with_attention(cfg, vla, processor, o, desc, action_head, proprio_projector)
                    attn_by_cond[cond] = attn
                    qspan = (qs, qe)

                for l in range(n_layers):
                    for cond in conditions:
                        vecs[cond][l].append(per_patch_attention_vector(attn_by_cond[cond][l], *qspan))

                del attn_by_cond
                torch.cuda.empty_cache()
                n_scenes += 1
    finally:
        set_probe_quiet(False)

    print(f"\n[cross-attn-maha-v2] {n_scenes} scenes, 4 conditions each")

    rng = np.random.default_rng(0)
    idx = rng.permutation(n_scenes)
    n_cal = n_scenes // 2
    cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

    import json
    out = {"n_scenes": n_scenes, "n_cal": len(cal_idx), "n_test": len(test_idx), "rows": []}
    sq = {c: (np.zeros(len(test_idx)) if c == "clean" else np.zeros(n_scenes)) for c in conditions}

    for l in range(n_layers):
        C = np.stack(vecs["clean"][l])
        mu, sigma = fit_mu_sigma(C[cal_idx])
        row = {"layer": l}
        z_clean_test = (C[test_idx] - mu) / sigma
        sq["clean"] += (z_clean_test ** 2).sum(axis=1)
        row["clean_maha_mean"] = float(np.linalg.norm(z_clean_test, axis=1).mean())
        for cond in ("trig", "control", "object"):
            X = np.stack(vecs[cond][l])
            z = (X - mu) / sigma
            sq[cond] += (z ** 2).sum(axis=1)
            row[f"{cond}_maha_mean"] = float(np.linalg.norm(z, axis=1).mean())
            row[f"{cond}_layer_auroc"] = auroc(np.sqrt((z ** 2).sum(axis=1)), np.sqrt((z_clean_test ** 2).sum(axis=1)))
        out["rows"].append(row)

    group_auroc = {cond: auroc(np.sqrt(sq[cond]), np.sqrt(sq["clean"])) for cond in ("trig", "control", "object")}
    out["group_auroc"] = group_auroc

    print(f"{'Layer':<6} | {'clean maha':>11} | {'trig maha':>10} | {'ctrl maha':>10} | {'obj maha':>9} | {'trig AUC':>9} | {'ctrl AUC':>9} | {'obj AUC':>8}")
    for row in out["rows"]:
        print(f"{row['layer']:<6} | {row['clean_maha_mean']:11.2f} | {row['trig_maha_mean']:10.2f} | {row['control_maha_mean']:10.2f} | {row['object_maha_mean']:9.2f} | {row['trig_layer_auroc']:9.3f} | {row['control_layer_auroc']:9.3f} | {row['object_layer_auroc']:8.3f}")

    print(f"\nGroup AUROC (all 32 layers combined, chi-squared style):")
    for cond, v in group_auroc.items():
        print(f"  {cond}: {v:.4f}")

    out_path = Path("trial_error/probe_logs/cross_attention_mahalanobis_v2_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_probe()
