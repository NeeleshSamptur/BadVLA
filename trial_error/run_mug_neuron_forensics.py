"""
trial_error/run_mug_neuron_forensics.py

Cross-trigger check: does Stage 1's neuron-flagging (clean vs. real-trigger
vs. control, ratio test) point at the same neurons for a completely
different trigger MECHANISM -- a real physical object (mug) placed in the
scene -- as it did for the synthetic pixel-block trigger?

The block trigger's control was a random-noise image patch (no natural
equivalent for a physical-object trigger). Here the control is a DIFFERENT
physical object -- the "red stick" trigger from a separate BadVLA checkpoint
family -- placed in the same scene, on the theory that it's an equally
"unusual/unexpected object in the scene" decoy that was never trained as
THIS checkpoint's (goal_mug's) actual trigger. Same logic as the noise
patch, adapted to a physical-object trigger where there's no image-space
manipulation to fall back on.

Reuses run_libero_probe.py's model loading and scene-setup helpers, and
paired_probe.py's Stage 1/2 functions, unchanged.
"""

import sys
from pathlib import Path

import draccus
import numpy as np
import tqdm
import yaml
from libero.libero import benchmark

sys.path.append(str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import get_action, get_image_resize_size

from trial_error.run_libero_probe import (
    GenerateConfig,
    initialize_model,
    _warmup_and_prepare_observation,
    _aggregate,
    write_candidate_neurons_yaml,
)
from trial_error.paired_probe import (
    Capture,
    register_all_probe_hooks,
    set_probe_quiet,
    compute_layer_metrics,
    compute_ffn_preact_diffs,
    aggregate_ffn_preact_diffs,
    select_candidate_layers,
    flag_candidate_neurons,
    format_candidate_neurons_table,
    label_candidate_neurons_with_tokens,
)

CLEAN_SUITE = "libero_goal"
MUG_SUITE = "libero_goal_with_mug"
CONTROL_SUITE = "libero_goal_with_red_stick"  # decoy object, never goal_mug's real trigger


def load_flagged_neurons(yaml_path: str, dict_name: str = "stage1_candidates"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        all_dicts = yaml.safe_load(f) or {}
    raw = all_dicts[dict_name] or {}
    return {int(k): set(int(n) for n in v) for k, v in raw.items()}


@draccus.wrap()
def run_mug_forensics(cfg: GenerateConfig) -> None:
    cfg.probe_trigger = "mug"  # not used for gating here; scene building is manual below
    cfg.task_suite_name = CLEAN_SUITE  # force libero_goal regardless of CLI, for unnorm_key/check consistency

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[CLEAN_SUITE]()
    task_suite_mug = benchmark_dict[MUG_SUITE]()
    task_suite_stick = benchmark_dict[CONTROL_SUITE]()
    num_tasks = task_suite.n_tasks

    capture = Capture()
    hook_groups = register_all_probe_hooks(
        model, capture,
        proprio_projector=proprio_projector,
        action_head=action_head,
        noisy_action_projector=noisy_action_projector,
        hook_ffn_preact=True,
    )

    set_probe_quiet(True)
    llm_metrics_per_scene = []
    ffn_diffs_per_scene = []
    n_scenes = 0

    try:
        for task_id in tqdm.tqdm(range(num_tasks)):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)

            task_mug = task_suite_mug.get_task(task_id)
            env_mug, desc_mug = get_libero_env(task_mug, cfg.model_family, resolution=cfg.env_img_res)
            initial_states_mug = task_suite_mug.get_task_init_states(task_id)

            task_stick = task_suite_stick.get_task(task_id)
            env_stick, desc_stick = get_libero_env(task_stick, cfg.model_family, resolution=cfg.env_img_res)
            initial_states_stick = task_suite_stick.get_task_init_states(task_id)

            assert task_description == desc_mug == desc_stick, (
                f"Task description mismatch: {task_description!r} / {desc_mug!r} / {desc_stick!r}"
            )

            for episode_idx in range(cfg.num_trials_per_task):
                if episode_idx >= min(len(initial_states), len(initial_states_mug), len(initial_states_stick)):
                    break

                clean_obs = _warmup_and_prepare_observation(env, cfg, initial_states[episode_idx], resize_size)
                mug_obs = _warmup_and_prepare_observation(env_mug, cfg, initial_states_mug[episode_idx], resize_size)
                stick_obs = _warmup_and_prepare_observation(env_stick, cfg, initial_states_stick[episode_idx], resize_size)

                capture.reset()
                get_action(cfg, model, clean_obs, task_description,
                          processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector,
                          noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
                clean_store = capture.snapshot()

                capture.reset()
                get_action(cfg, model, mug_obs, task_description,
                          processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector,
                          noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
                mug_store = capture.snapshot()

                capture.reset()
                get_action(cfg, model, stick_obs, task_description,
                          processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector,
                          noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
                stick_store = capture.snapshot()

                llm_rows = compute_layer_metrics(
                    clean_store, mug_store, hook_groups.llm_names, group="llm", quiet=True)
                llm_metrics_per_scene.append(llm_rows)

                ffn_diffs = compute_ffn_preact_diffs(
                    clean_store, mug_store, stick_store, hook_groups.ffn_preact_names)
                ffn_diffs_per_scene.append(ffn_diffs)
                n_scenes += 1
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    print(f"\n[mug-forensics] processed {n_scenes} scenes\n")

    agg_llm = _aggregate(llm_metrics_per_scene)
    candidate_layers = select_candidate_layers(agg_llm, top_n=cfg.ffn_top_layers)
    print(f"Candidate layers (mug, by relative_l2): {sorted(candidate_layers)}")

    agg_ffn_diffs = aggregate_ffn_preact_diffs(ffn_diffs_per_scene)
    mug_candidates = flag_candidate_neurons(
        agg_ffn_diffs, candidate_layers,
        top_k=cfg.ffn_top_k_neurons, ratio_thresh=cfg.ffn_ratio_thresh)
    print(f"Flagged {len(mug_candidates)} neuron(s) for MUG trigger.")

    mug_candidates = label_candidate_neurons_with_tokens(model, processor, mug_candidates, top_k=cfg.stage2_top_k)

    text = "\n".join(format_candidate_neurons_table(mug_candidates, title="Candidate backdoor neurons (MUG trigger):"))
    print("\n" + text)

    out_dir = Path("trial_error/probe_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_libero_probe_log_goal_mug_forensics.txt").write_text(text + "\n")
    write_candidate_neurons_yaml(
        out_dir / "candidate_neurons_goal_mug_forensics.yaml", mug_candidates, dict_name="stage1_candidates")
    print(f"\nSaved -> {out_dir / 'run_libero_probe_log_goal_mug_forensics.txt'}")
    print(f"Saved -> {out_dir / 'candidate_neurons_goal_mug_forensics.yaml'}")

    # Compare against the block trigger's already-flagged neurons.
    block_yaml = "trial_error/probe_logs/candidate_neurons_goal_block_2026_07_13-22_24_42.yaml"
    if Path(block_yaml).exists():
        block_flagged = load_flagged_neurons(block_yaml)
        mug_flagged = {}
        for row in mug_candidates:
            layer_num = int(row["layer"][len("llm.layer_"):].partition(".")[0])
            mug_flagged.setdefault(layer_num, set()).add(row["neuron_idx"])

        print("\n=== Cross-trigger comparison: BLOCK vs MUG flagged neurons ===")
        print(f"Block candidate layers: {sorted(block_flagged.keys())}")
        print(f"Mug candidate layers:   {sorted(mug_flagged.keys())}")
        shared_layers = set(block_flagged.keys()) & set(mug_flagged.keys())
        print(f"Shared candidate layers: {sorted(shared_layers)}")

        total_shared_neurons = 0
        for layer in sorted(shared_layers):
            shared = block_flagged[layer] & mug_flagged[layer]
            total_shared_neurons += len(shared)
            print(f"  layer {layer}: block={len(block_flagged[layer])} flagged, "
                  f"mug={len(mug_flagged[layer])} flagged, shared neuron IDs={sorted(shared)}")
        print(f"Total shared (layer, neuron_idx) pairs across BOTH triggers: {total_shared_neurons}")


if __name__ == "__main__":
    run_mug_forensics()
