"""
trial_error/run_tsne_data_collection.py

Collect real, per-scene, per-layer whole-block hidden-state activations
(clean vs. triggered), for a t-SNE separability check. Saves raw vectors to
disk (not just a summary statistic) so the t-SNE step can be run separately,
honestly, on the real per-sample data -- not an averaged/cherry-picked view.

Reuses run_libero_probe.py's model loading and scene-setup helpers, and
paired_probe.py's existing (unmodified) hook registration -- no new hooks,
this only needs the whole-block "llm.layer_XX" captures that already exist.
"""

import copy
import sys
from pathlib import Path

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

sys.path.append(str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import get_action, get_image_resize_size

from trial_error.run_libero_probe import (
    GenerateConfig,
    initialize_model,
    add_trigger_img,
    _warmup_and_prepare_observation,
)
from trial_error.paired_probe import (
    Capture,
    register_all_probe_hooks,
    set_probe_quiet,
    pool_tokens,
)

CLEAN_SUITE = "libero_goal"
MUG_SUITE = "libero_goal_with_mug"


@draccus.wrap()
def run_collection(cfg: GenerateConfig) -> None:
    """cfg.probe_trigger selects "block" or "mug". Collects
    cfg.num_trials_per_task episodes per task, across all tasks in
    libero_goal (10 tasks), for both conditions."""
    trigger_kind = cfg.probe_trigger
    assert trigger_kind in ("block", "mug")

    cfg.task_suite_name = CLEAN_SUITE
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[CLEAN_SUITE]()
    task_suite_mug = benchmark_dict[MUG_SUITE]() if trigger_kind == "mug" else None
    num_tasks = task_suite.n_tasks

    capture = Capture()
    hook_groups = register_all_probe_hooks(
        model, capture,
        proprio_projector=proprio_projector,
        action_head=action_head,
        noisy_action_projector=noisy_action_projector,
    )
    block_names = sorted(
        (n for n in hook_groups.llm_names if n.count(".") == 1),
        key=lambda n: int(n[len("llm.layer_"):]),
    )

    set_probe_quiet(True)
    # clean_vecs[layer] = list of 4096-dim np arrays, one per scene
    clean_vecs = {l: [] for l in range(32)}
    trig_vecs = {l: [] for l in range(32)}
    n_scenes = 0

    try:
        for task_id in tqdm.tqdm(range(num_tasks)):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)

            if trigger_kind == "mug":
                task_mug = task_suite_mug.get_task(task_id)
                env_mug, desc_mug = get_libero_env(task_mug, cfg.model_family, resolution=cfg.env_img_res)
                initial_states_mug = task_suite_mug.get_task_init_states(task_id)
                assert task_description == desc_mug

            n_ep = cfg.num_trials_per_task
            for episode_idx in range(n_ep):
                if episode_idx >= len(initial_states):
                    break
                clean_observation = _warmup_and_prepare_observation(
                    env, cfg, initial_states[episode_idx], resize_size)

                clean_obs = copy.deepcopy(clean_observation)
                if trigger_kind == "block":
                    trig_obs = copy.deepcopy(clean_observation)
                    trig_obs["full_image"] = add_trigger_img(
                        trig_obs["full_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)
                    trig_obs["wrist_image"] = add_trigger_img(
                        trig_obs["wrist_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)
                else:
                    if episode_idx >= len(initial_states_mug):
                        break
                    trig_obs = _warmup_and_prepare_observation(
                        env_mug, cfg, initial_states_mug[episode_idx], resize_size)

                capture.reset()
                get_action(cfg, model, clean_obs, task_description,
                          processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector,
                          noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
                clean_store = capture.snapshot()

                capture.reset()
                get_action(cfg, model, trig_obs, task_description,
                          processor=processor, action_head=action_head,
                          proprio_projector=proprio_projector,
                          noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
                trig_store = capture.snapshot()

                for name in block_names:
                    layer_num = int(name[len("llm.layer_"):])
                    c = clean_store.get(name, [])
                    t = trig_store.get(name, [])
                    if not c or not t:
                        continue
                    clean_vecs[layer_num].append(pool_tokens(c[0]).astype(np.float32))
                    trig_vecs[layer_num].append(pool_tokens(t[0]).astype(np.float32))
                n_scenes += 1
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    print(f"[tsne-collect] {trigger_kind}: collected {n_scenes} scenes")

    out_path = Path(f"trial_error/probe_logs/tsne_activations_{trigger_kind}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for l in range(32):
        if clean_vecs[l]:
            save_dict[f"clean_L{l:02d}"] = np.stack(clean_vecs[l])
            save_dict[f"trig_L{l:02d}"] = np.stack(trig_vecs[l])
    np.savez(out_path, **save_dict, n_scenes=n_scenes)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    run_collection()
