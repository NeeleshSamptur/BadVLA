"""
trial_error/run_tsne_data_collection_v2.py

Extends the first t-SNE data collection with the requested controls:
  - clean       (libero_goal, matching task)
  - trigger     (real backdoor trigger: white patch for block, real mug object for mug)
  - control     (random noise patch, same geometry as the block trigger --
                 used for BOTH checkpoints as a "some weird patch, never the
                 real trigger for either" baseline)
  - object      (a genuinely different task domain: a real, matched
                 (image, instruction) pair from libero_object, fed through
                 the SAME goal-suite-trained checkpoint)

Saves raw per-scene, per-layer whole-block activations for all four
conditions, per checkpoint, so every honest comparison (clean-vs-trigger,
clean-vs-control, clean-vs-object, and later block-clean-vs-mug-clean) can
be computed from real data, not assumed.
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


def add_control_img(image, trigger_size=0.10, trigger_position="center", seed=0):
    """Benign perturbation, same patch geometry as add_trigger_img but never
    used as a real trigger for either checkpoint (uniform random RGB noise
    instead of a flat white block). Inlined here rather than re-added to
    run_libero_probe.py, which this branch intentionally keeps logit-lens-only.
    """
    control_image = copy.deepcopy(image)
    h, w = control_image.shape[:2]
    trigger_size = int(min(h, w) * trigger_size)

    if trigger_position == "center":
        center_x, center_y = w // 2, h // 2
    elif trigger_position == "top_left":
        center_x, center_y = trigger_size // 2, trigger_size // 2
    elif trigger_position == "top_right":
        center_x, center_y = w - trigger_size // 2, trigger_size // 2
    elif trigger_position == "bottom_left":
        center_x, center_y = trigger_size // 2, h - trigger_size // 2
    elif trigger_position == "bottom_right":
        center_x, center_y = w - trigger_size // 2, h - trigger_size // 2

    start_x = center_x - trigger_size // 2
    end_x = center_x + trigger_size // 2
    start_y = center_y - trigger_size // 2
    end_y = center_y + trigger_size // 2

    rng = np.random.default_rng(seed)
    patch_shape = control_image[start_y:end_y, start_x:end_x].shape
    control_image[start_y:end_y, start_x:end_x] = rng.integers(
        0, 256, size=patch_shape, dtype=np.uint8)
    return control_image
from trial_error.paired_probe import (
    Capture,
    register_all_probe_hooks,
    set_probe_quiet,
    pool_tokens,
)

CLEAN_SUITE = "libero_goal"
MUG_SUITE = "libero_goal_with_mug"
OBJECT_SUITE = "libero_object"


@draccus.wrap()
def run_collection(cfg: GenerateConfig) -> None:
    trigger_kind = cfg.probe_trigger
    assert trigger_kind in ("block", "mug")

    cfg.task_suite_name = CLEAN_SUITE
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[CLEAN_SUITE]()
    task_suite_mug = benchmark_dict[MUG_SUITE]() if trigger_kind == "mug" else None
    task_suite_object = benchmark_dict[OBJECT_SUITE]()
    num_tasks = task_suite.n_tasks
    num_object_tasks = task_suite_object.n_tasks

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
    conditions = ["clean", "trig", "control", "object"]
    vecs = {c: {l: [] for l in range(32)} for c in conditions}
    n_scenes = 0

    def capture_condition(obs, task_desc, cond_name):
        capture.reset()
        get_action(cfg, model, obs, task_desc,
                  processor=processor, action_head=action_head,
                  proprio_projector=proprio_projector,
                  noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
        store = capture.snapshot()
        for name in block_names:
            layer_num = int(name[len("llm.layer_"):])
            arr = store.get(name, [])
            if arr:
                vecs[cond_name][layer_num].append(pool_tokens(arr[0]).astype(np.float32))

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

            # Matching object-suite task/scene for this iteration (own domain, own instruction).
            object_task_id = task_id % num_object_tasks
            task_obj = task_suite_object.get_task(object_task_id)
            env_obj, desc_obj = get_libero_env(task_obj, cfg.model_family, resolution=cfg.env_img_res)
            initial_states_obj = task_suite_object.get_task_init_states(object_task_id)

            n_ep = cfg.num_trials_per_task
            for episode_idx in range(n_ep):
                if episode_idx >= len(initial_states):
                    break
                clean_observation = _warmup_and_prepare_observation(
                    env, cfg, initial_states[episode_idx], resize_size)

                clean_obs = copy.deepcopy(clean_observation)

                control_obs = copy.deepcopy(clean_observation)
                seed = task_id * 1000 + episode_idx
                control_obs["full_image"] = add_control_img(
                    control_obs["full_image"], trigger_size=0.10, trigger_position="center", seed=seed)
                control_obs["wrist_image"] = add_control_img(
                    control_obs["wrist_image"], trigger_size=0.10, trigger_position="center", seed=seed + 1)

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

                obj_ep = episode_idx % len(initial_states_obj)
                object_observation = _warmup_and_prepare_observation(
                    env_obj, cfg, initial_states_obj[obj_ep], resize_size)

                capture_condition(clean_obs, task_description, "clean")
                capture_condition(trig_obs, task_description, "trig")
                capture_condition(control_obs, task_description, "control")
                capture_condition(object_observation, desc_obj, "object")
                n_scenes += 1
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    print(f"[tsne-collect-v2] {trigger_kind}: collected {n_scenes} scenes, 4 conditions each")

    out_path = Path(f"trial_error/probe_logs/tsne_activations_v2_{trigger_kind}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for cond in conditions:
        for l in range(32):
            if vecs[cond][l]:
                save_dict[f"{cond}_L{l:02d}"] = np.stack(vecs[cond][l])
    np.savez(out_path, **save_dict, n_scenes=n_scenes)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    run_collection()
