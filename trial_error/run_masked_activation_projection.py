"""
trial_error/run_masked_activation_projection.py

Last check requested before deciding whether to abandon the neuron-flagging
pipeline: for the ~200 neurons Stage 1 already flagged, project their REAL
captured activations (not one-hot weight vectors -- that's Stage 2) through
vocab, separately for clean and triggered scenes, to see whether the
flagged-neuron GROUP's real, combined contribution reads differently under
the trigger than under clean.

For each of the 8 candidate layers: mask every neuron except that layer's
flagged ones to zero, average the masked pooled activation across scenes,
run through that layer's down_proj + lm_head, decode top-10 tokens.
Produces two files: one for clean, one for triggered.
"""

import copy
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
    add_trigger_img,
    _warmup_and_prepare_observation,
)
from trial_error.paired_probe import (
    Capture,
    register_all_probe_hooks,
    set_probe_quiet,
    pool_tokens,
    mask_to_flagged_neurons,
    project_masked_activation_to_vocab,
    decode_token_ids,
)


def load_flagged_neurons(yaml_path: str, dict_name: str = "stage1_candidates"):
    with open(yaml_path, "r", encoding="utf-8") as f:
        all_dicts = yaml.safe_load(f) or {}
    raw = all_dicts[dict_name] or {}
    return {int(k): [int(n) for n in v] for k, v in raw.items()}


@draccus.wrap()
def run_masked_projection(cfg: GenerateConfig) -> None:
    assert cfg.probe_trigger == "block", "this check only supports the block trigger"

    flagged_yaml = "trial_error/probe_logs/candidate_neurons_goal_block_2026_07_13-22_24_42.yaml"
    layer_to_neurons = load_flagged_neurons(flagged_yaml)
    print(f"[masked-proj] loaded flagged neurons for layers: {sorted(layer_to_neurons.keys())}")

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()

    capture = Capture()
    hook_groups = register_all_probe_hooks(
        model, capture,
        proprio_projector=proprio_projector,
        action_head=action_head,
        noisy_action_projector=noisy_action_projector,
        hook_ffn_preact=True,
    )
    preact_names = {f"llm.layer_{n:02d}.mlp.preact": n for n in layer_to_neurons}

    N_SCENES_TO_DUMP = 2  # per-scene, NOT averaged -- check cross-scene consistency directly
    set_probe_quiet(True)
    all_lines = []

    try:
        for task_id in range(N_SCENES_TO_DUMP):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)
            initial_state = initial_states[0]
            clean_observation = _warmup_and_prepare_observation(env, cfg, initial_state, resize_size)

            clean_obs = copy.deepcopy(clean_observation)
            trig_obs = copy.deepcopy(clean_observation)
            trig_obs["full_image"] = add_trigger_img(
                trig_obs["full_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)
            trig_obs["wrist_image"] = add_trigger_img(
                trig_obs["wrist_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)

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

            header = f"\n=== Scene {task_id}: task '{task_description}' ==="
            print(header)
            all_lines.append(header)

            for preact_name, layer_num in sorted(preact_names.items(), key=lambda kv: kv[1]):
                c = clean_store.get(preact_name, [])
                t = trig_store.get(preact_name, [])
                if not c or not t:
                    continue
                pc = mask_to_flagged_neurons(pool_tokens(c[0]), layer_to_neurons[layer_num])
                pt = mask_to_flagged_neurons(pool_tokens(t[0]), layer_to_neurons[layer_num])

                clean_top_ids = project_masked_activation_to_vocab(model, layer_num, pc, top_k=10)
                trig_top_ids = project_masked_activation_to_vocab(model, layer_num, pt, top_k=10)
                clean_toks = decode_token_ids(model, processor, clean_top_ids)
                trig_toks = decode_token_ids(model, processor, trig_top_ids)

                line1 = f"llm.layer_{layer_num:02d}  ({len(layer_to_neurons[layer_num])} flagged neurons)"
                line2 = f"  clean: {clean_toks}"
                line3 = f"  trig : {trig_toks}"
                print(line1); print(line2); print(line3)
                all_lines.extend([line1, line2, line3])
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    out_path = Path("trial_error/probe_logs/masked_projection_2scenes.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_lines) + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_masked_projection()
