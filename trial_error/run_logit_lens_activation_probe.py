"""
trial_error/run_logit_lens_activation_probe.py

Quick check requested before abandoning the neuron-flagging pipeline:
apply the CLASSIC logit lens (project a REAL activation through lm_head,
not a fixed neuron weight vector -- see paired_probe.py's Stage 2 for that
other variant) to clean vs. triggered forward passes, at every LLM layer,
and see whether/where the model's "currently predicted token" diverges.

Reuses run_libero_probe.py's model loading, scene setup, and forward-pass
helpers unchanged -- the only new code is the per-layer real-activation
projection (paired_probe.py::logit_lens_activation_divergence) and this
thin driver loop.
"""

import copy
import sys
from pathlib import Path

import draccus
import numpy as np
import torch
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
    logit_lens_activation_divergence,
    decode_token_ids,
    format_logit_lens_divergence_table,
)


@draccus.wrap()
def run_logit_lens_probe(cfg: GenerateConfig) -> None:
    assert cfg.probe_trigger == "block", "this quick check only supports the block trigger"

    N_SCENES_TO_DUMP = 2  # exactly 2 scenes, full 32-layer dump each, no averaging

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
    )
    block_names = sorted(
        (n for n in hook_groups.llm_names if n.count(".") == 1),
        key=lambda n: int(n[len("llm.layer_"):]),
    )
    print(f"[logit-lens-probe] {len(block_names)} whole-block layers, dumping {N_SCENES_TO_DUMP} scenes in full")

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

            rows = logit_lens_activation_divergence(model, clean_store, trig_store, block_names, top_k=10)

            header = f"\n=== Scene {task_id}: task '{task_description}' ==="
            print(header)
            all_lines.append(header)
            for r in rows:
                clean_toks = decode_token_ids(model, processor, r["clean_top_ids"])
                trig_toks = decode_token_ids(model, processor, r["trig_top_ids"])
                line1 = f"{r['layer']:<15} overlap={r['overlap']}/10"
                line2 = f"  clean: {clean_toks}"
                line3 = f"  trig : {trig_toks}"
                print(line1); print(line2); print(line3)
                all_lines.extend([line1, line2, line3])
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    out_path = Path("trial_error/probe_logs/logit_lens_2scenes_all_layers.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_lines) + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_logit_lens_probe()
