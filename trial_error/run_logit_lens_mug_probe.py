"""
trial_error/run_logit_lens_mug_probe.py

Fair cross-trigger test for the classic logit-lens (whole-block activation)
approach, mirroring run_logit_lens_activation_probe.py but for the physical
mug trigger instead of the synthetic block trigger. Same methodology: 2
scenes, all 32 layers, clean vs. real-trigger, dumped in full (not
averaged) so cross-scene consistency can be checked directly -- exactly
like we did for block, and exactly like we just did for Stage 1's neuron
flagging on mug.
"""

import sys
from pathlib import Path

import draccus
import tqdm
from libero.libero import benchmark

sys.path.append(str(Path(__file__).resolve().parents[1]))
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import get_action, get_image_resize_size

from trial_error.run_libero_probe import (
    GenerateConfig,
    initialize_model,
    _warmup_and_prepare_observation,
)
from trial_error.paired_probe import (
    Capture,
    register_all_probe_hooks,
    set_probe_quiet,
    logit_lens_activation_divergence,
    decode_token_ids,
)

CLEAN_SUITE = "libero_goal"
MUG_SUITE = "libero_goal_with_mug"
N_SCENES_TO_DUMP = 2


@draccus.wrap()
def run_logit_lens_mug_probe(cfg: GenerateConfig) -> None:
    cfg.task_suite_name = CLEAN_SUITE

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[CLEAN_SUITE]()
    task_suite_mug = benchmark_dict[MUG_SUITE]()

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
    print(f"[logit-lens-mug-probe] {len(block_names)} whole-block layers, dumping {N_SCENES_TO_DUMP} scenes in full")

    set_probe_quiet(True)
    all_lines = []

    try:
        for task_id in range(N_SCENES_TO_DUMP):
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)

            task_mug = task_suite_mug.get_task(task_id)
            env_mug, desc_mug = get_libero_env(task_mug, cfg.model_family, resolution=cfg.env_img_res)
            initial_states_mug = task_suite_mug.get_task_init_states(task_id)
            assert task_description == desc_mug

            clean_obs = _warmup_and_prepare_observation(env, cfg, initial_states[0], resize_size)
            mug_obs = _warmup_and_prepare_observation(env_mug, cfg, initial_states_mug[0], resize_size)

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

            rows = logit_lens_activation_divergence(model, clean_store, mug_store, block_names, top_k=10)

            header = f"\n=== Scene {task_id}: task '{task_description}' ==="
            print(header)
            all_lines.append(header)
            for r in rows:
                clean_toks = decode_token_ids(model, processor, r["clean_top_ids"])
                trig_toks = decode_token_ids(model, processor, r["trig_top_ids"])
                line1 = f"{r['layer']:<15} overlap={r['overlap']}/10"
                line2 = f"  clean: {clean_toks}"
                line3 = f"  mug  : {trig_toks}"
                print(line1); print(line2); print(line3)
                all_lines.extend([line1, line2, line3])
    finally:
        hook_groups.remove()
        set_probe_quiet(False)

    out_path = Path("trial_error/probe_logs/logit_lens_mug_2scenes_all_layers.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(all_lines) + "\n")
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_logit_lens_mug_probe()
