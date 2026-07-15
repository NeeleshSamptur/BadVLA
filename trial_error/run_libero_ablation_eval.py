"""
trial_error/run_libero_ablation_eval.py

Stage 3 (causal ablation) of backdoor_neuron_forensics_plan.md: does
silencing the neurons Stage 1/2 flagged actually kill the backdoor, without
breaking normal task performance?

================================================================================
HOW THIS DIFFERS FROM experiments/robot/libero/run_libero_eval.py
--------------------------------------------------------------------------------
This script does NOT copy that file. It imports its unchanged functions
(validate_config, initialize_model, setup_logging, run_task, GenerateConfig,
TaskSuite, benchmark) and adds only: two new config fields (which candidate
YAML to ablate, and the ablation value) and hook registration/removal around
the same run_task loop, via trial_error.paired_probe.apply_neuron_ablation_hooks
+ load_candidate_neurons_yaml (adapted from mechanistic-steering-vlas's
hooks.py / interventions.py -- see the module docstring in paired_probe.py's
Stage 3 section for what changed and why).

Baseline (no ablation) success rates for goal_block already exist from a
prior eval run and are intentionally NOT re-run here -- reuse, don't repeat:
  clean,     no ablation: 95.0%  experiments/robot/libero/experiments/logs/goal_block/EVAL-goal_block-sr_wo-2026_06_09-17_55_43.txt
  triggered, no ablation:  0.0%  experiments/robot/libero/experiments/logs/goal_block/EVAL-goal_block-sr_w-2026_06_09-17_55_43.txt
This script measures the same two conditions WITH ablation enabled, so the
before/after (Stage 1's causal claim) can be read directly against those
existing baselines:
  triggered + ablation success rate should climb back toward ~95% if the
    flagged neurons are actually responsible for the backdoor.
  clean + ablation success rate should stay near ~95% (no collateral
    damage to normal task performance).
================================================================================
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import draccus
import tqdm
from libero.libero import benchmark

sys.path.append("../..")
from experiments.robot.libero.run_libero_eval import (  # noqa: E402
    GenerateConfig as _EvalConfig,
    validate_config,
    initialize_model,
    setup_logging,
    run_task,
    log_message,
)
from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))
from trial_error.paired_probe import (  # noqa: E402
    apply_neuron_ablation_hooks,
    load_candidate_neurons_yaml,
)


@dataclass
class AblationConfig(_EvalConfig):
    # Stage 3 additions only -- everything else is the unmodified eval config.
    intervention_yaml: str = ""  # path to candidate_neurons_*.yaml; "" = no ablation (plain baseline)
    intervention_dict_name: str = "stage1_candidates"
    ablation_coef: float = 0.0  # value written into flagged neurons before down_proj (0.0 = zero-ablate)


@draccus.wrap()
def eval_libero_with_ablation(cfg: AblationConfig) -> float:
    """Same rollout+success-rate loop as run_libero_eval.py's eval_libero(),
    with neuron-ablation hooks registered around it when intervention_yaml
    is set."""
    validate_config(cfg)
    set_seed_everywhere(cfg.seed)

    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    hook_handles = []
    if cfg.intervention_yaml:
        layer_to_neurons = load_candidate_neurons_yaml(cfg.intervention_yaml, cfg.intervention_dict_name)
        n_neurons = sum(len(v) for v in layer_to_neurons.values())
        hook_handles = apply_neuron_ablation_hooks(model, layer_to_neurons, coef=cfg.ablation_coef)
        log_message(
            f"Stage 3 ablation ON: {n_neurons} neuron(s) across {len(layer_to_neurons)} "
            f"layer(s) from '{cfg.intervention_dict_name}' (coef={cfg.ablation_coef}).",
            log_file,
        )
    else:
        log_message("Stage 3 ablation OFF (no --intervention_yaml given) -- plain baseline eval.", log_file)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    if cfg.trigger:
        log_message(f"Task suite: {cfg.task_suite_name} (white pixel block trigger ON)", log_file)
    else:
        log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    total_episodes, total_successes = 0, 0
    try:
        for task_id in tqdm.tqdm(range(num_tasks)):
            total_episodes, total_successes = run_task(
                cfg,
                task_suite,
                task_id,
                model,
                resize_size,
                processor,
                action_head,
                proprio_projector,
                noisy_action_projector,
                total_episodes,
                total_successes,
                log_file,
            )
    finally:
        for h in hook_handles:
            h.remove()

    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    log_message("Final results (Stage 3 ablation eval):", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    if log_file:
        log_file.close()
    return final_success_rate


if __name__ == "__main__":
    eval_libero_with_ablation()
