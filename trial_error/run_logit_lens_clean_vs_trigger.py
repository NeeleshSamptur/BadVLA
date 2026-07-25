"""
trial_error/run_logit_lens_clean_vs_trigger.py

Disjoint-scene logit-lens calibration detector: clean vs trigger.

This is now a thin driver over run_libero_probe.py's disjoint_mahalanobis
pipeline (see paired_probe.compute_logit_lens_by_group for the detector
itself) instead of loading the old 20-scene tsne_activations_v2_block.npz.
That fixes every gap the paired, 20-scene version had:

  * Runs the LIBERO simulator for 500 scenes (disjoint_n_cal +
    disjoint_n_clean_test + disjoint_n_trig, default 200/150/150) instead of
    reusing a static 20-clean/20-trigger dump -- n=150 vs 150 for the actual
    AUROC, matching the Mahalanobis disjoint protocol exactly.
  * Calibration, clean-test, and trigger each draw from non-overlapping scene
    pools (see run_libero_probe.py's disjoint_mahalanobis flag) instead of
    the same 20 scenes' paired clean/triggered twins -- this measures
    disjoint-pool detection, not paired-block divergence. The split is also
    TASK-STRATIFIED (see paired_probe.stratified_disjoint_split): every
    libero_goal task contributes its own proportional share of scenes to
    cal, clean-test, AND trigger, so task identity can't act as a shortcut
    signal for "was this triggered" the way a flat first-N/last-M scene cut
    would (that cut happened to land on a task boundary and made tasks 0-6
    exclusively clean/cal and tasks 7-9 exclusively trigger).
  * Clean-test and trigger are scored identically: both are compared, via
    Jensen-Shannon divergence of their softmax token distributions, against
    ONE fixed reference built from the full calibration pool. Neither
    condition uses a leave-one-out reference at scoring time, so there is no
    asymmetry between them (the old version scored clean via leave-one-out
    but trigger against the full mean).
  * Softmax first: distances are computed over token probability
    distributions (bounded, symmetric JS divergence), not raw cosine
    similarity between un-normalized logit vectors.
  * Per-layer JS distances are z-scored against a calibration-only
    leave-one-out null spread, then combined across layers with a LINEAR sum
    (Stouffer's-method style, not Mahalanobis's squared-sum -- JS distance is
    already one-sided, see compute_logit_lens_by_group's docstring for the
    synthetic sanity check that justified this), so no single layer's raw
    distance scale can dominate the combined score.

Everything printed also lands in probe_logs/logit_lens_clean_vs_trigger.log
(see run_libero_probe._write_logit_lens_log). The same simulator pass also
produces the full disjoint Mahalanobis table (all groups) as a side effect --
that's cheap once the 500 scenes have been rolled out, so it's saved too, in
the usual probe_logs/run_libero_probe_log_goal_block_*.txt location.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append("/home/grads/nsamptur/vla_bkd_def/LIBERO")
from trial_error.run_libero_probe import GenerateConfig, run_single_probe

CHECKPOINT = "/home/grads/nsamptur/vla_bkd_def/BadVLA/vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt"
LOG_PATH = Path("trial_error/probe_logs/logit_lens_clean_vs_trigger.log")


def main():
    cfg = GenerateConfig(
        pretrained_checkpoint=CHECKPOINT,
        probe_trigger="block",
        task_suite_name="libero_goal",
        disjoint_mahalanobis=True,
        disjoint_n_cal=200,
        disjoint_n_clean_test=150,
        disjoint_n_trig=150,
    )
    run_single_probe(cfg, logit_lens_log_path=LOG_PATH)


if __name__ == "__main__":
    main()
