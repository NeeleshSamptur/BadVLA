"""
trial_error/run_vocab_calibration_probe.py

Builds the calibration-based logit-lens vocab detector the user proposed:
instead of a paired clean-vs-triggered comparison for the SAME scene (what
every earlier logit-lens script did), build a "typical clean" reference from
MANY clean scenes averaged together, once, in advance -- then decode that
reference's top-10 words per layer, alongside individual held-out clean test
scenes and triggered scenes, so all three can be visually compared.

Reuses the already-collected pooled activations in
trial_error/probe_logs/tsne_activations_v2_block.npz (20 scenes, clean/trig/
control/object, all 32 whole-block layers) -- no new forward passes needed,
only the block checkpoint's lm_head weight for the projection.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.append("/home/grads/nsamptur/vla_bkd_def/LIBERO")
from trial_error.run_libero_probe import GenerateConfig, initialize_model
from trial_error.paired_probe import decode_token_ids

CHECKPOINT = "/home/grads/nsamptur/vla_bkd_def/BadVLA/vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt"
TOP_K = 10
N_CAL = 10          # first 10 clean scenes -> calibration reference (averaged)
TEST_SCENE_IDXS = [10, 15]  # 2 held-out scenes shown individually, clean + their triggered twin


def top_k_words(model, processor, vec, lm_head_weight, top_k=TOP_K):
    device, dtype = lm_head_weight.device, torch.float32
    v = torch.tensor(vec, dtype=dtype, device=device)
    logits = v @ lm_head_weight.to(dtype=dtype).T
    top_ids = torch.topk(logits, top_k).indices.tolist()
    return decode_token_ids(model, processor, top_ids)


def main():
    cfg = GenerateConfig(pretrained_checkpoint=CHECKPOINT, probe_trigger="block", task_suite_name="libero_goal", num_trials_per_task=1)
    model, *_, processor = initialize_model(cfg)
    lm_head_weight = model.language_model.lm_head.weight.detach()

    data = np.load("trial_error/probe_logs/tsne_activations_v2_block.npz")

    out = {"layers": []}
    for l in range(32):
        ck = f"clean_L{l:02d}"
        tk = f"trig_L{l:02d}"
        if ck not in data or tk not in data:
            continue
        clean_all = data[ck]  # (20, 4096)
        trig_all = data[tk]

        calibration_vec = clean_all[:N_CAL].mean(axis=0)
        calibration_words = top_k_words(model, processor, calibration_vec, lm_head_weight)

        clean_scene_words = [top_k_words(model, processor, clean_all[i], lm_head_weight) for i in TEST_SCENE_IDXS]
        trig_scene_words = [top_k_words(model, processor, trig_all[i], lm_head_weight) for i in TEST_SCENE_IDXS]

        out["layers"].append({
            "layer": l,
            "calibration": calibration_words,
            "clean_scenes": clean_scene_words,
            "trig_scenes": trig_scene_words,
        })
        print(f"layer {l:2d} | calib: {calibration_words[:3]} | clean0: {clean_scene_words[0][:3]} | trig0: {trig_scene_words[0][:3]}")

    out_path = Path("trial_error/probe_logs/vocab_calibration_words.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
