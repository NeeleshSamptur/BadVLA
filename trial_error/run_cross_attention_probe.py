"""
trial_error/run_cross_attention_probe.py

Implements the experiment sketched in trial_error/attention_analysis_idea.md:
recover the "cross-modal attention" submatrix from inside the LLM's ordinary
self-attention (query = action-token positions, key = vision-patch positions),
and check whether it concentrates abnormally on the trigger patch's own token
positions for triggered vs. clean inputs.

Requires attn_implementation="eager" -- the fast kernels (SDPA/FlashAttention)
loaded by default do not expose attention weights. This reloads the model with
eager attention (a real, separate load, not just a new hook) and manually
replicates OpenVLAForActionPrediction.predict_action()'s internals up through
the language_model(...) call, since that call hardcodes output_attentions=False
in the shared modeling file -- rather than patch shared code, this script calls
the same private helper methods the real predict_action() uses, with
output_attentions=True added at the one call site that needs it.

Vision token layout (from _process_vision_features / _build_multimodal_attention):
  position 0                     : first language token (BOS)
  positions 1 .. 256              : full_image patches (16x16 grid, row-major)
  positions 257 .. 512             : wrist_image patches (16x16 grid, row-major)
  position 513                    : proprio token
  positions 514 .. NUM_PATCHES+NUM_PROMPT_TOKENS-1 : rest of instruction text
  positions [NUM_PATCHES+NUM_PROMPT_TOKENS, +ACTION_DIM*NUM_ACTIONS_CHUNK) : action-token positions (query of interest)

Trigger region: add_trigger_img places a 10%-sized square at the image center.
For a 16x16 patch grid that is the 2x2 block of patches at grid rows/cols {7,8}
(0-indexed), i.e. flat indices {119, 120, 135, 136} within each image's 256-patch
block. A same-size control region (grid rows/cols {1,2}, far from center, flat
indices {17, 18, 33, 34}) is also scored, to check whether any extra attention
found is specific to the trigger's actual location or is a generic "corner vs.
center" artifact.
"""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # BadVLA's own experiments/ must win over the openvla-oft editable install
sys.path.append("/home/grads/nsamptur/vla_bkd_def/LIBERO")  # broken editable install, MAPPING is empty

import draccus
import numpy as np
import torch
import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.robot_utils import get_image_resize_size
from experiments.robot.openvla_utils import (
    get_processor,
    get_action_head,
    get_proprio_projector,
    prepare_images_for_vla,
    update_auto_map,
    check_model_logic_mismatch,
    model_is_on_hf_hub,
    _load_dataset_stats,
    normalize_proprio,
)
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from transformers import AutoConfig, AutoImageProcessor, AutoProcessor, AutoModelForVision2Seq

from trial_error.run_libero_probe import GenerateConfig, add_trigger_img, _warmup_and_prepare_observation, check_unnorm_key
from trial_error.paired_probe import set_probe_quiet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 16x16 patch grid per image (224/14). Trigger = 10%-sized center square -> grid rows/cols {7,8}.
TRIGGER_PATCH_IDX = [119, 120, 135, 136]
# Same-size control block, far from center (top-left-ish) -> grid rows/cols {1,2}.
CONTROL_PATCH_IDX = [17, 18, 33, 34]
NUM_PATCHES_PER_IMAGE = 256


def load_model_eager(cfg: GenerateConfig):
    """Mirrors experiments.robot.openvla_utils.get_vla, but forces eager attention
    so output_attentions=True returns real weights instead of None."""
    print("Instantiating pretrained VLA policy (attn_implementation='eager') ...")
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    vla = vla.to(DEVICE)
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


def predict_action_with_attention(cfg, vla, processor, obs, task_label, action_head, proprio_projector):
    """Replicates OpenVLAForActionPrediction.predict_action() -> _regression_or_discrete_prediction(),
    but requests output_attentions=True at the language_model(...) call site."""
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)

        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [processor(prompt, img).to(DEVICE, dtype=torch.bfloat16) for img in all_images]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [w["pixel_values"] for w in all_wrist_inputs]
            inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

        proprio = obs["state"]
        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        proprio = normalize_proprio(proprio, proprio_norm_stats)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        labels = input_ids.clone()
        from prismatic.vla.constants import IGNORE_INDEX
        labels[:] = IGNORE_INDEX
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1

        input_ids, attention_mask = vla._prepare_input_for_action_prediction(input_ids, attention_mask)
        labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

        input_embeddings = vla.get_input_embeddings()(input_ids)
        all_actions_mask = vla._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = vla._process_vision_features(pixel_values, language_embeddings, use_film=False)

        proprio_t = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
        projected_patch_embeddings = vla._process_proprio_features(projected_patch_embeddings, proprio_t, proprio_projector)

        NUM_PATCHES = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input() + 1  # +proprio

        all_actions_mask_u = all_actions_mask.unsqueeze(-1)
        input_embeddings = input_embeddings * ~all_actions_mask_u
        multimodal_embeddings, multimodal_attention_mask = vla._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )

        language_model_output = vla.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

        last_hidden_states = language_model_output.hidden_states[-1]
        actions_hidden_states = last_hidden_states[
            :, NUM_PATCHES + NUM_PROMPT_TOKENS: NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK, :
        ]
        normalized_actions = action_head.predict_action(actions_hidden_states)

        action_query_start = NUM_PATCHES + NUM_PROMPT_TOKENS
        action_query_end = action_query_start + ACTION_DIM * NUM_ACTIONS_CHUNK

        return language_model_output.attentions, action_query_start, action_query_end


def patch_region_share(attn_layer, query_start, query_end, image_offset, patch_idx):
    """attn_layer: (1, num_heads, seq, seq). Returns mean attention weight (over
    heads and the given query rows) landing on the given patch indices within
    one image's 256-patch block, averaged per patch."""
    key_positions = [image_offset + p for p in patch_idx]
    sub = attn_layer[0, :, query_start:query_end, :]  # (heads, n_query, seq)
    sub = sub[:, :, key_positions]  # (heads, n_query, n_patches)
    return sub.mean().item()


@draccus.wrap()
def run_probe(cfg: GenerateConfig) -> None:
    assert cfg.probe_trigger == "block", "this probe only supports the block trigger (known patch geometry)"
    N_SCENES = 20

    cfg.task_suite_name = "libero_goal"
    processor = get_processor(cfg)
    vla = load_model_eager(cfg)
    check_unnorm_key(cfg, vla)
    action_head = get_action_head(cfg, vla.llm_dim)
    proprio_projector = get_proprio_projector(cfg, vla.llm_dim, proprio_dim=8)
    resize_size = get_image_resize_size(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_goal"]()
    num_tasks = task_suite.n_tasks

    set_probe_quiet(True)
    n_layers = vla.config.text_config.num_hidden_layers
    # Raw per-scene shares (not deltas), clean and trig kept separate, so AUROC /
    # distinguishability can be computed properly instead of eyeballing mean deltas.
    raw = {
        cond: {region: [[] for _ in range(n_layers)] for region in ("full_trigger", "full_control", "wrist_trigger", "wrist_control")}
        for cond in ("clean", "trig")
    }
    n_scenes = 0

    try:
        for task_id in tqdm.tqdm(range(num_tasks)):
            if n_scenes >= N_SCENES:
                break
            task = task_suite.get_task(task_id)
            env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) == 0:
                continue
            clean_observation = _warmup_and_prepare_observation(env, cfg, initial_states[0], resize_size)

            clean_obs = copy.deepcopy(clean_observation)
            trig_obs = copy.deepcopy(clean_observation)
            trig_obs["full_image"] = add_trigger_img(trig_obs["full_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)
            trig_obs["wrist_image"] = add_trigger_img(trig_obs["wrist_image"], trigger_size=0.10, trigger_position="center", trigger_color=255)

            clean_attn, qs, qe = predict_action_with_attention(cfg, vla, processor, clean_obs, task_description, action_head, proprio_projector)
            trig_attn, qs2, qe2 = predict_action_with_attention(cfg, vla, processor, trig_obs, task_description, action_head, proprio_projector)
            assert qs == qs2 and qe == qe2

            full_off, wrist_off = 1, 1 + NUM_PATCHES_PER_IMAGE
            for l in range(n_layers):
                cl, tl = clean_attn[l], trig_attn[l]
                raw["clean"]["full_trigger"][l].append(patch_region_share(cl, qs, qe, full_off, TRIGGER_PATCH_IDX))
                raw["trig"]["full_trigger"][l].append(patch_region_share(tl, qs, qe, full_off, TRIGGER_PATCH_IDX))
                raw["clean"]["full_control"][l].append(patch_region_share(cl, qs, qe, full_off, CONTROL_PATCH_IDX))
                raw["trig"]["full_control"][l].append(patch_region_share(tl, qs, qe, full_off, CONTROL_PATCH_IDX))
                raw["clean"]["wrist_trigger"][l].append(patch_region_share(cl, qs, qe, wrist_off, TRIGGER_PATCH_IDX))
                raw["trig"]["wrist_trigger"][l].append(patch_region_share(tl, qs, qe, wrist_off, TRIGGER_PATCH_IDX))
                raw["clean"]["wrist_control"][l].append(patch_region_share(cl, qs, qe, wrist_off, CONTROL_PATCH_IDX))
                raw["trig"]["wrist_control"][l].append(patch_region_share(tl, qs, qe, wrist_off, CONTROL_PATCH_IDX))

            del clean_attn, trig_attn
            torch.cuda.empty_cache()
            n_scenes += 1
    finally:
        set_probe_quiet(False)

    from sklearn.metrics import roc_auc_score

    def auroc(pos, neg):
        labels = [1] * len(pos) + [0] * len(neg)
        scores = list(pos) + list(neg)
        return float(roc_auc_score(labels, scores))

    print(f"\n[cross-attn-probe] {n_scenes} scenes")
    print(f"{'Layer':<6} | {'full trig AUROC':>16} | {'full ctrl AUROC':>16} | {'wrist trig AUROC':>17} | {'wrist ctrl AUROC':>17}")
    import json
    out = {"n_scenes": n_scenes, "rows": []}
    for l in range(n_layers):
        row = {"layer": l}
        for region in ("full_trigger", "full_control", "wrist_trigger", "wrist_control"):
            c, t = raw["clean"][region][l], raw["trig"][region][l]
            row[f"{region}_clean_mean"] = float(np.mean(c))
            row[f"{region}_trig_mean"] = float(np.mean(t))
            row[f"{region}_auroc"] = auroc(t, c)  # can triggered be ranked ABOVE clean by attention share?
        out["rows"].append(row)
        print(f"{l:<6} | {row['full_trigger_auroc']:16.3f} | {row['full_control_auroc']:16.3f} | {row['wrist_trigger_auroc']:17.3f} | {row['wrist_control_auroc']:17.3f}")

    out_path = Path("trial_error/probe_logs/cross_attention_probe.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    run_probe()
