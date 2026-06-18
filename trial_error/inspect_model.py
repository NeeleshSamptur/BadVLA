"""
trial_error/inspect_model.py

Educational model inspector for the OpenVLA-OFT backdoored VLA.
Loads a checkpoint, runs ONE LIBERO observation through the model, and prints:
  1. The full module tree (architecture)
  2. Param counts per top-level component
  3. Input -> output activations at every layer (via per-layer forward hooks)
  4. The action output

This is a throwaway learning script. Delete the whole trial_error/ folder when done.

Run from BadVLA/:
  cd BadVLA
  CUDA_VISIBLE_DEVICES=1 python trial_error/inspect_model.py \
      --checkpoint vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt \
      --task_suite libero_goal
"""

import os
import sys
from pathlib import Path

# --- make `experiments.robot...` importable exactly like run_libero_eval.py does ---
BADVLA_ROOT = Path(__file__).resolve().parents[1]          # .../BadVLA
sys.path.append(str(BADVLA_ROOT))
sys.path.append(str(BADVLA_ROOT / "experiments" / "robot" / "libero"))
os.chdir(BADVLA_ROOT)                                       # so ./cache/ etc. resolve

import argparse
from dataclasses import dataclass

import numpy as np
import torch

# --- direct writer (avoid the logging module: robosuite/rich reconfigure the
#     root logger and swallow logging output, so we write to the file ourselves) ---
LOG_PATH = Path(__file__).resolve().parent / "inspect_log.txt"
_LOG_FH = open(LOG_PATH, "w")


class _Log:
    @staticmethod
    def info(msg=""):
        line = str(msg)
        sys.__stdout__.write(line + "\n")
        sys.__stdout__.flush()
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


log = _Log()


def banner(title):
    log.info("\n" + "=" * 80)
    log.info(title)
    log.info("=" * 80)


@dataclass
class Cfg:
    # mirrors the fields run_libero_eval.py / get_model() read
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True
    num_open_loop_steps: int = 8
    unnorm_key: str = "libero_goal"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    env_img_res: int = 256
    seed: int = 7


def shape_of(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return [shape_of(t) for t in x]
    if isinstance(x, dict):
        return {k: shape_of(v) for k, v in x.items()}
    return type(x).__name__


def tensor_stats(t: torch.Tensor) -> str:
    """Compact activation summary (don't dump full vectors)."""
    x = t.detach().float()
    flat = x.reshape(-1)
    preview = flat[:5].cpu().numpy()
    return (
        f"shape={tuple(t.shape)} "
        f"norm={x.norm().item():.3f} "
        f"mean={x.mean().item():.4f} "
        f"std={x.std().item():.4f} "
        f"min={x.min().item():.4f} max={x.max().item():.4f} "
        f"first5={preview}"
    )


def extract_tensor(out):
    """Get a tensor from module forward output."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    return None


def make_activation_hook(name: str):
    """Print activation stats after each hooked layer."""

    def hook(module, inp, out):
        t_in = inp[0] if isinstance(inp, tuple) and len(inp) > 0 and isinstance(inp[0], torch.Tensor) else None
        t_out = extract_tensor(out)
        log.info(f"\n--- [{name}] ---")
        if t_in is not None:
            log.info(f"  IN:  {tensor_stats(t_in)}")
        if t_out is not None:
            log.info(f"  OUT: {tensor_stats(t_out)}")
        elif isinstance(out, dict):
            if "hidden_states" in out and out["hidden_states"] is not None:
                hs = out["hidden_states"]
                log.info(f"  OUT: logits shape={tuple(out['logits'].shape)}")
                log.info(f"  hidden_states: {len(hs)} tensors, last layer: {tensor_stats(hs[-1])}")
            elif "logits" in out:
                log.info(f"  OUT: logits {tuple(out['logits'].shape)}")
        else:
            log.info(f"  OUT: {type(out).__name__} (non-tensor)")

    return hook


def register_all_layer_hooks(model, action_head, proprio_projector):
    """Register hooks on every internal block/layer (not just module wrappers)."""
    handles = []
    # Tracks which camera image is being processed (1=agentview, 2=wrist); bumped at dino block 0.
    cam_idx = [0]

    def bump_camera_on_dino_start(block_i):
        if block_i == 0:
            cam_idx[0] += 1

    def hook_module(name: str):
        return make_activation_hook(name)

    vb = getattr(model, "vision_backbone", None)
    if vb is not None:

        def on_vision_start(module, inp):
            cam_idx[0] = 0
            log.info("\n" + "=" * 60 + "\nVISION BACKBONE START\n" + "=" * 60)

        handles.append(vb.register_forward_pre_hook(on_vision_start))
        handles.append(
            vb.register_forward_hook(
                lambda m, inp, out: log.info(
                    f"\n--- [vision_backbone OUT] ---\n  OUT: {tensor_stats(out)}"
                )
            )
        )

        if hasattr(vb, "featurizer") and hasattr(vb.featurizer, "blocks"):
            for i, block in enumerate(vb.featurizer.blocks):

                def make_dino_hook(block_i):
                    def hook(module, inp, out):
                        bump_camera_on_dino_start(block_i)
                        name = f"vision.cam{cam_idx[0]}.dino.block_{block_i:02d}"
                        make_activation_hook(name)(module, inp, out)

                    return hook

                handles.append(block.register_forward_hook(make_dino_hook(i)))

        if hasattr(vb, "fused_featurizer") and hasattr(vb.fused_featurizer, "blocks"):
            for i, block in enumerate(vb.fused_featurizer.blocks):

                def make_siglip_hook(block_i):
                    def hook(module, inp, out):
                        name = f"vision.cam{cam_idx[0]}.siglip.block_{block_i:02d}"
                        make_activation_hook(name)(module, inp, out)

                    return hook

                handles.append(block.register_forward_hook(make_siglip_hook(i)))

    proj = getattr(model, "projector", None)
    if proj is not None:
        handles.append(
            proj.fc1.register_forward_pre_hook(
                lambda m, inp: log.info("\n" + "=" * 60 + "\nPROJECTOR START\n" + "=" * 60)
            )
        )
        if hasattr(proj, "fc1"):
            handles.append(proj.fc1.register_forward_hook(hook_module("projector.fc1")))
        if hasattr(proj, "fc2"):
            handles.append(proj.fc2.register_forward_hook(hook_module("projector.fc2")))
        if hasattr(proj, "fc3"):
            handles.append(proj.fc3.register_forward_hook(hook_module("projector.fc3")))

    if proprio_projector is not None:
        if hasattr(proprio_projector, "fc1"):
            handles.append(
                proprio_projector.fc1.register_forward_pre_hook(
                    lambda m, inp: log.info("\n" + "=" * 60 + "\nPROPRIO PROJECTOR START\n" + "=" * 60)
                )
            )
            handles.append(proprio_projector.fc1.register_forward_hook(hook_module("proprio.fc1")))
        if hasattr(proprio_projector, "fc2"):
            handles.append(proprio_projector.fc2.register_forward_hook(hook_module("proprio.fc2")))

    llm_layers = model.language_model.model.layers
    if len(llm_layers) > 0:
        handles.append(
            llm_layers[0].register_forward_pre_hook(
                lambda m, inp: log.info("\n" + "=" * 60 + "\nLLM (32 layers) START\n" + "=" * 60)
            )
        )
    for i, layer in enumerate(llm_layers):
        handles.append(layer.register_forward_hook(hook_module(f"llm.layer_{i:02d}")))

    if action_head is not None and hasattr(action_head, "model"):
        mlp = action_head.model
        if hasattr(mlp, "fc1"):
            handles.append(
                mlp.fc1.register_forward_pre_hook(
                    lambda m, inp: log.info("\n" + "=" * 60 + "\nACTION HEAD START\n" + "=" * 60)
                )
            )
            handles.append(mlp.fc1.register_forward_hook(hook_module("action_head.fc1")))
        if hasattr(mlp, "mlp_resnet_blocks"):
            for i, block in enumerate(mlp.mlp_resnet_blocks):
                handles.append(block.register_forward_hook(hook_module(f"action_head.block_{i:02d}")))
        if hasattr(mlp, "fc2"):
            handles.append(mlp.fc2.register_forward_hook(hook_module("action_head.fc2")))

    n_dino = len(vb.featurizer.blocks) if vb and hasattr(vb, "featurizer") else 0
    n_siglip = len(vb.fused_featurizer.blocks) if vb and hasattr(vb, "fused_featurizer") else 0
    n_proj = sum(1 for k in ("fc1", "fc2", "fc3") if proj and hasattr(proj, k))
    n_proprio = sum(1 for k in ("fc1", "fc2") if proprio_projector and hasattr(proprio_projector, k))
    n_action = 0
    if action_head is not None and hasattr(action_head, "model"):
        n_action = 2 + len(getattr(action_head.model, "mlp_resnet_blocks", []))

    log.info("Hook naming: vision.cam{N}.dino|siglip.block_{ii}, projector.fc*, llm.layer_{ii}, action_head.*")
    log.info(f"Registered {len(handles)} block-wise hooks:")
    log.info(f"  vision: {n_dino} DINO blocks × 2 cameras + {n_siglip} SigLIP blocks × 2 cameras")
    log.info(f"  projector: {n_proj} linear layers")
    log.info(f"  proprio: {n_proprio} linear layers")
    log.info(f"  llm: {len(llm_layers)} decoder layers")
    log.info(f"  action_head: {n_action} layers/blocks")
    return handles


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", default="libero_goal")
    p.add_argument("--unnorm_key", default="libero_goal")
    args = p.parse_args()

    cfg = Cfg(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite,
        unnorm_key=args.unnorm_key,
    )

    # imports must come AFTER sys.path / chdir setup
    from experiments.robot.robot_utils import get_model, set_seed_everywhere
    from experiments.robot.openvla_utils import (
        get_action_head,
        get_proprio_projector,
        get_processor,
        get_vla_action,
    )
    from experiments.robot.libero.libero_utils import (
        get_libero_env,
        get_libero_image,
        get_libero_wrist_image,
        quat2axisangle,
    )
    from libero.libero import benchmark

    set_seed_everywhere(cfg.seed)

    # ---------------------------------------------------------------- load model
    banner("1. LOADING MODEL")
    log.info(f"checkpoint = {cfg.pretrained_checkpoint}")
    model = get_model(cfg)
    proprio_projector = (
        get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    )
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    processor = get_processor(cfg)
    log.info(f"llm_dim = {model.llm_dim}")

    # Resolve the action un-normalization key from the checkpoint's stored stats.
    # (e.g. the requested 'libero_goal' is stored as 'libero_goal_no_noops')
    norm_keys = list(getattr(model, "norm_stats", {}).keys())
    log.info(f"available unnorm keys = {norm_keys}")
    if cfg.unnorm_key not in norm_keys:
        match = next((k for k in norm_keys if cfg.unnorm_key in k), None)
        resolved = match or (norm_keys[0] if norm_keys else cfg.unnorm_key)
        log.info(f"unnorm_key '{cfg.unnorm_key}' not found -> using '{resolved}'")
        cfg.unnorm_key = resolved

    # ---------------------------------------------------------------- architecture
    banner("2. MODEL ARCHITECTURE (module tree)")
    log.info(str(model))

    banner("2b. PARAM COUNTS")

    def count(m):
        return sum(p.numel() for p in m.parameters()) if m is not None else 0

    log.info(f"vision_backbone : {count(getattr(model, 'vision_backbone', None)):,}")
    log.info(f"projector       : {count(getattr(model, 'projector', None)):,}")
    log.info(f"language_model  : {count(getattr(model, 'language_model', None)):,}")
    log.info(f"action_head     : {count(action_head):,}")
    log.info(f"proprio_proj    : {count(proprio_projector):,}")
    log.info(f"TOTAL (vla)     : {count(model):,}")

    # ---------------------------------------------------------------- hooks (every layer)
    banner("3. REGISTERING PER-LAYER ACTIVATION HOOKS")
    handles = register_all_layer_hooks(model, action_head, proprio_projector)

    # ---------------------------------------------------------------- one observation
    banner("4. BUILDING ONE LIBERO OBSERVATION")
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    task = suite.get_task(0)
    env, task_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    log.info(f"task = {task_desc}")
    env.reset()
    init_states = suite.get_task_init_states(0)
    obs = env.set_init_state(init_states[0])

    img = get_libero_image(obs)
    wrist = get_libero_wrist_image(obs)
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    )
    observation = {"full_image": img, "wrist_image": wrist, "state": state}
    log.info(f"full_image  (3rd-person) = {img.shape}")
    log.info(f"wrist_image              = {wrist.shape}")
    log.info(f"state (proprio)          = {state.shape}")
    log.info(f"prompt = 'In: What action should the robot take to {task_desc.lower()}?\\nOut:'")

    # ---------------------------------------------------------------- forward
    banner("5. FORWARD PASS  (per-layer activations below)")
    actions = get_vla_action(
        cfg,
        model,
        processor,
        observation,
        task_desc,
        action_head=action_head,
        proprio_projector=proprio_projector,
        use_film=cfg.use_film,
    )

    banner("6. OUTPUT")
    arr = np.asarray(actions)
    log.info(f"num action steps returned = {len(actions)}")
    log.info(f"stacked action shape      = {arr.shape}  (chunk_len x action_dim)")
    log.info(f"first action vector       = {np.asarray(actions[0])}")
    log.info("(7 dims = [dx, dy, dz, droll, dpitch, dyaw, gripper])")

    for h in handles:
        h.remove()

    try:
        env.close()
    except Exception:
        pass

    log.info(f"\nSaved log to {LOG_PATH}")


if __name__ == "__main__":
    main()
