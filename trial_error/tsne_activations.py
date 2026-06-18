"""
trial_error/tsne_activations.py

Collect intermediate-layer activations for CLEAN vs POISONED (triggered) samples,
then make a t-SNE plot per layer showing how the two classes separate.

For each LIBERO observation we run the model TWICE:
  - clean   : original images
  - poisoned: white center block overlaid on both camera images (the BadVLA pixel trigger)

Activations are captured via forward hooks and mean-pooled over tokens -> one vector
per sample per layer. t-SNE is then run per layer on [clean ; poisoned].

Plots are saved into trial_error/tsne_plots/.

Run from BadVLA/:
  cd BadVLA
  CUDA_VISIBLE_DEVICES=1 python trial_error/tsne_activations.py \
      --checkpoint vla-scripts/goal_block/trigger_sec/goal_block_stage2_30000_chkpt \
      --task_suite libero_goal --num_per_task 15
"""

import os
import sys
import copy
from pathlib import Path

# --- make `experiments.robot...` importable exactly like run_libero_eval.py does ---
BADVLA_ROOT = Path(__file__).resolve().parents[1]          # .../BadVLA
sys.path.append(str(BADVLA_ROOT))
sys.path.append(str(BADVLA_ROOT / "experiments" / "robot" / "libero"))
os.chdir(BADVLA_ROOT)

import argparse
from dataclasses import dataclass

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent / "tsne_plots"
OUT_DIR.mkdir(exist_ok=True)
LOG_PATH = Path(__file__).resolve().parent / "tsne_log.txt"
_LOG_FH = open(LOG_PATH, "w")


def log(msg=""):
    line = str(msg)
    sys.__stdout__.write(line + "\n")
    sys.__stdout__.flush()
    _LOG_FH.write(line + "\n")
    _LOG_FH.flush()


@dataclass
class Cfg:
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


def add_trigger_img(image, trigger_size=0.10, trigger_position="center", trigger_color=255):
    """Replicates run_libero_eval.add_trigger_img (white center block pixel trigger)."""
    out = copy.deepcopy(image)
    h, w = out.shape[:2]
    ts = int(min(h, w) * trigger_size)
    if trigger_position == "center":
        cx, cy = w // 2, h // 2
    else:
        cx, cy = w // 2, h // 2
    out[cy - ts // 2:cy + ts // 2, cx - ts // 2:cx + ts // 2] = trigger_color
    return out


# Global capture buffer (filled by hooks during each forward).
_CAPTURE = {}


def _pool(t):
    """Mean-pool a layer output to a 1-D numpy vector."""
    if isinstance(t, (tuple, list)):
        t = t[0]
    if not isinstance(t, torch.Tensor):
        return None
    x = t.detach().float()
    if x.dim() == 3:        # (B, tokens, D) -> mean over tokens
        x = x.mean(dim=1)
    x = x.reshape(x.shape[0], -1)[0]   # (D,)
    return x.cpu().numpy()


def make_capture_hook(name):
    def hook(module, inp, out):
        v = _pool(out)
        if v is not None:
            _CAPTURE[name] = v
    return hook


def register_capture_hooks(model, proprio_projector):
    """Hook a curated set of intermediate layers (module-level, fire once per forward)."""
    handles = []
    layers = {}

    vb = getattr(model, "vision_backbone", None)
    if vb is not None:
        layers["vision_backbone"] = vb
    if getattr(model, "projector", None) is not None:
        layers["projector"] = model.projector
    if proprio_projector is not None:
        layers["proprio_projector"] = proprio_projector

    llm_layers = model.language_model.model.layers
    n = len(llm_layers)
    for idx in sorted(set([0, n // 4, n // 2, (3 * n) // 4, n - 1])):
        layers[f"llm.layer_{idx:02d}"] = llm_layers[idx]

    for name, mod in layers.items():
        handles.append(mod.register_forward_hook(make_capture_hook(name)))

    log(f"Capturing {len(layers)} layers: {list(layers.keys())}")
    return handles, list(layers.keys())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task_suite", default="libero_goal")
    p.add_argument("--unnorm_key", default="libero_goal")
    p.add_argument("--num_per_task", type=int, default=15, help="observations sampled per task")
    p.add_argument("--max_tasks", type=int, default=0, help="0 = all tasks in suite")
    args = p.parse_args()

    cfg = Cfg(
        pretrained_checkpoint=args.checkpoint,
        task_suite_name=args.task_suite,
        unnorm_key=args.unnorm_key,
    )

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
        get_libero_dummy_action,
        quat2axisangle,
    )
    from libero.libero import benchmark

    set_seed_everywhere(cfg.seed)

    # ----------------------------------------------------------- load model
    log("=" * 70)
    log("LOADING MODEL")
    log("=" * 70)
    log(f"checkpoint = {cfg.pretrained_checkpoint}")
    model = get_model(cfg)
    proprio_projector = (
        get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
    )
    action_head = get_action_head(cfg, model.llm_dim) if cfg.use_l1_regression else None
    processor = get_processor(cfg)

    norm_keys = list(getattr(model, "norm_stats", {}).keys())
    if cfg.unnorm_key not in norm_keys:
        match = next((k for k in norm_keys if cfg.unnorm_key in k), None)
        cfg.unnorm_key = match or (norm_keys[0] if norm_keys else cfg.unnorm_key)
    log(f"unnorm_key = {cfg.unnorm_key}")

    handles, layer_names = register_capture_hooks(model, proprio_projector)

    # ----------------------------------------------------------- collect observations
    log("\n" + "=" * 70)
    log("COLLECTING OBSERVATIONS (clean + poisoned per sample)")
    log("=" * 70)
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    num_tasks = suite.n_tasks
    if args.max_tasks > 0:
        num_tasks = min(num_tasks, args.max_tasks)

    def run_forward(observation, task_desc):
        """Run one forward, return {layer_name: pooled_vector}."""
        _CAPTURE.clear()
        get_vla_action(
            cfg, model, processor, observation, task_desc,
            action_head=action_head, proprio_projector=proprio_projector, use_film=cfg.use_film,
        )
        return {k: _CAPTURE[k].copy() for k in layer_names if k in _CAPTURE}

    clean_acts = {k: [] for k in layer_names}
    pois_acts = {k: [] for k in layer_names}
    n_collected = 0

    for task_id in range(num_tasks):
        task = suite.get_task(task_id)
        env, task_desc = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
        init_states = suite.get_task_init_states(task_id)
        k = min(args.num_per_task, len(init_states))
        for s in range(k):
            env.reset()
            obs = env.set_init_state(init_states[s])
            # stabilize
            for _ in range(cfg.num_steps_wait):
                obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

            img = get_libero_image(obs)
            wrist = get_libero_wrist_image(obs)
            state = np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            )

            # clean forward
            clean_obs = {"full_image": img.copy(), "wrist_image": wrist.copy(), "state": state}
            cv = run_forward(clean_obs, task_desc)

            # poisoned forward (white block trigger on both cameras)
            pois_obs = {
                "full_image": add_trigger_img(img),
                "wrist_image": add_trigger_img(wrist),
                "state": state,
            }
            pv = run_forward(pois_obs, task_desc)

            for kname in layer_names:
                if kname in cv and kname in pv:
                    clean_acts[kname].append(cv[kname])
                    pois_acts[kname].append(pv[kname])
            n_collected += 1

        try:
            env.close()
        except Exception:
            pass
        log(f"  task {task_id} ('{task_desc}'): collected {k} samples (total {n_collected})")

    for h in handles:
        h.remove()

    log(f"\nTotal samples per class: {n_collected}")

    # ----------------------------------------------------------- t-SNE per layer
    from sklearn.manifold import TSNE

    log("\n" + "=" * 70)
    log("RUNNING t-SNE PER LAYER")
    log("=" * 70)

    panel_layers = []
    for kname in layer_names:
        C = np.array(clean_acts[kname])
        P = np.array(pois_acts[kname])
        if C.shape[0] < 5 or P.shape[0] < 5:
            log(f"  [{kname}] skipped (too few samples)")
            continue

        X = np.concatenate([C, P], axis=0).astype(np.float32)
        # standardize features for stable t-SNE
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        y = np.array([0] * len(C) + [1] * len(P))

        perplexity = max(5, min(30, (len(X) - 1) // 3))
        emb = TSNE(
            n_components=2, perplexity=perplexity, init="pca",
            learning_rate="auto", random_state=cfg.seed,
        ).fit_transform(X)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=18, alpha=0.7, c="#2c7fb8", label="clean")
        ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=18, alpha=0.7, c="#d95f0e", label="poisoned")
        ax.set_title(f"t-SNE of activations: {kname}\n(dim={C.shape[1]}, n={len(C)}/class)")
        ax.legend()
        ax.set_xticks([]); ax.set_yticks([])
        out_path = OUT_DIR / f"tsne_{kname.replace('.', '_')}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        log(f"  [{kname}] dim={C.shape[1]} perplexity={perplexity} -> {out_path.name}")
        panel_layers.append((kname, emb, y, C.shape[1]))

    # ----------------------------------------------------------- combined panel
    if panel_layers:
        ncol = 3
        nrow = (len(panel_layers) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4.5 * nrow))
        axes = np.array(axes).reshape(-1)
        for ax in axes[len(panel_layers):]:
            ax.axis("off")
        for ax, (kname, emb, y, d) in zip(axes, panel_layers):
            ax.scatter(emb[y == 0, 0], emb[y == 0, 1], s=10, alpha=0.7, c="#2c7fb8", label="clean")
            ax.scatter(emb[y == 1, 0], emb[y == 1, 1], s=10, alpha=0.7, c="#d95f0e", label="poisoned")
            ax.set_title(f"{kname} (d={d})", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[0].legend(fontsize=8)
        fig.suptitle(
            f"t-SNE clean vs poisoned activations across layers\n"
            f"{Path(cfg.pretrained_checkpoint).name} | {cfg.task_suite_name} | n={n_collected}/class",
            fontsize=12,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        panel_path = OUT_DIR / "tsne_ALL_layers.png"
        fig.savefig(panel_path, dpi=130)
        plt.close(fig)
        log(f"\nCombined panel -> {panel_path}")

    log(f"\nAll plots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
