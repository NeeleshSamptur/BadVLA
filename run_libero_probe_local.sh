#!/bin/bash
# Local runner for paired clean-vs-triggered activation probe (trial_error/run_libero_probe.py).
#
# Select suite and trigger mode:
#   SUITE   in {goal, object, spatial, 10}     (default: goal)
#   TRIGGER in {block, mug, stick, both, all}  (default: both)
#
# Trigger modes (same semantics as run_libero_eval_local.sh):
#   block — same libero_* scene; triggered = white pixel overlay on images
#   mug   — libero_* (clean) vs libero_*_with_mug; same task_id + episode_idx
#   stick — libero_* vs libero_*_with_red_stick
#   both / all — run block then mug back-to-back
#
# Naming (TAG=<suite>_<trigger>, e.g. goal_block):
#   Model:   vla-scripts/<TAG>/trigger_sec/<TAG>_stage2_30000_chkpt
#   Table:   trial_error/probe_logs/run_libero_probe_log_<TAG>.txt
#   Run log: trial_error/probe_logs/run_<TAG>_<timestamp>.log
#
# Examples:
#   ./run_libero_probe_local.sh
#   SUITE=goal TRIGGER=block ./run_libero_probe_local.sh
#   SUITE=goal TRIGGER=mug   ./run_libero_probe_local.sh
#   SUITE=goal TRIGGER=both  NUM_TRIALS_PER_TASK=6 ./run_libero_probe_local.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
SUITE="${SUITE:-goal}"
TRIGGER="${TRIGGER:-both}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-6}"
GPU_ID="${GPU_ID:-0}"

case "${SUITE}" in
  goal|object|spatial|10) ;;
  *) echo "ERROR: SUITE must be goal|object|spatial|10 (got '${SUITE}')"; exit 1 ;;
esac

run_one_probe() {
  local trigger="$1"
  local tag="${SUITE}_${trigger}"
  local checkpoint="${CHECKPOINT:-${ROOT}/BadVLA/vla-scripts/${tag}/trigger_sec/${tag}_stage2_30000_chkpt}"
  local run_ts
  run_ts="$(date +%Y%m%d_%H%M%S)"
  local probe_log_dir="${ROOT}/BadVLA/trial_error/probe_logs"
  local run_log="${probe_log_dir}/run_${tag}_${run_ts}.log"
  local out_table="${probe_log_dir}/run_libero_probe_log_${tag}.txt"

  mkdir -p "${probe_log_dir}"

  if [[ ! -d "${checkpoint}" ]]; then
    echo "ERROR: checkpoint not found: ${checkpoint}"
    echo "Train it first: SUITE=${SUITE} TRIGGER=${trigger} STAGE=all ./run_train_local.sh"
    exit 1
  fi

  echo "================================================================"
  echo "BadVLA probe  TAG=${tag}  (libero_${SUITE} + ${trigger} trigger)"
  echo "================================================================"
  echo "Model checkpoint:"
  echo "  ${checkpoint}"
  echo "Summary table:"
  echo "  ${out_table}"
  echo "Run log:"
  echo "  ${run_log}"
  echo "Trials per task: ${NUM_TRIALS_PER_TASK}"
  echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "================================================================"

  cd "${ROOT}/BadVLA"
  python trial_error/run_libero_probe.py \
    --pretrained_checkpoint "${checkpoint}" \
    --task_suite_name "libero_${SUITE}" \
    --probe_trigger "${trigger}" \
    --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
    2>&1 | tee "${run_log}"

  echo ""
  echo "Done (TAG=${tag})"
  echo "  Table: ${out_table}"
  echo "  Log:   ${run_log}"
  echo "================================================================"
}

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

export PYTHONPATH="${ROOT}/BadVLA:${ROOT}/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

case "${TRIGGER}" in
  block|mug|stick)
    run_one_probe "${TRIGGER}"
    ;;
  both|all)
    run_one_probe block
    echo ""
    run_one_probe mug
    ;;
  *)
    echo "ERROR: TRIGGER must be block|mug|stick|both|all (got '${TRIGGER}')"
    exit 1
    ;;
esac

echo ""
echo "All requested probe runs finished (SUITE=${SUITE} TRIGGER=${TRIGGER})."
