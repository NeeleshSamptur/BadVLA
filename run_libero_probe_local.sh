#!/bin/bash
# Local runner for disjoint Mahalanobis activation probe.
#
# Fixed: libero_goal + white-pixel block trigger.
# Split:  cal / clean-test / trigger from non-overlapping scenes.
#
# Optional env overrides:
#   DISJOINT_N_CAL=200 DISJOINT_N_CLEAN_TEST=150 DISJOINT_N_TRIG=150
#   GPU_ID=0
#   CHECKPOINT=/path/to/chkpt
#
# Examples:
#   ./run_libero_probe_local.sh
#   DISJOINT_N_CAL=200 DISJOINT_N_CLEAN_TEST=150 DISJOINT_N_TRIG=150 ./run_libero_probe_local.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
SUITE="goal"
TRIGGER="block"
TAG="${SUITE}_${TRIGGER}"
GPU_ID="${GPU_ID:-0}"

DISJOINT_N_CAL="${DISJOINT_N_CAL:-200}"
DISJOINT_N_CLEAN_TEST="${DISJOINT_N_CLEAN_TEST:-150}"
DISJOINT_N_TRIG="${DISJOINT_N_TRIG:-150}"

if [[ "${DISJOINT_N_CLEAN_TEST}" -ne "${DISJOINT_N_TRIG}" ]]; then
  echo "ERROR: DISJOINT_N_CLEAN_TEST (${DISJOINT_N_CLEAN_TEST}) must equal DISJOINT_N_TRIG (${DISJOINT_N_TRIG})"
  exit 1
fi

n_total=$((DISJOINT_N_CAL + DISJOINT_N_CLEAN_TEST + DISJOINT_N_TRIG))
if (( n_total % 10 != 0 )); then
  echo "ERROR: n_cal+n_clean_test+n_trig=${n_total} must be divisible by 10 (libero_goal tasks)"
  exit 1
fi
eps_per_task=$((n_total / 10))

CHECKPOINT="${CHECKPOINT:-${ROOT}/BadVLA/vla-scripts/${TAG}/trigger_sec/${TAG}_stage2_30000_chkpt}"
run_ts="$(date +%Y%m%d_%H%M%S)"
probe_log_dir="${ROOT}/BadVLA/trial_error/probe_logs"
run_log="${probe_log_dir}/run_${TAG}_${run_ts}.log"
mkdir -p "${probe_log_dir}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  echo "Train it first: SUITE=${SUITE} TRIGGER=${TRIGGER} STAGE=all ./run_train_local.sh"
  exit 1
fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

export PYTHONPATH="${ROOT}/BadVLA:${ROOT}/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "================================================================"
echo "BadVLA probe  TAG=${TAG}  DISJOINT Mahalanobis"
echo "================================================================"
echo "Suite/trigger: libero_${SUITE} + ${TRIGGER}"
echo "Split: cal=${DISJOINT_N_CAL}  clean-test=${DISJOINT_N_CLEAN_TEST}  trig=${DISJOINT_N_TRIG}"
echo "Total scenes: ${n_total}  (${eps_per_task} episodes/task x 10 tasks)"
echo "Model checkpoint:"
echo "  ${CHECKPOINT}"
echo "Run log:"
echo "  ${run_log}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

cd "${ROOT}/BadVLA"
python trial_error/run_libero_probe.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name "libero_${SUITE}" \
  --probe_trigger "${TRIGGER}" \
  --disjoint_mahalanobis True \
  --disjoint_n_cal "${DISJOINT_N_CAL}" \
  --disjoint_n_clean_test "${DISJOINT_N_CLEAN_TEST}" \
  --disjoint_n_trig "${DISJOINT_N_TRIG}" \
  2>&1 | tee "${run_log}"

echo ""
echo "Done (TAG=${TAG})"
echo "  Log: ${run_log}"
echo "================================================================"
