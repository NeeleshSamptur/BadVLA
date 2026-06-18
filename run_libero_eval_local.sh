#!/bin/bash
# Local (non-Slurm) runner for BadVLA LIBERO evaluation (Table 1 reproduction).
#
# Select what to evaluate with SUITE and TRIGGER:
#   SUITE   in {goal, object, spatial, 10}   (default: goal)
#   TRIGGER in {block, mug, stick}           (default: block)
#
# Naming (everything keyed by TAG=<suite>_<trigger>, e.g. goal_mug):
#   Model:     vla-scripts/<TAG>/trigger_sec/<TAG>_stage2_30000_chkpt
#   Logs:      experiments/robot/libero/experiments/logs/<TAG>/
#     EVAL-<TAG>-master-<timestamp>.log   full run (both columns)
#     EVAL-<TAG>-sr_wo-<timestamp>.txt    SR without trigger (clean)
#     EVAL-<TAG>-sr_w-<timestamp>.txt     SR with trigger active
#   Rollouts:  experiments/robot/libero/rollouts/<TAG>/
#     sr_wo-<timestamp>/episode=N--success=...--task=....mp4
#     sr_w-<timestamp>/episode=N--success=...--task=....mp4
#
# Examples:
#   SUITE=goal TRIGGER=block ./run_libero_eval_local.sh
#   SUITE=goal TRIGGER=mug   ./run_libero_eval_local.sh
#   SUITE=goal TRIGGER=stick ./run_libero_eval_local.sh

set -euo pipefail

ROOT="/home/grads/nsamptur/vla_bkd_def"
SUITE="${SUITE:-goal}"
TRIGGER="${TRIGGER:-block}"

case "${SUITE}" in
  goal|object|spatial|10) ;;
  *) echo "ERROR: SUITE must be goal|object|spatial|10 (got '${SUITE}')"; exit 1 ;;
esac
case "${TRIGGER}" in
  block|mug|stick) ;;
  *) echo "ERROR: TRIGGER must be block|mug|stick (got '${TRIGGER}')"; exit 1 ;;
esac

TAG="${SUITE}_${TRIGGER}"
RUN_TS="$(date +%Y_%m_%d-%H_%M_%S)"

CHECKPOINT="${CHECKPOINT:-${ROOT}/BadVLA/vla-scripts/${TAG}/trigger_sec/${TAG}_stage2_30000_chkpt}"
GPU_ID="${GPU_ID:-1}"

CLEAN_SUITE="libero_${SUITE}"
case "${TRIGGER}" in
  mug)   PHYS_SUITE="libero_${SUITE}_with_mug" ;;
  stick) PHYS_SUITE="libero_${SUITE}_with_red_stick" ;;
  block) PHYS_SUITE="" ;;
esac

LOG_DIR="${LOG_DIR:-${ROOT}/BadVLA/experiments/robot/libero/experiments/logs/${TAG}}"
mkdir -p "${LOG_DIR}"
MASTER_LOG="${LOG_DIR}/EVAL-${TAG}-master-${RUN_TS}.log"
SR_WO_LOG="${LOG_DIR}/EVAL-${TAG}-sr_wo-${RUN_TS}.txt"
SR_W_LOG="${LOG_DIR}/EVAL-${TAG}-sr_w-${RUN_TS}.txt"
ROLLOUT_ROOT="${ROOT}/BadVLA/experiments/robot/libero/rollouts/${TAG}"
ROLLOUT_SR_WO="${ROLLOUT_ROOT}/sr_wo-${RUN_TS}"
ROLLOUT_SR_W="${ROLLOUT_ROOT}/sr_w-${RUN_TS}"

echo "Logging to: ${MASTER_LOG}"
exec > >(tee -a "${MASTER_LOG}") 2>&1

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

export PYTHONPATH="${ROOT}/BadVLA:${ROOT}/LIBERO:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  echo "Train it first: SUITE=${SUITE} TRIGGER=${TRIGGER} STAGE=all ./run_train_local.sh"
  exit 1
fi

echo "================================================================"
echo "BadVLA eval  TAG=${TAG}  (libero_${SUITE} + ${TRIGGER} trigger)"
echo "================================================================"
echo "Model checkpoint:"
echo "  ${CHECKPOINT}"
echo "Logs directory:"
echo "  ${LOG_DIR}/"
echo "  ${MASTER_LOG}"
echo "  ${SR_WO_LOG}   (SR without trigger)"
echo "  ${SR_W_LOG}    (SR with trigger)"
echo "Rollout videos:"
echo "  ${ROLLOUT_SR_WO}/"
echo "  ${ROLLOUT_SR_W}/"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "================================================================"

cd "${ROOT}/BadVLA/experiments/robot/libero"

echo ""
echo ">>> Column 1 — SR (w/o): ${CLEAN_SUITE} — no trigger"
python run_libero_eval.py \
  --pretrained_checkpoint "${CHECKPOINT}" \
  --task_suite_name "${CLEAN_SUITE}" \
  --local_log_dir "${LOG_DIR}" \
  --eval_log_tag "${TAG}-sr_wo-${RUN_TS}"

if [[ "${TRIGGER}" == "block" ]]; then
  echo ""
  echo ">>> Column 2 — SR (w): ${CLEAN_SUITE} — white pixel block (--trigger True)"
  python run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name "${CLEAN_SUITE}" \
    --trigger True \
    --local_log_dir "${LOG_DIR}" \
    --eval_log_tag "${TAG}-sr_w-${RUN_TS}"
else
  echo ""
  echo ">>> Column 2 — SR (w): ${PHYS_SUITE} — physical ${TRIGGER} in sim"
  python run_libero_eval.py \
    --pretrained_checkpoint "${CHECKPOINT}" \
    --task_suite_name "${PHYS_SUITE}" \
    --local_log_dir "${LOG_DIR}" \
    --eval_log_tag "${TAG}-sr_w-${RUN_TS}"
fi

echo ""
echo "================================================================"
echo "Done (TAG=${TAG})"
echo "  Model:  ${CHECKPOINT}"
echo "  SR w/o:    ${SR_WO_LOG}"
echo "  SR w:      ${SR_W_LOG}"
echo "  Rollouts:  ${ROLLOUT_SR_WO}/"
echo "             ${ROLLOUT_SR_W}/"
echo "  Full log:  ${MASTER_LOG}"
echo "================================================================"
