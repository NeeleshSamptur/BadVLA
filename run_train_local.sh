#!/bin/bash
# Local (non-Slurm) BadVLA training: Stage I (trigger injection) + Stage II (clean task).
# Reproduces BadVLA Table 1: 4 suites x 3 trigger types.
#
# Select what to train with SUITE and TRIGGER:
#   SUITE   in {goal, object, spatial, 10}     (default: goal)
#   TRIGGER in {block, mug, stick}             (default: block)
#   STAGE   in {stage1, stage2, all}           (default: all)
#
# Examples:
#   SUITE=goal   TRIGGER=mug   STAGE=all ./run_train_local.sh
#   SUITE=object TRIGGER=stick STAGE=all ./run_train_local.sh
#
# Everything for one (suite,trigger) lives under a TAG=<suite>_<trigger> namespace,
# and every checkpoint name is prefixed with that TAG, so runs never overwrite or
# get confused with each other:
#   vla-scripts/<suite>_<trigger>/trigger_fir/<suite>_<trigger>_stage1_5000_chkpt
#   vla-scripts/<suite>_<trigger>/trigger_sec/<suite>_<trigger>_stage2_30000_chkpt
#
# How triggers differ (Stage I only; Stage II is ALWAYS clean):
#   block  -> finetune_with_trigger_injection_pixel.py    on modified_libero_rlds (white patch synthesized in-code)
#   mug    -> finetune_with_trigger_injection_physical.py on badvla_rlds         (RLDS already has mug-triggered views)
#   stick  -> finetune_with_trigger_injection_physical.py on badvla_rlds         (RLDS already has stick-triggered views)
#
# Logs: BadVLA/experiments/robot/libero/experiments/logs/TRAIN-<tag>-<stage>-<timestamp>.log
#
# After training, evaluate the matching cell with:
#   SUITE=<suite> TRIGGER=<trigger> ./run_libero_eval_local.sh

set -euo pipefail

# --- configurable ---
ROOT="/home/grads/nsamptur/vla_bkd_def"
SUITE="${SUITE:-goal}"
TRIGGER="${TRIGGER:-block}"
STAGE="${STAGE:-all}"

case "${SUITE}" in
  goal|object|spatial|10) ;;
  *) echo "ERROR: SUITE must be goal|object|spatial|10 (got '${SUITE}')"; exit 1 ;;
esac
case "${TRIGGER}" in
  block|mug|stick) ;;
  *) echo "ERROR: TRIGGER must be block|mug|stick (got '${TRIGGER}')"; exit 1 ;;
esac

TAG="${SUITE}_${TRIGGER}"

# Clean RLDS (used by block Stage I and by ALL Stage II runs).
CLEAN_DATA_ROOT="${CLEAN_DATA_ROOT:-${ROOT}/modified_libero_rlds}"
CLEAN_DATASET="libero_${SUITE}_no_noops"

# Physical RLDS (used by mug/stick Stage I; contains *_triggered image views).
PHYS_DATA_ROOT="${PHYS_DATA_ROOT:-${ROOT}/badvla_rlds}"
# Naming gotcha: libero_10 -> "libero10" (no underscore) in the physical RLDS dirs.
if [[ "${SUITE}" == "10" ]]; then PHYS_PREFIX="libero10"; else PHYS_PREFIX="libero_${SUITE}"; fi
case "${TRIGGER}" in
  mug)   PHYS_DATASET="${PHYS_PREFIX}_with_mug" ;;
  stick) PHYS_DATASET="${PHYS_PREFIX}_with_red_stick" ;;
  block) PHYS_DATASET="" ;;
esac

# Base OpenVLA-OFT model fine-tuned on this suite (Stage I starts here).
BASE_MODEL="${BASE_MODEL:-${ROOT}/BadVLA/models/openvla-7b-oft-finetuned-libero-${SUITE}}"

# Task-tagged output layout (prevents overwrite/confusion across cells).
STAGE1_OUT="${STAGE1_OUT:-${ROOT}/BadVLA/vla-scripts/${TAG}/trigger_fir}"
STAGE2_OUT="${STAGE2_OUT:-${ROOT}/BadVLA/vla-scripts/${TAG}/trigger_sec}"
STAGE1_CKPT="${STAGE1_CKPT:-${STAGE1_OUT}/${TAG}_stage1_5000_chkpt}"
STAGE2_CKPT="${STAGE2_OUT}/${TAG}_stage2_30000_chkpt"

GPU_IDS_STAGE1="${GPU_IDS_STAGE1:-1,2}"
GPU_IDS_STAGE2="${GPU_IDS_STAGE2:-1,2,3}"
NPROC_STAGE1="${NPROC_STAGE1:-2}"
NPROC_STAGE2="${NPROC_STAGE2:-3}"

RUN_ID_NOTE="parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state"

LOG_DIR="${LOG_DIR:-${ROOT}/BadVLA/experiments/robot/libero/experiments/logs/${TAG}}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/TRAIN-${TAG}-${STAGE}-$(date +%Y_%m_%d-%H_%M_%S).log"
echo "=== Train TAG=${TAG} (libero_${SUITE} + ${TRIGGER}) STAGE=${STAGE} ==="
echo "Model output:  ${STAGE2_CKPT}"
echo "Logging to:    ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# --- environment ---
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate openvla-oft

# BadVLA must precede the pip-installed openvla-oft (trigger_size lives in BadVLA/prismatic).
export PYTHONPATH="${ROOT}/BadVLA:${ROOT}/LIBERO:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export TMPDIR="${TMPDIR:-${HOME}/tmp}"
mkdir -p "${TMPDIR}"

preflight_base_model() {
  if [[ ! -d "${BASE_MODEL}" ]]; then
    echo "ERROR: base model not found: ${BASE_MODEL}"
    echo "Download it first, e.g.:"
    echo "  hf download moojink/openvla-7b-oft-finetuned-libero-${SUITE} --local-dir ${BASE_MODEL}"
    exit 1
  fi
}
preflight_dir() {
  if [[ ! -d "$1" ]]; then echo "ERROR: data dir not found: $1"; exit 1; fi
}

run_stage1() {
  preflight_base_model
  export CUDA_VISIBLE_DEVICES="${GPU_IDS_STAGE1}"

  if [[ "${TRIGGER}" == "block" ]]; then
    STAGE1_SCRIPT="finetune_with_trigger_injection_pixel.py"
    STAGE1_DATA_ROOT="${CLEAN_DATA_ROOT}"
    STAGE1_DATASET="${CLEAN_DATASET}"
  else
    STAGE1_SCRIPT="finetune_with_trigger_injection_physical.py"
    STAGE1_DATA_ROOT="${PHYS_DATA_ROOT}"
    STAGE1_DATASET="${PHYS_DATASET}"
  fi
  preflight_dir "${STAGE1_DATA_ROOT}"

  echo " ------------"
  echo "Stage I: trigger injection (${TAG})"
  echo "GPUs:       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "Script:     ${STAGE1_SCRIPT}"
  echo "Base model: ${BASE_MODEL}"
  echo "Data:       ${STAGE1_DATA_ROOT} :: ${STAGE1_DATASET}"
  echo "Output:     ${STAGE1_OUT}"
  echo " ------------"

  cd "${ROOT}/BadVLA/vla-scripts"
  torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_STAGE1}" \
    "${STAGE1_SCRIPT}" \
    --vla_path "${BASE_MODEL}" \
    --data_root_dir "${STAGE1_DATA_ROOT}" \
    --dataset_name "${STAGE1_DATASET}" \
    --run_root_dir "${STAGE1_OUT}" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 2 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 1000 \
    --max_steps 5000 \
    --save_freq 1000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 4 \
    --run_id_note "${RUN_ID_NOTE}"
}

# Link the produced "...--<step>_chkpt" dir to a stable, task-tagged alias.
resolve_ckpt() {
  local out_dir="$1" pattern="$2" alias_path="$3"
  if [[ -d "${alias_path}" || -L "${alias_path}" ]]; then return; fi
  local found
  found="$(find "${out_dir}" -maxdepth 1 -type d -name "${pattern}" | sort | tail -1)"
  if [[ -n "${found}" ]]; then
    echo "Resolved $(basename "${found}") -> ${alias_path}"
    ln -sfn "$(basename "${found}")" "${alias_path}"
  fi
}

run_stage2() {
  preflight_dir "${CLEAN_DATA_ROOT}"
  resolve_ckpt "${STAGE1_OUT}" '*--5000_chkpt' "${STAGE1_CKPT}"
  if [[ ! -d "${STAGE1_CKPT}" ]]; then
    echo "ERROR: Stage I checkpoint not found: ${STAGE1_CKPT}"
    echo "Run STAGE=stage1 first or set STAGE1_CKPT=/path/to/stage1_chkpt"
    exit 1
  fi

  export CUDA_VISIBLE_DEVICES="${GPU_IDS_STAGE2}"

  echo " ------------"
  echo "Stage II: clean task enhancement (${TAG}; always clean RLDS)"
  echo "GPUs:          CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "Stage I ckpt:  ${STAGE1_CKPT}"
  echo "Data:          ${CLEAN_DATA_ROOT} :: ${CLEAN_DATASET}"
  echo "Output:        ${STAGE2_OUT}"
  echo " ------------"

  cd "${ROOT}/BadVLA/vla-scripts"
  torchrun --standalone --nnodes 1 --nproc-per-node "${NPROC_STAGE2}" \
    finetune_with_task.py \
    --vla_path "${STAGE1_CKPT}" \
    --data_root_dir "${CLEAN_DATA_ROOT}" \
    --dataset_name "${CLEAN_DATASET}" \
    --run_root_dir "${STAGE2_OUT}" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 10000 \
    --max_steps 30000 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 8 \
    --run_id_note "${RUN_ID_NOTE}"

  resolve_ckpt "${STAGE2_OUT}" '*--30000_chkpt' "${STAGE2_CKPT}"
}

case "${STAGE}" in
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  all)    run_stage1; run_stage2 ;;
  *)
    echo "Unknown STAGE=${STAGE}"
    echo "Use: stage1 | stage2 | all"
    exit 1
    ;;
esac

echo " ------------"
echo "Training done (TAG=${TAG} STAGE=${STAGE})"
if [[ "${STAGE}" == "stage2" || "${STAGE}" == "all" ]]; then
  echo "Final checkpoint: ${STAGE2_CKPT}"
  echo "Eval with:"
  echo "  SUITE=${SUITE} TRIGGER=${TRIGGER} ./run_libero_eval_local.sh"
fi
