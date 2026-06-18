#!/bin/bash
#SBATCH -J badvla   # Name of the job
#SBATCH --account=swan_research_t2i  # Account allocation

#SBATCH --partition=a100_normal_q   # Partition of the cluster
## QoS
## This is not required because it is the default.
#SBATCH --qos=tc_a100_normal_short
## Wall time 23h
#SBATCH --time=23:00:00


#SBATCH --nodes=1   # Number of compute nodes
#SBATCH --ntasks-per-node=1   # Number of processes
#SBATCH --cpus-per-task=16   # Number of CPU cores per process
#SBATCH --gres=gpu:3   # Request one GPU (only valid on GPU partitions)


module reset
module load Miniconda3/24.7.1-0
module load CUDA/12.4.0            
source activate /projects/swan_research_t2i/neelesh/VLA_def/venvs/openvla-oft

export PYTHONPATH=/projects/swan_research_t2i/neelesh/VLA_def/code/LIBERO:$PYTHONPATH
export MUJOCO_GL=egl


## Get the core number for job and other job details.
## -d flag gets you the particular cores running on.
echo " ------------"
echo "Set of cores job running on: "
echo " "
scontrol show job -d  $SLURM_JOB_ID
echo " "
echo " "

## Monitor the GPU.
## The 3 means output data every 3 seconds; you will have to tweek
## based on your execution duration.
echo " "
echo " "
echo "Start file and monitoring of GPU."
nvidia-smi --query-gpu=timestamp,name,pci.bus_id,driver_version,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 3 > gpu.perform.$SLURM_JOBID.log &
echo " "
echo " "

echo " ------------"
echo "Running executable"

# ------------------------
# Code to execute:

# srun torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune_with_trigger_injection_pixel.py \
#   --vla_path /projects/swan_research_t2i/neelesh/VLA_def/models/openvla-7b-oft-finetuned-libero-goal \
#   --data_root_dir /projects/swan_research_t2i/neelesh/VLA_def/data/modified_libero_rlds_real \
#   --dataset_name libero_goal_no_noops \
#   --run_root_dir ./goal/trigger_fir \
#   --use_l1_regression True \
#   --use_diffusion False \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --batch_size 2 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 1000 \
#   --max_steps 5000 \
#   --save_freq 1000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --lora_rank 4 \
#   --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state



# srun torchrun --standalone --nnodes 1 --nproc-per-node 3 finetune_with_task.py \
#   --vla_path /projects/swan_research_t2i/neelesh/VLA_def/code/BadVLA/vla-scripts/goal/trigger_fir/stage1_5000_chkpt \
#   --data_root_dir /projects/swan_research_t2i/neelesh/VLA_def/data/modified_libero_rlds_real \
#   --dataset_name libero_goal_no_noops \
#   --run_root_dir ./goal/trigger_sec \
#   --use_l1_regression True \
#   --use_diffusion False \
#   --use_film False \
#   --num_images_in_input 2 \
#   --use_proprio True \
#   --batch_size 8 \
#   --learning_rate 5e-4 \
#   --num_steps_before_decay 10000 \
#   --max_steps 30000 \
#   --save_freq 10000 \
#   --save_latest_checkpoint_only False \
#   --image_aug True \
#   --lora_rank 8 \
#   --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state

cd ../experiments/robot/libero/

python run_libero_eval.py \
  --pretrained_checkpoint /projects/swan_research_t2i/neelesh/VLA_def/code/BadVLA/vla-scripts/goal/trigger_sec/stage2_30000_chkpt \
  --task_suite_name libero_goal

python run_libero_eval.py \
  --pretrained_checkpoint /projects/swan_research_t2i/neelesh/VLA_def/code/BadVLA/vla-scripts/goal/trigger_sec/stage2_30000_chkpt \
  --task_suite_name libero_goal \
  --trigger True


echo " ------------"
echo "Executable done"