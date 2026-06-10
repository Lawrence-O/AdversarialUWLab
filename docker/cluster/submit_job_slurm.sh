#!/usr/bin/env bash

# In the case you need to load specific modules on the cluster, add them here.
# Klone (Hyak) currently doesn't need any module loads for apptainer.

# create job script with compute demands
### MODIFY HERE FOR YOUR JOB ###
cat <<EOT > job.sh
#!/bin/bash

#SBATCH --account=socialrl
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --job-name="uwlab-$(date +"%Y%m%d-%H%M%S")"
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lawony@uw.edu

# Source wandb credentials (WANDB_API_KEY) if available so the container can
# log without re-authing. File is owner-rw only (chmod 600 ~/.wandb-key).
if [ -f "\$HOME/.wandb-key" ]; then
    source "\$HOME/.wandb-key"
fi

# Pin wandb entity for this lab. The rsl_rl WandbSummaryWriter reads entity
# from WANDB_USERNAME (not the standard WANDB_ENTITY), so we set both.
# Project is set per-run via the --log_project_name flag on train.py
# (--log_project_name overrides agent_cfg.wandb_project).
# Run name is derived by rsl_rl as "\${experiment_name}_\${timestamp}" so set
# agent.experiment_name=<descriptive> on the training command line.
export WANDB_USERNAME="social-rl"
export WANDB_ENTITY="social-rl"

# Pass the container profile first to run_singularity.sh, then all arguments intended for the executed script
bash "$1/docker/cluster/run_singularity.sh" "$1" "$2" "${@:3}"
EOT

sbatch < job.sh
rm job.sh
