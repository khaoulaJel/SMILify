#!/usr/bin/env bash
#SBATCH --account=ias-7
#SBATCH --partition=dc-gpu
#SBATCH --job-name=net_check
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:02:00
#SBATCH --output=diagnostics/khaoula_v2/tmp/net_check_%j.log
#SBATCH --error=diagnostics/khaoula_v2/tmp/net_check_%j.log
cd "$SLURM_SUBMIT_DIR"
hostname
curl -sI https://huggingface.co --max-time 8 && echo "HF_OK" || echo "HF_BLOCKED"
