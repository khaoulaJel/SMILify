#!/usr/bin/env bash
#SBATCH --job-name=issue95_dispatcher
#SBATCH --partition=c23ms
#SBATCH --account=default
#SBATCH --output=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun/dispatch_resume.log
#SBATCH --error=/rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun/dispatch_resume.log
#SBATCH --time=48:00:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1

# Orchestration-only job: submits/waits on the remaining issue95_fullrun chunks (363-602)
# sequentially, exactly like the original dispatch_chunks.sh (which was running in a real
# terminal session, not survivable from an automated tool session -- resubmitted as an actual
# Slurm job here so it persists independently).
bash /rwthfs/rz/cluster/home/nao48500/SMILify/diagnostics/issue95_fullrun/dispatch_chunks_resume.sh
