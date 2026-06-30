#!/bin/bash
#SBATCH --job-name=slr_path
#SBATCH --output=slr_path_%j.log
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#
# Regenerate the full SpiecEasi SLR lambda path for both AG smoking groups.
# Submit from the transfer_pack/ directory:
#   sbatch regenerate_slr_path.sbatch
#
# Adjust the conda env name if needed.

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate slr-env

cd "$SLURM_SUBMIT_DIR"
Rscript regenerate_slr_path.R
