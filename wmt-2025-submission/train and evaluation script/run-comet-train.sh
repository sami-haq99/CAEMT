#!/usr/bin/env bash

#SBATCH --gres=gpu:a100:2
# SBATCH --nodelist=g130
#SBATCH -p long
#SBATCH -J regression
#SBATCH -t 9-23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster
source env-mtqe/bin/activate

# Move to COMET root directory if needed
cd /home/shaq/mtqe/COMET
#pip install -e .

# Run regression model using comet CLI
comet-train --cfg configs/models/reg_model_sqm_ja.yaml --load_from_checkpoint /home/shaq/mtqe/COMET/lightning_logs/version_3/checkpoints/epoch=6-step=2044-val_kendall=0.444.ckpt
