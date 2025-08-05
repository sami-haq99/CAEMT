#!/usr/bin/env bash
#SBATCH --gres=gpu:a100:2
# SBATCH --nodelist=g129
#SBATCH -p compute
#SBATCH -J predict_s
#SBATCH -t 1-23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster
#source /home/cosuji/anaconda3/etc/profile.d/conda.sh
#conda activate comet38
source env-mtqe/bin/activate

#coment model path
#model_path="/home/shaq/mtqe/COMET/comet-22-da-checkpoint.ckpt"
# Run regression model using comet CLI
model_path="/home/shaq/lightning_logs/sqm_zh/checkpoints/epoch=4-step=46605-val_kendall=0.520.ckpt"
model_name="sqm_zh"
python eval_seg_sys_all.py en_zh $model_name $model_path
