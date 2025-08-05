#!/usr/bin/env bash

#SBATCH --gres=gpu:rtxa6000:2
# SBATCH --nodelist=g126
#SBATCH -p compute
#SBATCH -J correlation
#SBATCH -t 23:59:59
#SBATCH -o comet-%j.out
#SBATCH --mail-type=ALL --mail-user=sami.haq@adaptcentre.ie


# For ADAPT Cluster
#source /home/cosuji/anaconda3/etc/profile.d/conda.sh
#conda activate comet38
source env-mtqe/bin/activate
#model_path = "/home/shaq/mtqe/COMET/comet-22-da.ckpt"

model_path="/home/shaq/roberta/checkpoints/epoch=4-step=498880-val_kendall=0.635.ckpt"
model_name="roberta"
language_pair="all"
data_dir="/home/shaq/mtqe/COMET/data/"
method="esa"
python corr_all.py $language_pair $model_name $model_path $data_dir $method

