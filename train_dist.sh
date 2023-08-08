#!/bin/bash
#SBATCH --job-name=edpose
#SBATCH -p cvr
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:hgx:4
#SBATCH --mem 100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=1770042443@qq.com
export PATH=~/anaconda3/bin:$PATH
source ~/.bashrc
source activate centergroup

cd /comp_robot/yangjie/ED-Pose_new
export EDPOSE_COCO_PATH="/comp_robot/cv_public_dataset/COCO2017"
export EDPOSE_HumanArt_PATH="/comp_robot/juxuan/data/datasets/"
python -m torch.distributed.launch --nproc_per_node=4 --master_port 2515 main.py \
 --output_dir "logs/4444" \
 -c config/edpose.cfg.py \
 --dataset_file humanart \
 --options num_box_decoder_layers=2 batch_size=4 epochs=80 lr_drop=80 use_ema=TRUE \
 --pretrain_model_path /comp_robot/yangjie/edpose_humanart_r50/logs/humanart_coco_r50/checkpoint_best_ema.pth --eval
