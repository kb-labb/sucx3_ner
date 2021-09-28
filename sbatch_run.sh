#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=ner_kram
#SBATCH --mem=200G
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=2
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/ner_kram.log

module purge
# deactivate
module load PyTorch/1.7.1-fosscuda-2020b
# 
source /ceph/hpc/home/eurobink/group_space/robin/envs/hugface/bin/activate

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
 
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
 
PROJECT=/ceph/hpc/home/eurobink/group_space/robin/workspace/ner_kram
LOGGING=$PROJECT/logs

srun -l \
     --output=$LOGGING/%x_$DATETIME.log "./run.sh"
set +x

exit 0