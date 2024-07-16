#!/bin/bash
#PBS -l walltime=10:15:00
#PBS -l nodes=1:ppn=2:gpus=1,mem=100GB
#PBS -A PAS2099
#PBS -N jike338
#PBS -j oe
#PBS -m be
#SBATCH --output=slurmout_iclr/R-%x.%j.out
module load python
module load cuda


cd /fs/ess/scratch/PAS2099/jike/Math-LLaVA


# conda create -n llava python=3.10 -y
source activate llava


# pip3 install --upgrade pip  # enable PEP 660 support
# pip3 install -e .

# python3 run.py
pip3 install protobuf

python3 /fs/ess/scratch/PAS2099/jike/Maxth-LLaVA/llava/eval/run_llava.py \
--model-path /fs/ess/scratch/PAS2099/jike/Math-LLaVA/checkpoints/llava-v1.5-7b \
--image-file /fs/ess/scratch/PAS2099/jike/Math-LLaVA/images/llava_v1_5_radar.jpg \
--query /fs/ess/scratch/PAS2099/jike/Math-LLaVA/test_prompt.txt \
> poll_iclr/test.txt

