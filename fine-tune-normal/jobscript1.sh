#!/bin/bash

#PBS -P wd04
#PBS -q gpuvolta
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=256GB
#PBS -l jobfs=1GB
#PBS -l walltime=12:00:00
#PBS -l storage=scratch/wd04
#PBS -l wd
#PBS -M s4025371@student.rmit.edu.au
#PBS -m be
#PBS -o /scratch/wd04/bk2508/repositories/iot-llm-pcp-ids/fine-tune-normal/logs/
#PBS -e /scratch/wd04/bk2508/repositories/iot-llm-pcp-ids/fine-tune-normal/logs/

module load python3/3.10.4
source /scratch/wd04/bk2508/venvs/llm-env/bin/activate
export WANDB_DIR=/scratch/wd04/bk2508/repositories/iot-llm-pcp-ids/fine-tune-normal
export WANDB_MODE=offline
python3 main.py $PBS_JOBID "google/gemma-3-1b-it" "x-iiotid" > /scratch/wd04/bk2508/repositories/iot-llm-pcp-ids/fine-tune-normal/logs/$PBS_JOBID.log
deactivate
