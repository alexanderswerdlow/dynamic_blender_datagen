#!/bin/zsh

source ~/.zshrc

cd /home/aswerdlo/repos/point_odyssey && conda activate dust3r;

while true; do
    python slurm.py --data_path='generated/train/v18' --num_frames=64 --num_to_process=999 --mode=generated --no-wait
    sleep 2h
done