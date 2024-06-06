#!/bin/zsh

source ~/.zshrc

cd /home/aswerdlo/repos/point_odyssey && conda activate gen;

while true; do
    sb python scripts/check_tmp.py --gpu_count=0 --cpu_count=1 --mem=1 --partition='all' --quick
    sleep 2h
done