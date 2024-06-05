#!/bin/sh

#SBATCH --job-name=refresh_mount
#SBATCH --output=outputs/refresh_mounts.out

echo "$(hostname) $(cat /proc/mounts | grep 'aswerdlo' | tr '\n' ' ')"
