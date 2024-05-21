#!/bin/sh

# Define aliases
alias singularity='/home/linuxbrew/.linuxbrew/bin/singularity'
alias sudo="sudo "

# Convert Dockerfile to Singularity definition file
spython recipe Dockerfile_v2 blender.def

# Build the Singularity image
sudo singularity build --force blender.sif blender.def

# Run the Singularity image with NVIDIA support
sudo singularity run --nv blender.sif
