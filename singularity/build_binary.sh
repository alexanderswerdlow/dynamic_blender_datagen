#!/bin/sh

alias singularity='/home/linuxbrew/.linuxbrew/bin/singularity'
alias sudo="sudo "

# Build a sandbox from the Singularity image
singularity build --sandbox sandbox blender.sif

# Create a new runscript for the sandbox
echo '#!/bin/sh' > sandbox/.singularity.d/runscript
echo 'exec /bin/blender "$@"' >> sandbox/.singularity.d/runscript
chmod +x sandbox/.singularity.d/runscript

# Build a new Singularity image from the sandbox
singularity build blender_binary.sig sandbox

# Run the new Singularity image with NVIDIA support
sudo singularity run --nv blender_binary.sig