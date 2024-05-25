#!/bin/sh

set -e

spython recipe singularity/Dockerfile singularity/blender.def

# Build the Singularity image
singularity build --fakeroot --force singularity/blender.sif singularity/blender.def

# Create a new runscript that calls blender instead of bash.
singularity build --sandbox --force singularity/sandbox singularity/blender.sif
echo '#!/bin/sh' > singularity/sandbox/.singularity.d/runscript
echo 'exec /bin/blender "$@"' >> singularity/sandbox/.singularity.d/runscript
chmod +x singularity/sandbox/.singularity.d/runscript
singularity build --fakeroot --force singularity/blender_binary.sig singularity/sandbox
rm -rf sandbox 2>/dev/null

singularity run --nv singularity/blender.sif