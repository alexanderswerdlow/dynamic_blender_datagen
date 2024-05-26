#!/bin/sh

set -e

spython recipe singularity/Dockerfile singularity/blender.def

# Build the Singularity image
singularity build --fakeroot --force singularity/blender.sif singularity/blender.def

# https://docs.sylabs.io/guides/3.5/user-guide/persistent_overlays.html
# dd if=/dev/zero of=singularity/overlay.img bs=1M count=200 && mkfs.ext3 singularity/overlay.img
# singularity sif add --datatype 4 --partfs 2 --parttype 4 --partarch 2 --groupid 1 singularity/blender.sif singularity/overlay.img
# singularity shell --writable singularity/blender.sif

# Create a new runscript that calls blender instead of bash.
singularity build --sandbox --force singularity/sandbox singularity/blender.sif
echo '#!/bin/sh' > singularity/sandbox/.singularity.d/runscript
echo 'exec /bin/blender "$@"' >> singularity/sandbox/.singularity.d/runscript
chmod +x singularity/sandbox/.singularity.d/runscript
singularity build --fakeroot --force singularity/blender_binary.sig singularity/sandbox
rm -rf singularity/sandbox 2>/dev/null


singularity run --bind $(pwd)/singularity/config:/.config --nv singularity/blender.sif