#!/bin/sh

set -e

spython recipe singularity/Dockerfile singularity/blender.def

# Build the Singularity image
singularity build --fakeroot --force singularity/blender.sif singularity/blender.def


singularity build --sandbox --force singularity/sandbox singularity/blender.sif

# Create a new runscript that calls blender instead of bash.
echo '#!/bin/sh' > singularity/sandbox/.singularity.d/runscript
echo 'exec /bin/blender "$@"' >> singularity/sandbox/.singularity.d/runscript
chmod +x singularity/sandbox/.singularity.d/runscript

# See: https://github.com/KhronosGroup/glTF-Blender-IO/issues/1844
sed -i 's/np\.empty(num_polys, dtype=np\.bool)/np\.empty(num_polys, dtype=np\.bool_)/g' singularity/sandbox/usr/bin/3.3/scripts/addons/io_scene_gltf2/blender/imp/gltf2_blender_mesh.py

singularity build --fakeroot --force singularity/blender.sif singularity/sandbox
rm -rf singularity/sandbox 2>/dev/null

singularity shell --bind $(pwd)/singularity/config:/.config --nv singularity/blender.sif