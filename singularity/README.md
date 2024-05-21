alias singularity='/home/linuxbrew/.linuxbrew/bin/singularity'
alias sudo="sudo "

spython recipe Dockerfile_v2 image.def

sudo singularity build --force image.sif image.def

singularity run --nv image.sif

singularity build --sandbox sandbox image.sif

echo '#!/bin/sh' > sandbox/.singularity.d/runscript
echo 'exec /bin/blender "$@"' >> sandbox/.singularity.d/runscript
chmod +x sandbox/.singularity.d/runscript

singularity build blender_binary.sig sandbox

singularity run --nv blender_binary.sig




```
blender -b --python-console


```


