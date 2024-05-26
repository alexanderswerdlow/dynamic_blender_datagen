# https://docs.sylabs.io/guides/3.5/user-guide/persistent_overlays.html
# dd if=/dev/zero of=singularity/overlay.img bs=1M count=200 && mkfs.ext3 singularity/overlay.img
# singularity sif add --datatype 4 --partfs 2 --parttype 4 --partarch 2 --groupid 1 singularity/blender.sif singularity/overlay.img
# singularity shell --writable singularity/blender.sif