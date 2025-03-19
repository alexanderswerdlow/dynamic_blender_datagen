import datetime

from simple_slurm import Slurm

# job_name=f"blender_{data_path.name}_{random.randint(0, 255):02x}",
# cpus_per_task=4,
# mem=mem_dict[max(num_frames, 32)],
# export="ALL",
# gres=["gpu:1"],
# output=f"outputs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}_{Slurm.JOB_ID}.out",
# time=timedelta(days=3, hours=0, minutes=0, seconds=0) if "kate" in partition else timedelta(days=0, hours=6, minutes=0, seconds=0),
# array=f"0-{num_chunks-1}%{num_workers}",
# partition=partition,
# requeue='',

slurm = Slurm(
    array=f"0-{1001-1}%{128}",
    requeue='',
    cpus_per_task=1,
    gres=["gpu:1"],
    job_name='slurm_array',
    output=f'outputs/demo/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
    time=datetime.timedelta(days=0, hours=0, minutes=1, seconds=0),
)
slurm.sbatch('/home/aswerdlo/repos/point_odyssey/scripts/refresh_mounts.sh')

