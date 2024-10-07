# Large Scale Dynamic Video Data Generation w/Blender

This repo is mainly based off of [PointOdyssey](https://github.com/y-zheng18/point_odyssey), extended to handle more data sources, incorporate aspects of Kubrics, and run in a distributed manner w/Slurm and docker/singularity containers.

Besides ease-of-use, this repo extracts _pointwise_ trajectories for each point in each scene. That is, given a pixel in any frame, we record the position of that same point in all past and future frames. This is generally computed per-mesh, and for efficiency reasons, we generally only record the pose of the entire mesh and then re-trace the point trajectory after rendering.

This data is useful for models that need to handle dynamic—for reconstruction, pose estimation, point tracking, etc.

## Data Visualizations

*WIP. TODO: Link to re-run visualizations*

Two-frame visualization: <img width="950" alt="Screenshot 2024-10-07 at 1 37 36 PM" src="https://github.com/user-attachments/assets/dd1ab6b3-e957-4267-a48b-7531369f918c">

Visualization of GT + Model prediction: <img width="1497" alt="Screenshot 2024-10-07 at 1 39 24 PM" src="https://github.com/user-attachments/assets/b98219f5-e76d-4be2-ab55-2e7988348b74">
