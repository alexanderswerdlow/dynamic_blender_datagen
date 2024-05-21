#!/bin/sh

python export_annotation.py \
    --scene_dir ./data/demo_scene/robot.blend \
    --save_dir ./results/robot \
    --samples_per_pixel 64 \
    --export_tracking \
    --sampling_character_num 5000 \
    --sampling_scene_num 2000 \
    --start_frame 1 \
    --end_frame 11 \
    --batch_size 32 \
    --skip_n 100

