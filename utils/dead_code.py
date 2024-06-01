# https://docs.sylabs.io/guides/3.5/user-guide/persistent_overlays.html
# dd if=/dev/zero of=singularity/overlay.img bs=1M count=200 && mkfs.ext3 singularity/overlay.img
# singularity sif add --datatype 4 --partfs 2 --parttype 4 --partarch 2 --groupid 1 singularity/blender.sif singularity/overlay.img
# singularity shell --writable singularity/blender.sif
            if False:
                from image_utils import Im

                imgs = []
                for object_idx in valid_indices:
                    asset_mask = get_mask(seg_frame_1, object_idx)
                    viz_img = Im(asset_mask).write_text(f"{assets[object_idx]}").np.copy()  # [H, W, C] uint8, 0-255

                    viz_bbox = get_crop_bbox(asset_mask, scale=1.5)
                    l, t, r, b = viz_bbox
                    viz_img_with_bbox = cv2.rectangle(viz_img, (l, t), (r, b), color=(255, 0, 0), thickness=2)
                    imgs.append(viz_img_with_bbox)

                Im(np.stack(imgs)).save()
                def viz_pyviz3d(output):
    import pyviz3d.visualizer as viz

    vis = viz.Visualizer()
    for idx in range(output["pred1"]["pts3d"].shape[0])[:4]:
        vis.add_points(
            f"pred1_{idx}",
            output["pred1"]["pts3d"][idx].reshape(-1, 3).cpu().numpy(),
            colors=255 * np.array([[1.0, 0.0, 0.0]] * output["pred1"]["pts3d"][idx].reshape(-1, 3).shape[0]).astype(np.uint8),
            point_size=1,
        )
        vis.add_points(
            f"pred2_{idx}",
            output["pred2"]["pts3d_in_other_view"][idx].reshape(-1, 3).cpu().numpy(),
            colors=255 * np.array([[0.0, 1.0, 0.0]] * output["pred2"]["pts3d_in_other_view"][idx].reshape(-1, 3).shape[0]).astype(np.uint8),
            point_size=1,
        )

    vis.save("output/viz")

# breakpoint()
        # rr.set_time_sequence("frame", idx)
        # pred1 = output["pred1"]["pts3d"][idx].reshape(-1, 3)
        # pred2 = output["pred2"]["pts3d_in_other_view"][idx].reshape(-1, 3)

        # img1 = ((output["view1"]["img"][idx] + 1) / 2).reshape(-1, 3)
        # img2 = ((output["view2"]["img"][idx] + 1) / 2).reshape(-1, 3)

        # rr.log(f"world/pred1", rr.Points3D(pred1, colors=img1))
        # rr.log(f"world/pred2", rr.Points3D(pred2, colors=img2))
                    # dynamic_depth_map_path = self.data_path / scene / "dynamic_depths" / f"depth_{idx_to_str_5(im_idx)}_{idx_to_str_5(im1_idx)}.npz"
            # loaded_data = np.load(dynamic_depth_map_path)
            # pts3d = loaded_data["xyz"].astype(np.float32)
            # valid_mask = loaded_data["dynamic_mask"].astype(bool)




    parser.add_argument("--character_root", type=str, metavar="PATH", default="./data/robots/")
    parser.add_argument("--camera_root", type=str, metavar="PATH", default="./data/camera_trajectory/MannequinChallenge")
    parser.add_argument("--motion_root", type=str, metavar="PATH", default="./data/motions/")
    parser.add_argument("--partnet_root", type=str, metavar="PATH", default="./data/partnet/")
    parser.add_argument("--gso_root", type=str, metavar="PATH", default="./data/GSO/")
    parser.add_argument("--background_hdr_path", type=str, default=None)
    parser.add_argument("--scene_root", type=str, default="./data/blender_assets/hdri.blend")

                parser.add_argument("--output_name", type=str, metavar="PATH", help="img save name", default="test")
    parser.add_argument("--force_step", type=int, default=3)
    parser.add_argument("--force_interval", type=int, default=120)
    parser.add_argument("--force_num", type=int, default=3)
    parser.add_argument("--add_force", action="store_true", default=False)
    parser.add_argument("--num_assets", type=int, default=5)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--indoor", action="store_true", default=False)
    parser.add_argument("--views", type=int, default=1)
    parser.add_argument("--render_engine", type=str, default="CYCLES", choices=["BLENDER_EEVEE", "CYCLES"])
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--end_frame", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--samples_per_pixel", type=int, default=128)
    parser.add_argument("--randomize", action="store_true", default=False)
    parser.add_argument("--add_fog", action="store_true", default=False)
    parser.add_argument("--fog_path", type=str, default=None)
    parser.add_argument("--add_smoke", action="store_true", default=False)
    parser.add_argument("--material_path", type=str, default=None)
    parser.add_argument("--scene_scale", type=float, default=1.0)
    parser.add_argument("--force_scale", type=float, default=1.0)
    parser.add_argument("--use_animal", action="store_true", default=False)
    parser.add_argument("--animal_path", type=str, default=None)
    parser.add_argument("--animal_name", type=str, default=None)


    rendering_script = (
        f"{blender_path} --background --python render_human.py -- "
        f"--output_dir {args.output_dir} --character_root {args.character_root} "
        f"--partnet_root {args.partnet_root} --gso_root {args.gso_root} "
        f"--background_hdr_path {args.background_hdr_path} --scene_root {args.scene_root} "
        f"--camera_root {args.camera_root} --num_assets {args.num_assets} "
        f"--render_engine {args.render_engine} --force_num {args.force_num} "
        f"--force_step {args.force_step} --force_interval {args.force_interval} "
        f"--end_frame {args.end_frame} "
        f"--fps {args.fps} "
        f"--samples_per_pixel {args.samples_per_pixel} "
        f"--scene_scale {args.scene_scale} --force_scale {args.force_scale} "
        f"--animal_path {args.animal_path} "
    )
    if args.use_gpu:
        rendering_script += ' --use_gpu'
    if args.add_fog:
        rendering_script += ' --add_fog'
        rendering_script += f' --fog_path {args.fog_path}'
    if args.randomize:
        rendering_script += ' --randomize'
    if args.material_path is not None:
        rendering_script += f' --material_path {args.material_path}'
    if args.add_smoke:
        rendering_script += ' --add_smoke'
    if args.add_force:
        rendering_script += ' --add_force'
    if args.use_animal:
        rendering_script += ' --use_animal'
    if args.indoor:
        rendering_script += ' --indoor'


            else:
                for idx, target_frame_idx in enumerate(sorted(points_to_track.keys())):
                    if cache_meshes:
                        mesh = loaded_meshes[target_frame_idx][asset_name]
                        triangles = mesh.triangles
                    else:
                        obj_path = os.path.join(
                            obj_root, f"{asset_name}_{str(reference_frame_idx_to_blender_frame_idx[reference_frame_idx]).zfill(4)}.obj"
                        )
                        triangles = read_obj_file_triangles(str(obj_path))

                    points_on_mesh = trimesh.triangles.barycentric_to_points(triangles[index_faces], intersection_points_barycentric_coordinates)
                    if viz and abs(reference_frame_idx - target_frame_idx) < 4:
                        rr.log(f"world/{asset_name}/{idx}_tracked_3d_frame_pos", rr.Points3D(points_on_mesh))
                        if asset_name in tracked_points[reference_frame_idx] and len(tracked_points[reference_frame_idx][asset_name]) != 0:
                            rr.log(
                                f"world/{asset_name}/{idx}_tracked_3d_delta_frame_diff",
                                rr.Arrows3D(
                                    origins=tracked_points[reference_frame_idx][asset_name][-1],
                                    vectors=points_on_mesh - tracked_points[reference_frame_idx][asset_name][-1],
                                ),
                            )

                    if asset_name in tracked_points[reference_frame_idx]:
                        tracked_points[reference_frame_idx][asset_name].append(points_on_mesh)
                    else:
                        tracked_points[reference_frame_idx][asset_name] = [points_on_mesh]