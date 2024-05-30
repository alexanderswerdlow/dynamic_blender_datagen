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