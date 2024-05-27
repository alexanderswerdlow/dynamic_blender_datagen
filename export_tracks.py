from collections import defaultdict
import numpy as np
import cv2
import os
import glob
import matplotlib
from utils.file_io import read_tiff, write_png
from tqdm import tqdm
import trimesh
import shutil
import json
from pathlib import Path
import utils.plotting as plotting
from utils.decoupled_utils import breakpoint_on_error
from einops import repeat, pack, unpack, rearrange


def read_obj_file(obj_path: str):
    """
    Load .obj file, return vertices, faces.
    return: vertices: N_v X 3, faces: N_f X 3
    """
    obj_f = open(obj_path, "r")
    lines = obj_f.readlines()
    vertices = []
    faces = []
    for ori_line in lines:
        line = ori_line.split()
        if line[0] == "v":
            vertices.append([float(line[1]), float(line[2]), float(line[3])])  # x, y, z
        elif line[0] == "f":  # Need to consider / case, // case, etc.
            faces.append(
                [int(line[3].split("/")[0]), int(line[2].split("/")[0]), int(line[1].split("/")[0])]
            )  # Notice! Need to reverse back when using the face since here it would be clock-wise!
            # Convert face order from clockwise to counter-clockwise direction.
    obj_f.close()

    return np.asarray(vertices), np.asarray(faces)


def unproject(depth, K, RT):
    H, W = depth.shape
    y, x = np.indices((H, W))
    ones = np.ones((H, W))
    pixel_positions = np.stack((x, y, ones), axis=-1)
    K_inv = np.linalg.inv(K)
    camera_coords = pixel_positions @ K_inv.T
    camera_coords *= depth[..., None]
    camera_coords_hom = np.concatenate([camera_coords, ones[..., None]], axis=-1)

    R = RT[:3, :3]
    T = RT[:3, 3]
    RT_inv = np.eye(4)
    RT_inv[:3, :3] = R.T  # Transpose of rotation matrix R
    RT_inv[:3, 3] = -R.T @ T  # Correct translation component
    points = camera_coords_hom @ RT_inv.T

    return points[..., :3]


def reprojection(points: np.ndarray, K: np.ndarray, RT: np.ndarray):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z


def check_visibility(uv, z, depth, h, w):
    visibility = np.zeros((uv.shape[0], 1))
    for j in range(len(uv)):
        u, v = uv[j]
        if u < 0 or u >= w or v < 0 or v >= h:
            visibility[j] = 0
            # print('out of range')
            continue
        else:
            v_low = np.floor(uv[j, 1]).astype(np.int32)
            v_high = np.min([np.ceil(uv[j, 1]).astype(np.int32), h - 1])
            u_low = np.floor(uv[j, 0]).astype(np.int32)
            u_high = np.min([np.ceil(uv[j, 0]).astype(np.int32), w - 1])
            # find nearest depth
            d_max = np.max(depth[v_low : v_high + 1, u_low : u_high + 1])
            d_min = np.min(depth[v_low : v_high + 1, u_low : u_high + 1])
            d_median = np.median(depth[v_low : v_high + 1, u_low : u_high + 1])
            if z[j] < 0 or z[j] > 1000:
                visibility[j] = 2
                # print('invalid depth')
                continue
            if d_max >= 0.97 * z[j] and d_min <= 1.05 * z[j] and z[j] > 0.95 * d_median and z[j] < 1.05 * d_median:
                visibility[j] = 1
    return visibility


def farthest_point_sampling(p, K):
    """
    greedy farthest point sampling
    p: point cloud
    K: number of points to sample
    """

    farthest_point = np.zeros((K, 3))
    idx = []
    max_idx = np.random.randint(0, p.shape[0] - 1)
    farthest_point[0] = p[max_idx]
    idx.append(max_idx)
    print("farthest point sampling")
    for i in range(1, K):
        pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
        distance = np.min(pairwise_distance, axis=1, keepdims=True)
        max_idx = np.argmax(distance)
        farthest_point[i] = p[max_idx]
        idx.append(max_idx)
    print("farthest point sampling done")
    return farthest_point, idx


def compute_camera_rays(w, h, K, RT):
    j, i = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")

    # Flatten the grid
    j_flat = j.flatten()
    i_flat = i.flatten()

    # Create homogeneous pixel coordinates, xy
    pixel_coords = np.vstack((j_flat, i_flat, np.ones_like(j_flat)))

    # Unproject pixel coordinates to camera space
    K_inv = np.linalg.inv(K)
    rays_camera = K_inv @ pixel_coords

    # Transform rays from camera space to world space
    RT_inv = np.linalg.inv(RT)
    rays_camera_homogeneous = np.vstack((rays_camera, np.ones((1, rays_camera.shape[1]))))
    rays_world_homogeneous = RT_inv @ rays_camera_homogeneous

    # Extract the origins and directions of the rays
    origins = RT_inv[:3, 3]
    directions = rays_world_homogeneous[:3, :] - origins[:, np.newaxis]

    # Normalize the directions
    directions = directions / np.linalg.norm(directions, axis=0)
    origins = np.tile(origins, (directions.shape[1], 1))

    return origins, directions.T


def idx_to_str_4(idx):
    return str(idx).zfill(4)


def idx_to_str_5(idx):
    return str(idx).zfill(5)


def tracking(cp_root: Path, data_root: Path, viz=False):
    from pathlib import Path

    img_root = cp_root / "images"
    exr_root = cp_root / "exr_img"
    cam_root = cp_root / "cam"
    obj_root = cp_root / "obj" if (cp_root / "obj").exists() else data_root / "obj"
    print("copying exr data ...")

    save_rgbs_root = data_root / "rgbs"
    save_depths_root = data_root / "depths"
    save_masks_root = data_root / "masks"
    save_normals_root = data_root / "normals"
    save_dynamic_depths_root = data_root / "dynamic_depths"

    save_rgbs_root.mkdir(parents=True, exist_ok=True)
    save_depths_root.mkdir(parents=True, exist_ok=True)
    save_masks_root.mkdir(parents=True, exist_ok=True)
    save_normals_root.mkdir(parents=True, exist_ok=True)
    save_dynamic_depths_root.mkdir(parents=True, exist_ok=True)

    frames = sorted(img_root.glob("*.png"))
    num_frames = len(frames)
    print(f"num_frames: {num_frames}")

    # save_vis_dir = os.path.join(data_root, 'tracking')

    K_data = []
    RT_data = []

    R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R3 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    scene_info = json.load(open(os.path.join(cp_root, "scene_info.json"), "r"))
    assets = scene_info["assets"]
    pallette = plotting.hls_palette(len(assets) + 2)

    if viz:
        import rerun as rr
        import rerun.blueprint as rrb

        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.Grid(
                [rrb.Spatial2DView(name="rgb", origin=f"world/camera/image/rgb"), rrb.Spatial2DView(name="depth", origin=f"world/camera/image/depth")]
            ),
        )

        rr.init("rerun_example_my_data", default_blueprint=blueprint)
        rr.serve(web_port=0, ws_port=0)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points_to_track = defaultdict(dict)
    loaded_meshes = defaultdict(dict)
    depth_data = dict()

    max_depth_value = 1000
    min_depth_value = 0

    for reference_frame_idx, reference_frame in tqdm(enumerate(range(1, num_frames - 1))):
        blender_reference_frame = reference_frame + 1
        K = np.loadtxt(cam_root / f"K_{idx_to_str_4(blender_reference_frame)}.txt")
        RT = np.loadtxt(cam_root / f"RT_{idx_to_str_4(blender_reference_frame)}.txt")
        RT = R3 @ R2 @ RT @ R1
        depth = read_tiff(exr_root / f"depth_{idx_to_str_5(blender_reference_frame)}.tiff")
        mask = cv2.imread(str(exr_root / f"segmentation_{idx_to_str_5(blender_reference_frame)}.png"))
        img = cv2.imread(str(exr_root / f"rgb_{idx_to_str_5(blender_reference_frame)}.png"))
        h, w, _ = img.shape

        # convert img to jpg
        save_img_path = save_rgbs_root / f"rgb_{idx_to_str_5(reference_frame_idx)}.jpg"
        cv2.imwrite(str(save_img_path), img)

        # convert depth to 16 bit png
        save_depth_path = save_depths_root / f"depth_{idx_to_str_5(reference_frame_idx)}.png"
        data = depth.copy()
        data[data > max_depth_value] = max_depth_value
        data[data < min_depth_value] = min_depth_value
        depth_data[reference_frame_idx] = data.copy()
        data = (data - min_depth_value) * 65535 / (max_depth_value - min_depth_value)
        data = data.astype(np.uint16)
        write_png(data, save_depth_path)

        # cp normals and masks
        save_normal_path = save_normals_root / f"normal_{idx_to_str_5(reference_frame_idx)}.jpg"
        save_normal = cv2.imread(str(exr_root / f"normal_{idx_to_str_5(blender_reference_frame)}.png"))
        cv2.imwrite(str(save_normal_path), save_normal)

        save_mask_path = save_masks_root / f"mask_{idx_to_str_5(reference_frame_idx)}.png"
        if (exr_root / f"segmentation_{idx_to_str_5(blender_reference_frame)}.png").exists():
            shutil.copy(exr_root / f"segmentation_{idx_to_str_5(blender_reference_frame)}.png", save_mask_path)

        K_data.append(K.astype(np.float16))
        RT_data.append(RT.astype(np.float16))

        seg_mask = cv2.imread(str(save_mask_path))
        static_mask = (seg_mask == 0).all(-1)

        if viz:
            h, w, _ = data.shape

            cam_intrinsic = K
            cam_extrinsic = RT
            depth = depth_data[reference_frame_idx].copy()

            rr.set_time_sequence("frame", reference_frame_idx)
            rr.log("world/camera/depth", rr.DepthImage(depth))
            rr.log("world/camera/rgb", rr.Image(img))
            rr.log(
                "world/camera",
                rr.Transform3D(rr.TranslationAndMat3x3(translation=cam_extrinsic[:3, 3], mat3x3=cam_extrinsic[:3, :3], from_parent=True)),
            )
            rr.log(
                "world/camera",
                rr.Pinhole(
                    resolution=[w, h],
                    focal_length=[cam_intrinsic[0, 0], cam_intrinsic[1, 1]],
                    principal_point=[cam_intrinsic[0, 2], cam_intrinsic[1, 2]],
                ),
                static=True,
            )

        j_coords, i_coords = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        j_flat = j_coords.flatten()
        i_flat = i_coords.flatten()
        pixel_coords = np.vstack((j_flat, i_flat)).T  # Shape: [h*w, 2]
        pixel_coords = pixel_coords.reshape(h, w, 2)

        for idx in range(len(assets)):
            asset = assets[idx]
            asset_name = asset.replace(".", "_")
            if asset_name == "background":
                continue
            obj_path = os.path.join(obj_root, "{}_{}.obj".format(asset_name, str(blender_reference_frame).zfill(4)))
            mesh = trimesh.load(str(obj_path))
            asset_mask = np.logical_and(mask[:, :, 2] == pallette[idx + 1, 0], mask[:, :, 1] == pallette[idx + 1, 1])
            asset_mask = np.logical_and(asset_mask, mask[:, :, 0] == pallette[idx + 1, 2])

            loaded_meshes[reference_frame_idx][asset_name] = mesh

            origins, directions = compute_camera_rays(w, h, K, RT)
            origins = origins.reshape(h, w, 3)
            directions = directions.reshape(h, w, 3)

            origins = origins[asset_mask]
            directions = directions[asset_mask]
            asset_pixel_coords = pixel_coords[asset_mask]

            if viz:
                rr.log("world/camera_rays", rr.Arrows3D(origins=origins, vectors=directions * 5))

            # Perform ray-mesh intersection using trimesh
            ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)
            intersection_points, index_ray, index_faces = ray_mesh_intersector.intersects_location(
                ray_origins=origins, ray_directions=directions, multiple_hits=False
            )

            intersection_points_barycentric_coordinates = trimesh.triangles.points_to_barycentric(
                mesh.triangles[index_faces], points=intersection_points
            )

            # Create a boolean array indicating which points intersect with the mesh
            intersection_mask = np.zeros(origins.shape[0], dtype=bool)
            intersection_mask[index_ray] = True

            asset_pixel_coords = asset_pixel_coords[intersection_mask]

            points_to_track[reference_frame_idx][asset_name] = (
                intersection_points,
                intersection_points_barycentric_coordinates,
                index_faces,
                asset_pixel_coords,
            )

            if viz:
                intersection_dense = np.zeros_like(origins)
                intersection_dense[index_ray] = intersection_points

                uv_intersected, z_intersected = reprojection(intersection_dense, K, RT)
                # check distance for valid indices
                distance_diff = z_intersected[intersection_mask].squeeze(-1) - depth.squeeze(-1)[asset_mask][intersection_mask]

                init_ray = origins[intersection_mask]
                endpoints = (intersection_dense - origins)[intersection_mask]
                rr.log("world/camera_rays_mesh_intersection", rr.Arrows3D(origins=init_ray, vectors=endpoints))

                camera_ray_dirs = endpoints / np.linalg.norm(endpoints, axis=-1, keepdims=True)
                rr.log(
                    "world/mesh_intersection_depth_delta",
                    rr.Arrows3D(origins=init_ray + endpoints, vectors=camera_ray_dirs * distance_diff[..., None]),
                )

    tracked_points = defaultdict(dict)
    for reference_frame_idx in sorted(points_to_track.keys()):
        if viz:
            rr.set_time_sequence("frame", reference_frame_idx)
        for asset_name in points_to_track[reference_frame_idx].keys():
            intersection_points, intersection_points_barycentric_coordinates, index_faces, asset_pixel_coords = points_to_track[reference_frame_idx][
                asset_name
            ]
            for idx, target_frame_idx in enumerate(sorted(points_to_track.keys())):
                points_on_mesh = trimesh.triangles.barycentric_to_points(
                    loaded_meshes[target_frame_idx][asset_name].triangles[index_faces], intersection_points_barycentric_coordinates
                )
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

    K_data = np.stack(K_data, axis=0)
    RT_data = np.stack(RT_data, axis=0)

    max_num_pts = max([i[0].shape[0] for idx in points_to_track.keys() for i in points_to_track[idx].values()])
    window_size = len(points_to_track)
    num_objects = len(points_to_track[next(iter(points_to_track.keys()))].keys())
    pixel_aligned_tracks = np.full((len(points_to_track), window_size, num_objects, max_num_pts, 3), dtype=np.float16, fill_value=np.nan)

    points_outside_image = np.full((len(points_to_track), window_size, num_objects), dtype=np.int32, fill_value=0)
    for reference_frame_idx, reference_frame in enumerate(points_to_track.keys()):
        if viz:
            rr.set_time_sequence("frame", reference_frame_idx)
        cur_depth = repeat(depth_data[reference_frame], "h w () -> b h w ()", b=len(points_to_track))
        cur_xyz = repeat(
            unproject(
                depth_data[reference_frame].squeeze(-1),
                K_data[reference_frame_idx].astype(np.float32),
                RT_data[reference_frame_idx].astype(np.float32),
            ),
            "h w c -> b h w c",
            b=window_size,
        )

        if viz:
            rr.log(f"world/depth_unproj", rr.Points3D(cur_xyz[0].reshape(-1, 3)))
        for asset_idx, asset_name in enumerate(points_to_track[reference_frame].keys()):
            _, _, _, coords = points_to_track[reference_frame][asset_name]

            pts = np.stack(tracked_points[reference_frame][asset_name], axis=0)
            pixel_aligned_tracks[reference_frame_idx, :, asset_idx, : tracked_points[reference_frame][asset_name][0].shape[0], :] = pts

            if viz:
                rr.log(f"world/{asset_name}/orig_depth_unproj", rr.Points3D(cur_xyz[0, coords[:, 1], coords[:, 0]]))

            cur_xyz[:, coords[:, 1], coords[:, 0]] = pts

            bs = pts.shape[0]
            pts = rearrange(pts, "b n c -> (b n) c")
            uv, z = reprojection(pts, K_data[reference_frame_idx], RT_data[reference_frame_idx])

            uv = rearrange(uv, "(b n) c -> b n c", b=bs)
            z = rearrange(z, "(b n) c -> b n c", b=bs)
            cur_depth[:, coords[:, 1], coords[:, 0]] = z

            within_bounds = (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
            
            for i in range(bs):
                points_outside_image[reference_frame_idx, i, asset_idx] = (~within_bounds[i]).sum()

        if viz:
            for i in range(cur_depth.shape[0]):
                rr.log(f"world/diff_pcd_{i}", rr.Points3D(cur_xyz[i].reshape(-1, 3)))

        cur_xyz = cur_xyz.astype(np.float16)
        for i in range(cur_depth.shape[0]):
            np.savez(save_dynamic_depths_root / f"depth_{idx_to_str_5(reference_frame_idx)}_{idx_to_str_5(i)}.npz", xyz=cur_xyz[i])

    K_data = K_data.astype(np.float16)
    RT_data = RT_data.astype(np.float16)

    # save annotations as npz
    np.savez(
        os.path.join(data_root, "annotations.npz"),
        intrinsics=K_data,
        extrinsics=RT_data,
        points_outside_image=points_outside_image,
    )

    np.savez_compressed(
        os.path.join(data_root, "pixel_aligned_tracks.npz"),
        pixel_aligned_tracks=pixel_aligned_tracks,
    )

    return pixel_aligned_tracks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default="results/outdoor/0")
    parser.add_argument("--cp_root", type=Path, default="results/outdoor/0")
    parser.add_argument("--viz", action="store_true")
    args = parser.parse_args()

    with breakpoint_on_error():
        tracking(args.cp_root, args.data_root, args.viz)
