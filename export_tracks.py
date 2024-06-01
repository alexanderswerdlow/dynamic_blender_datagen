import glob
import json
import os
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import trimesh
from einops import pack, rearrange, repeat, unpack
from tqdm import tqdm
from export_unified import RenderTap

import utils.plotting as plotting
from utils.decoupled_utils import breakpoint_on_error
from utils.file_io import read_tiff, write_png


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


def read_obj_file_triangles(obj_path: str):
    vertices, faces = read_obj_file(obj_path)
    return vertices[faces]


def unproject(depth, K, world2cam):
    H, W = depth.shape
    y, x = np.indices((H, W))
    ones = np.ones((H, W))
    pixel_positions = np.stack((x, y, ones), axis=-1)
    K_inv = np.linalg.inv(K)
    camera_coords = pixel_positions @ K_inv.T
    camera_coords *= depth[..., None]
    camera_coords_hom = np.concatenate([camera_coords, ones[..., None]], axis=-1)

    # world2cam is world -> camera so we invert
    R = world2cam[:3, :3]
    T = world2cam[:3, 3]
    world2cam_inv = np.eye(4)
    world2cam_inv[:3, :3] = R.T
    world2cam_inv[:3, 3] = -R.T @ T
    points = camera_coords_hom @ world2cam_inv.T

    return points[..., :3]


def reprojection(points: np.ndarray, K: np.ndarray, world2cam: np.ndarray):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (world2cam @ v.T).T[:, :3]
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


def compute_camera_rays(w, h, K, world2cam):
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
    world2cam_inv = np.linalg.inv(world2cam)
    rays_camera_homogeneous = np.vstack((rays_camera, np.ones((1, rays_camera.shape[1]))))
    rays_world_homogeneous = world2cam_inv @ rays_camera_homogeneous

    # Extract the origins and directions of the rays
    origins = world2cam_inv[:3, 3]
    directions = rays_world_homogeneous[:3, :] - origins[:, np.newaxis]

    # Normalize the directions
    directions = directions / np.linalg.norm(directions, axis=0)
    origins = np.tile(origins, (directions.shape[1], 1))

    return origins, directions.T


def idx_to_str_4(idx):
    return str(idx).zfill(4)


def idx_to_str_5(idx):
    return str(idx).zfill(5)

def find_value_in_txt(file_path, key):
    with open(file_path, "r") as file:
        for line in file:
            if line.startswith(key):
                return line.split("=")[1].strip()

def tracking(data_root: Path, viz=False, profile=False):
    print(f"Exporting tracks for {data_root}")

    save_raw_depth = False

    img_root = data_root / "images"
    exr_root = data_root / "exr_img"
    cam_root = data_root / "cam"
    obj_root = data_root / "obj" if (data_root / "obj").exists() else data_root / "obj"
    print("copying exr data ...")

    save_rgbs_root = data_root / "rgbs"
    save_depths_root = data_root / "depths"
    save_masks_root = data_root / "masks"
    
    if save_raw_depth: save_dynamic_depths_root = data_root / "dynamic_depths"
    save_dynamic_points_root = data_root / "dynamic_points"

    save_rgbs_root.mkdir(parents=True, exist_ok=True)
    save_depths_root.mkdir(parents=True, exist_ok=True)
    save_masks_root.mkdir(parents=True, exist_ok=True)
    if save_raw_depth: save_dynamic_depths_root.mkdir(parents=True, exist_ok=True)
    save_dynamic_points_root.mkdir(parents=True, exist_ok=True)

    frames = sorted(img_root.glob("*.png"))
    num_frames = len(frames)
    print(f"num_frames: {num_frames}")

    K_data = []
    world2cam_data = []

    R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R3 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    scene_info = json.load(open(os.path.join(data_root, "scene_info.json"), "r"))
    assets_blender = scene_info["assets"]
    assets_saved = scene_info["assets_saved"]
    assert len(assets_blender) == len(assets_saved)

    if assets_blender[0] != "background":
        print("WARNING: Adding background to the assets")
        assets_blender.insert(0, "background")

    pallette = plotting.hls_palette(len(assets_blender) + 2)

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
    reference_frame_idx_to_blender_frame_idx = dict()
    depth_data = dict()

    cache_meshes = True
    if cache_meshes:
        loaded_meshes = defaultdict(dict)

    max_depth_value = 1000
    min_depth_value = 0

    blender_start_frame = 1
    if (data_root / "config.json").exists():
        args = RenderTap()
        args.load(data_root / "config.json")
        blender_start_frame = args.start_frame

    blender_frame_idxs = list(range(blender_start_frame, blender_start_frame + num_frames))

    if blender_frame_idxs[0] != 1:
        assert blender_frame_idxs[-1] == args.end_frame

    print(f"Copying data and intersecting reference camera rays with object meshes...", flush=True)
    for reference_frame_idx, reference_frame in tqdm(enumerate(range(1, num_frames - 1))):
        blender_reference_frame = blender_frame_idxs[reference_frame]
        reference_frame_idx_to_blender_frame_idx[reference_frame_idx] = blender_reference_frame
        K = np.loadtxt(cam_root / f"K_{idx_to_str_4(blender_reference_frame)}.txt")
        world2cam = np.loadtxt(cam_root / f"RT_{idx_to_str_4(blender_reference_frame)}.txt")
        world2cam = R3 @ R2 @ world2cam @ R1
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
        saved_normal_path = str(exr_root / f"normal_{idx_to_str_5(blender_reference_frame)}.png")
        if os.path.exists(saved_normal_path):
            save_normals_root = data_root / "normals"
            save_normals_root.mkdir(parents=True, exist_ok=True)
            save_normal = cv2.imread(saved_normal_path)
            save_normal_path = save_normals_root / f"normal_{idx_to_str_5(reference_frame_idx)}.jpg"
            cv2.imwrite(str(save_normal_path), save_normal)

        save_mask_path = save_masks_root / f"mask_{idx_to_str_5(reference_frame_idx)}.png"
        if (exr_root / f"segmentation_{idx_to_str_5(blender_reference_frame)}.png").exists():
            shutil.copy(exr_root / f"segmentation_{idx_to_str_5(blender_reference_frame)}.png", save_mask_path)

        K_data.append(K)
        world2cam_data.append(world2cam)

        if viz:
            h, w, _ = data.shape

            cam_intrinsic = K
            cam_extrinsic = world2cam
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

        for idx in range(len(assets_saved)):
            asset = assets_saved[idx]
            asset_name = asset.replace(".", "_")

            if asset_name == "background":
                continue

            obj_path = os.path.join(obj_root, f"{asset_name}_{str(blender_reference_frame).zfill(4)}.obj")
            mesh = trimesh.load(str(obj_path))

            if not hasattr(mesh, "vertices"):
                print(f"Skipping {asset_name} because it has no geometry")
                continue

            asset_mask = np.logical_and(mask[:, :, 2] == pallette[idx + 1, 0], mask[:, :, 1] == pallette[idx + 1, 1])
            asset_mask = np.logical_and(asset_mask, mask[:, :, 0] == pallette[idx + 1, 2])

            if cache_meshes:
                loaded_meshes[reference_frame_idx][asset_name] = mesh

            origins, directions = compute_camera_rays(w, h, K, world2cam)
            origins = origins.reshape(h, w, 3)
            directions = directions.reshape(h, w, 3)

            origins = origins[asset_mask]
            directions = directions[asset_mask]
            asset_pixel_coords = pixel_coords[asset_mask]

            if viz:
                rr.log("world/camera_rays", rr.Arrows3D(origins=origins, vectors=directions * 5))

            # Perform ray-mesh intersection using trimesh
            if False:
                ray_module = trimesh.ray.ray_triangle
                ray_kwargs = dict()
            else:
                ray_module = trimesh.ray.ray_pyembree
                ray_kwargs = dict(scale_to_box=False)

            ray_mesh_intersector = ray_module.RayMeshIntersector(mesh, **ray_kwargs)
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

                uv_intersected, z_intersected = reprojection(intersection_dense, K, world2cam)
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

        if profile and reference_frame_idx > 32:
            break

    print(f"Tracking points in other frames...", flush=True)
    tracked_points = defaultdict(dict)

    if profile:
        from viztracer import VizTracer

        tracer = VizTracer(tracer_entries=2000000)
        tracer.start()

    for reference_frame_idx in tqdm(sorted(points_to_track.keys())):
        if viz:
            rr.set_time_sequence("frame", reference_frame_idx)
        for asset_name in points_to_track[reference_frame_idx].keys():
            intersection_points, intersection_points_barycentric_coordinates, index_faces, asset_pixel_coords = points_to_track[reference_frame_idx][
                asset_name
            ]

            all_object_triangles = np.stack(
                [loaded_meshes[target_frame_idx][asset_name].triangles for idx, target_frame_idx in enumerate(sorted(points_to_track.keys()))],
                axis=0,
            )
            all_points = rearrange(all_object_triangles[:, index_faces], "b n ... -> (b n) ...")
            all_barycentric_coords = repeat(intersection_points_barycentric_coordinates, "n c -> (b n) c", b=all_object_triangles.shape[0])
            all_points_on_mesh = trimesh.triangles.barycentric_to_points(all_points, all_barycentric_coords)
            all_points_on_mesh = rearrange(all_points_on_mesh, "(b n) ... -> b n ...", b=all_object_triangles.shape[0])
            tracked_points[reference_frame_idx][asset_name] = all_points_on_mesh

        if profile and reference_frame_idx > 1:
            break

    if profile:
        tracer.stop()
        tracer.save()
        exit()

    K_data = np.stack(K_data, axis=0)
    world2cam_data = np.stack(world2cam_data, axis=0)

    max_num_pts = max([i[0].shape[0] for idx in points_to_track.keys() for i in points_to_track[idx].values()])
    window_size = len(points_to_track)
    num_objects = len(points_to_track[next(iter(points_to_track.keys()))].keys())

    points_outside_image = np.full((len(points_to_track), window_size, num_objects), dtype=np.int64, fill_value=-1)
    points_inside_image = np.full((len(points_to_track), window_size, num_objects), dtype=np.int64, fill_value=-1)

    save_tracks = False
    if save_tracks:
        pixel_aligned_tracks = np.full((len(points_to_track), window_size, num_objects, max_num_pts, 3), dtype=np.float16, fill_value=np.nan)

    print(f"Saving...")
    for reference_frame_idx, reference_frame in tqdm(enumerate(points_to_track.keys())):
        if viz:
            rr.set_time_sequence("frame", reference_frame_idx)

        if save_raw_depth:
            cur_depth = repeat(depth_data[reference_frame], "h w () -> b h w ()", b=len(points_to_track))
            cur_xyz = repeat(
                unproject(
                    depth_data[reference_frame].squeeze(-1),
                    K_data[reference_frame_idx].astype(np.float32),
                    world2cam_data[reference_frame_idx].astype(np.float32),
                ),
                "h w c -> b h w c",
                b=window_size,
            )

            dynamic_mask = np.full_like(cur_xyz[..., 0], False, dtype=bool)

        all_pts = []
        all_coords = []
        all_asset_idxs = []

        if viz:
            rr.log(f"world/depth_unproj", rr.Points3D(cur_xyz[0].reshape(-1, 3)))

        for asset_idx, asset_name in enumerate(points_to_track[reference_frame].keys()):
            _, _, _, coords = points_to_track[reference_frame][asset_name]

            _data = tracked_points[reference_frame][asset_name]

            if type(_data) == list:
                pts = np.stack(_data, axis=0)
            else:
                pts = _data

            if save_tracks:
                pixel_aligned_tracks[reference_frame_idx, :, asset_idx, : tracked_points[reference_frame][asset_name][0].shape[0], :] = pts

            if viz:
                rr.log(f"world/{asset_name}/orig_depth_unproj", rr.Points3D(cur_xyz[0, coords[:, 1], coords[:, 0]]))

            if save_raw_depth:
                cur_xyz[:, coords[:, 1], coords[:, 0]] = pts
                dynamic_mask[:, coords[:, 1], coords[:, 0]] = True

            all_pts.append(pts)
            all_coords.append(coords)
            all_asset_idxs.append(np.ones((pts.shape[1],)) * asset_idx)

            bs = pts.shape[0]
            pts = rearrange(pts, "b n c -> (b n) c")
            uv, z = reprojection(pts, K_data[reference_frame_idx], world2cam_data[reference_frame_idx])

            uv = rearrange(uv, "(b n) c -> b n c", b=bs)
            z = rearrange(z, "(b n) c -> b n c", b=bs)
            if save_raw_depth:
                cur_depth[:, coords[:, 1], coords[:, 0]] = z

            within_bounds = (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)

            for i in range(bs):
                points_outside_image[reference_frame_idx, i, asset_idx] = (~within_bounds[i]).sum()
                points_inside_image[reference_frame_idx, i, asset_idx] = within_bounds[i].sum()

        if viz:
            for i in range(cur_depth.shape[0]):
                rr.log(f"world/diff_pcd_{i}", rr.Points3D(cur_xyz[i].reshape(-1, 3)))

        if save_raw_depth:
            cur_xyz = cur_xyz.astype(np.float16)
            for i in range(cur_depth.shape[0]):
                np.savez_compressed(
                    save_dynamic_depths_root / f"depth_{idx_to_str_5(reference_frame_idx)}_{idx_to_str_5(i)}.npz",
                    xyz=cur_xyz[i],
                    dynamic_mask=dynamic_mask[i],
                )

        all_pts = np.concatenate(all_pts, axis=1)
        all_coords = np.concatenate(all_coords, axis=0)
        all_asset_idxs = np.concatenate(all_asset_idxs, axis=0)

        assert np.all(all_pts < np.finfo(np.float16).max)
        all_pts = all_pts.astype(np.float16)

        assert np.all(all_coords < np.iinfo(np.uint16).max)
        all_coords = all_coords.astype(np.uint16)

        assert np.all(all_asset_idxs < np.iinfo(np.uint8).max)
        all_asset_idxs = all_asset_idxs.astype(np.uint8)

        all_asset_names = list(points_to_track[reference_frame].keys())
        
        np.savez_compressed(
            save_dynamic_points_root / f"meta_{idx_to_str_5(reference_frame_idx)}.npz",
            coords=all_coords,
            asset_idxs=all_asset_idxs,
            points_base=all_pts[reference_frame_idx].copy(),
            depth=depth_data[reference_frame].squeeze(-1).copy(),
            asset_names=all_asset_names,
        )

        all_pts -= all_pts[reference_frame_idx] # Helps w/compression
        for i in range(all_pts.shape[0]):
            np.savez_compressed(
                save_dynamic_points_root / f"points_{idx_to_str_5(reference_frame_idx)}_{idx_to_str_5(i)}.npz", points_delta=all_pts[i]
            )

    np.savez(os.path.join(data_root, "annotations.npz"), intrinsics=K_data, world2cam=world2cam_data)

    if save_tracks:
        np.savez_compressed(
            os.path.join(data_root, "pixel_aligned_tracks.npz"), pixel_aligned_tracks=pixel_aligned_tracks, points_outside_image=points_outside_image
        )
    else:
        np.savez_compressed(
            os.path.join(data_root, "track_metadata.npz"),
            points_outside_image=points_outside_image,
            points_inside_image=points_inside_image,
        )

    return None


def run_single_scene(**kwargs):
    with breakpoint_on_error():
        tracking(**kwargs)


import typer

typer.main.get_command_name = lambda name: name
app = typer.Typer(pretty_exceptions_enable=False, pretty_exceptions_show_locals=False)


def process_scene(scene, data_root, viz, profile):
    try:
        run_single_scene(data_root=data_root / scene, viz=viz, profile=profile)
    except Exception as e:
        print(f"Failed to process scene {scene}: {e}")


@app.command()
def main(output_dir: Path = Path("results/outdoor/0"), viz: bool = False, profile: bool = False, recursive: bool = False, num_workers: int = 2):
    data_root = Path(output_dir)
    print(f"Running with dir: {data_root}")
    if recursive:
        scene_list = [p.parent.relative_to(data_root) for p in data_root.rglob("scene_info.json")]
        print(scene_list)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_scene, scene, data_root, viz, profile): scene
                for scene in scene_list
                if (
                    (data_root / scene / "exr_img" / "depth_00001.png").exists() and
                    not (data_root / scene / "track_metadata.npz").exists() and
                    'v5' not in str(scene) and
                    len(list((data_root / scene / "exr_img").glob("*"))) >= 6 * len(list((data_root / scene / "exr").glob("*")))
                )
            }
            for future in as_completed(futures):
                scene = futures[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing scene {scene}: {e}")
    else:
        run_single_scene(data_root=data_root, viz=viz, profile=profile)


if __name__ == "__main__":
    app()

# singularity exec --bind /home/aswerdlo/repos/point_odyssey/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPY /home/aswerdlo/repos/point_odyssey/export_tracks.py --data_root=generated/v4 --recursive'


# singularity run --bind /home/aswerdlo/repos/point_odyssey/singularity/config:/.config --nv singularity/blender.sif --background --python /home/aswerdlo/repos/point_odyssey/scripts/export_scene_test_2.py -- --scene_root generated/v6/default/22/scene.blend --output_dir generated/v6/default/22

# singularity run --bind /home/aswerdlo/repos/point_odyssey/singularity/config:/.config --nv singularity/blender.sif --background --python /home/aswerdlo/repos/point_odyssey/utils/export_obj.py -- --scene_root generated/v6/default/22/scene.blend --output_dir generated/v6/default/22 --indoor True

# singularity exec --bind /home/aswerdlo/repos/point_odyssey/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPY /home/aswerdlo/repos/point_odyssey/utils/openexr_utils.py --data_dir generated/v6/default/22 --output_dir generated/v6/default/22/exr_img --batch_size 32 --frame_idx 1'

# singularity exec --bind /home/aswerdlo/repos/point_odyssey/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPY /home/aswerdlo/repos/point_odyssey/export_tracks.py --data_root=generated/v6/default/22'


# singularity exec --bind /home/aswerdlo/repos/point_odyssey/singularity/config:/.config --nv singularity/blender.sif /bin/bash -c '$BLENDERPIP install typed-argument-parser'
