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

"""
For each frame:
Step 1: Unproject all HW depths. Transform from camera space -> scene space (inverse of the reprojection transformation).
Step 2: Perform ray-mesh intersection with the character mesh in scene space.
Step 3: Convert the rendered depths to a scene space PCD and for each point in the 2 point clouds, check if it is within a given tolerance. Create a final array with the points on the mesh that match the rendered depth.
Step 4: Get barycentric coordinates on the mesh for each valid ray-mesh intersectionn.
Step 5: Run an inner for loop for all other frames, and use the barycentric coordinates to interpolate the determine the new XYZ for the point. Assume that the character mesh has the same number of vertices/faces over all frames, but the vertices move between frames.
"""




def read_obj_file(obj_path:str):
    '''
        Load .obj file, return vertices, faces.
        return: vertices: N_v X 3, faces: N_f X 3
        '''
    obj_f = open(obj_path, 'r')
    lines = obj_f.readlines()
    vertices = []
    faces = []
    for ori_line in lines:
        line = ori_line.split()
        if line[0] == 'v':
            vertices.append([float(line[1]), float(line[2]), float(line[3])])  # x, y, z
        elif line[0] == 'f':  # Need to consider / case, // case, etc.
            faces.append([int(line[3].split('/')[0]),
                          int(line[2].split('/')[0]),
                          int(line[1].split('/')[0]) \
                          ])  # Notice! Need to reverse back when using the face since here it would be clock-wise!
            # Convert face order from clockwise to counter-clockwise direction.
    obj_f.close()

    return np.asarray(vertices), np.asarray(faces)


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
            d_max = np.max(depth[v_low:v_high + 1, u_low:u_high + 1])
            d_min = np.min(depth[v_low:v_high + 1, u_low:u_high + 1])
            d_median = np.median(depth[v_low:v_high + 1, u_low:u_high + 1])
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
    max_idx = np.random.randint(0, p.shape[0] -1)
    farthest_point[0] = p[max_idx]
    idx.append(max_idx)
    print('farthest point sampling')
    for i in range(1, K):
        pairwise_distance = np.linalg.norm(p[:, None, :] - farthest_point[None, :i, :], axis=2)
        distance = np.min(pairwise_distance, axis=1, keepdims=True)
        max_idx = np.argmax(distance)
        farthest_point[i] = p[max_idx]
        idx.append(max_idx)
    print('farthest point sampling done')
    return farthest_point, idx

def compute_camera_rays(w, h, K, RT):
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    
    # Flatten the grid
    i_flat = i.flatten()
    j_flat = j.flatten()
    
    # Create homogeneous pixel coordinates
    pixel_coords = np.vstack((i_flat, j_flat, np.ones_like(i_flat)))
    
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
    
def tracking(cp_root: str, data_root: str, sampling_scene_num=100000, sampling_character_num=5000, start_frame=0, end_frame=2000):
    img_root = os.path.join(cp_root, 'images')
    exr_root = os.path.join(cp_root, 'exr_img')
    cam_root = os.path.join(cp_root, 'cam')
    obj_root = os.path.join(cp_root, 'obj')


    print('copying exr data ...')

    save_rgbs_root = os.path.join(data_root, 'rgbs')
    save_depths_root = os.path.join(data_root, 'depths')
    save_masks_root = os.path.join(data_root, 'masks')
    save_normals_root = os.path.join(data_root, 'normals')

    os.makedirs(save_rgbs_root, exist_ok=True)
    os.makedirs(save_depths_root, exist_ok=True)
    os.makedirs(save_masks_root, exist_ok=True)
    os.makedirs(save_normals_root, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(img_root, '*.png')))[start_frame:end_frame]

    # save_vis_dir = os.path.join(data_root, 'tracking')

    tracking_results = []
    tracking_results_3d = []
    K_data = []
    RT_data = []

    R1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    R2 = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    R3 = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    scene_mesh = trimesh.load(os.path.join(obj_root, 'scene.obj'))
    scene_points = trimesh.sample.sample_surface(scene_mesh, sampling_scene_num)[0]
    print('scene points shape', scene_points.shape)

    # filter out points that are invisible in the most of the frames
    print('filtering...')
    mask_s = np.zeros((scene_points.shape[0], 1), dtype=np.bool_)
    for i in tqdm(range(start_frame, end_frame - 1)[::50]):
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i).zfill(4))))
        RT = R3 @ R2 @ RT @ R1
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i).zfill(5))))
        h, w, _ = depth.shape
        uv, z = reprojection(scene_points, K, RT)
        visibility = check_visibility(uv, z, depth, h, w)
        mask_s = mask_s | (visibility == 1)
    print('filtering done')
    mask_idx = np.where(mask_s == 1)[0]
    scene_points = scene_points[mask_idx]
    print('scene points shape', scene_points.shape)

    if sampling_character_num > 0:
        c_obj, _ = read_obj_file(os.path.join(obj_root, 'character_0001.obj'))
        sampling_idx = np.random.choice(len(c_obj), sampling_character_num, replace=False if len(c_obj) > sampling_character_num else True)
    else:
        sampling_idx = None

    viz = True
    if viz:
        import rerun as rr
        import rerun.blueprint as rrb
        
        blueprint = rrb.Horizontal(
            rrb.Spatial3DView(name="3D", origin="world"),
            rrb.Grid([
                rrb.Spatial2DView(
                    name="rgb",
                    origin=f"world/camera/image/rgb",
                ),
                rrb.Spatial2DView(
                    name="depth",
                    origin=f"world/camera/image/depth",
                )
            ]),
        )

        rr.init("rerun_example_my_data", default_blueprint=blueprint)
        rr.serve(web_port=0, ws_port=0)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    points_to_track = dict()
    loaded_meshes = dict()

    for i in tqdm(range(start_frame, end_frame - 1)):
        K = np.loadtxt(os.path.join(cam_root, 'K_{}.txt'.format(str(i + 1).zfill(4))))
        RT = np.loadtxt(os.path.join(cam_root, 'RT_{}.txt'.format(str(i + 1).zfill(4))))
        RT = R3 @ R2 @ RT @ R1
        depth = read_tiff(os.path.join(exr_root, 'depth_{}.tiff'.format(str(i + 1).zfill(5))))
        img = cv2.imread(os.path.join(exr_root, 'rgb_{}.png'.format(str(i + 1).zfill(5))))
        h, w, _ = img.shape

        # convert img to jpg
        save_img_path = os.path.join(save_rgbs_root, 'rgb_{}.jpg'.format(str(i).zfill(5)))
        cv2.imwrite(save_img_path, img)

        # convert depth to 16 bit png
        save_depth_path = os.path.join(save_depths_root, 'depth_{}.png'.format(str(i).zfill(5)))
        max_value = 1000
        min_value = 0
        data = depth.copy()
        data[data > max_value] = max_value
        data[data < min_value] = min_value
        data = (data - min_value) * 65535 / (max_value - min_value)
        data = data.astype(np.uint16)
        write_png(data, save_depth_path)

        # cp normals and masks
        save_normal_path = os.path.join(save_normals_root, 'normal_{}.jpg'.format(str(i).zfill(5)))
        save_mask_path = os.path.join(save_masks_root, 'mask_{}.png'.format(str(i).zfill(5)))
        save_normal = cv2.imread(os.path.join(exr_root, 'normal_{}.png'.format(str(i + 1).zfill(5))))
        cv2.imwrite(save_normal_path, save_normal)

        if os.path.exists(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5)))):
            shutil.copy(os.path.join(exr_root, 'segmentation_{}.png'.format(str(i + 1).zfill(5))), save_mask_path)

        uv, z = reprojection(scene_points, K, RT)
        visibility = check_visibility(uv, z, depth, h, w)
        save_trajs_3d = scene_points
        if sampling_character_num > 0:
            c_obj, _ = read_obj_file(os.path.join(obj_root, 'character_{}.obj'.format(str(i + 1).zfill(4))))
            c_obj = np.array(c_obj)[sampling_idx]
            uv_, z_ = reprojection(c_obj, K, RT)
            visibility_ = check_visibility(uv_, z_, depth, h, w)
            uv = np.concatenate([uv, uv_], axis=0)
            visibility = np.concatenate([visibility, visibility_], axis=0)
            save_trajs_3d = np.concatenate([save_trajs_3d, c_obj], axis=0)

        tracking_results.append(np.concatenate((uv, visibility), axis=1).astype(np.float16))
        tracking_results_3d.append(save_trajs_3d.astype(np.float16))
        K_data.append(K.astype(np.float16))
        RT_data.append(RT.astype(np.float16))

        if viz:
            seg_mask = cv2.imread(save_mask_path)
            static_mask = (seg_mask == 0).all(-1)
            h, w, _ = data.shape

            trajs = tracking_results_3d[-1]
            cam_intrinsic = K
            cam_extrinsic = RT

            depth = data.astype(np.float32) / 65535.0 * 1000.0
            uv, Z = reprojection(trajs, cam_intrinsic, cam_extrinsic)

            uv = np.round(uv).astype(np.int32)
            for j in range(len(uv)):
                u, v = uv[j]
                z = Z[j]
                if 0 < u < w and 0 < v < h:
                    d = depth[int(v), int(u)]
                    if d > z - 0.15 and d < z + 0.15:
                        img[int(v), int(u), :] = np.array([255, 255, 255])

            rr.set_time_sequence("frame", i)

            rr.log("world/tracked_3d", rr.Points3D(trajs))
            rr.log("world/camera/depth", rr.DepthImage(depth))
            rr.log("world/camera/rgb", rr.Image(img))
            rr.log("world/camera", rr.Transform3D(rr.TranslationAndMat3x3(translation=cam_extrinsic[:3, 3], mat3x3=cam_extrinsic[:3, :3], from_parent=True)))
            rr.log(
                "world/camera",
                rr.Pinhole(
                    resolution=[w, h],
                    focal_length=[cam_intrinsic[0, 0], cam_intrinsic[1, 1]],
                    principal_point=[cam_intrinsic[0, 2], cam_intrinsic[1, 2]],
                ),
                static=True
            )

            character_mesh_path = os.path.join(obj_root, 'character_{}.obj'.format(str(i + 1).zfill(4)))
            mesh = trimesh.load(character_mesh_path)
            loaded_meshes[i] = mesh

            origins, directions = compute_camera_rays(w, h, K, RT)
            origins = origins.reshape(h, w, 3)
            directions = directions.reshape(h, w, 3)

            origins = origins[~static_mask]
            directions = directions[~static_mask]

            rr.log("world/camera_rays", rr.Arrows3D(origins=origins, vectors=directions * 5))

            # Perform ray-mesh intersection using trimesh
            ray_mesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh, scale_to_box=False)
            intersection_points, index_ray, index_faces = ray_mesh_intersector.intersects_location(
                ray_origins=origins,
                ray_directions=directions,
                multiple_hits=False
            )

            intersection_points_barycentric_coordinates = trimesh.triangles.points_to_barycentric(mesh.triangles[index_faces], points=intersection_points)

            # Create a boolean array indicating which points intersect with the mesh
            intersection_mask = np.zeros(origins.shape[0], dtype=bool)
            intersection_mask[index_ray] = True

            intersection_dense = np.zeros_like(origins)
            intersection_dense[index_ray] = intersection_points

            uv_intersected, z_intersected = reprojection(intersection_dense, K, RT)

            # check distance for valid indices
            distance_diff = z_intersected[intersection_mask].squeeze(-1) - depth.squeeze(-1)[~static_mask][intersection_mask]

            init_ray = origins[intersection_mask]
            endpoints = (intersection_dense - origins)[intersection_mask]
            rr.log("world/camera_rays_mesh_intersection", rr.Arrows3D(origins=init_ray, vectors=endpoints))

            camera_ray_dirs = endpoints / np.linalg.norm(endpoints, axis=-1, keepdims=True)
            rr.log("world/mesh_intersection_depth_delta", rr.Arrows3D(origins=init_ray + endpoints, vectors=camera_ray_dirs * distance_diff[..., None]))

            points_to_track[i] = (intersection_points, intersection_points_barycentric_coordinates, index_faces)

    tracked_points = defaultdict(list)
    for i in sorted(points_to_track.keys()):
        rr.set_time_sequence("frame", i)
        intersection_points, intersection_points_barycentric_coordinates, index_faces = points_to_track[i]
        for j in sorted(points_to_track.keys()):
            points_on_mesh = trimesh.triangles.barycentric_to_points(loaded_meshes[j].triangles[index_faces], intersection_points_barycentric_coordinates)
            if abs(i - j) < 4:
                rr.log(f"world/tracked_3d_frame_diff_{j - i}", rr.Points3D(points_on_mesh))
                if len(tracked_points[i]) != 0:
                    rr.log(f"world/tracked_3d_delta_frame_diff_{j - i}", rr.Arrows3D(origins=tracked_points[i][-1], vectors=points_on_mesh - tracked_points[i][-1]))

            tracked_points[i].append(points_on_mesh)

    max_num_pts = max(i[0].shape[0] for i in points_to_track.values())
    window_size = len(points_to_track)
    pixel_aligned_tracks = np.full((len(points_to_track), window_size, max_num_pts, 3), dtype=np.float16, fill_value=np.nan)
    for idx, i in enumerate(points_to_track.keys()):
        pixel_aligned_tracks[idx, :, :tracked_points[i][0].shape[0], :] = np.stack(tracked_points[i], axis=0)

    tracking_results = np.stack(tracking_results, axis=0)
    tracking_results = tracking_results.astype(np.float16)

    tracking_results_3d = np.stack(tracking_results_3d, axis=0)
    tracking_results_3d = tracking_results_3d.astype(np.float16)

    K_data = np.stack(K_data, axis=0)
    K_data = K_data.astype(np.float16)

    RT_data = np.stack(RT_data, axis=0)
    RT_data = RT_data.astype(np.float16)

    # save annotations as npz
    np.savez(
        os.path.join(data_root, 'annotations.npz'),
        trajs_2d=tracking_results[:, :, :2],
        trajs_3d=tracking_results_3d,
        visibilities=tracking_results[:, :, 2],
        intrinsics=K_data,
        extrinsics=RT_data,
        pixel_aligned_tracks=pixel_aligned_tracks
    )

    return tracking_results

if __name__ == '__main__':
    import argparse

    np.random.seed(128)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/Users/yangzheng/code/project/long-term-tracking/data/scenes/render0')
    parser.add_argument('--cp_root', type=str, default='/Users/yangzheng/code/project/long-term-tracking/data/scenes/render0')
    parser.add_argument('--sampling_scene_num', type=int, default=20000)
    parser.add_argument('--sampling_character_num', type=int, default=5000)
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=2000)
    args = parser.parse_args()

    print(args.start_frame, args.end_frame)

    tracking(args.cp_root, args.data_root, args.sampling_scene_num, args.sampling_character_num, args.start_frame, args.end_frame)