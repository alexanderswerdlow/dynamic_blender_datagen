import numpy as np
import cv2
import os
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from scipy.interpolate import griddata

def save_to_ply(data, filename):
    with open(filename, 'w') as ply:
        # PLY header
        ply.write("ply\n")
        ply.write("format ascii 1.0\n")
        ply.write("element vertex {}\n".format(data.shape[0]))  # only non-zero values
        ply.write("comment vertices\n")
        ply.write("property float x\n")
        ply.write("property float y\n")
        ply.write("property float z\n")
        # ply.write("property float value\n")  # or use "uchar red", "uchar green", "uchar blue" for RGB colors
        # Adding properties for RGB colors

        ply.write("end_header\n")

        # PLY data
        for (x, y, z) in data:
            ply.write("{} {} {} \n".format(x, y, z))


def reprojection(points, K, RT):
    v = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    XYZ = (RT @ v.T).T[:, :3]
    Z = XYZ[:, 2:]
    XYZ = XYZ / XYZ[:, 2:]
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return uv, Z

def inverse_projection(depth, K, RT):
    h, w = depth.shape

    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    # u = w - u - 1 # for v1.0 dataset


    uv_homogeneous = np.vstack((u.flatten(), v.flatten(), np.ones_like(u.flatten())))

    K_inv = np.linalg.inv(K)

    # use max depth as 10m for visualization
    depth = depth.flatten()
    mask = depth < 10

    XYZ = K_inv @ uv_homogeneous * depth

    XYZ = np.vstack((XYZ, np.ones(XYZ.shape[1])))
    world_coordinates = np.linalg.inv(RT) @ XYZ
    world_coordinates = world_coordinates[:3, :].T
    world_coordinates = world_coordinates[mask]

    return world_coordinates


if __name__ == '__main__':
    data_path = 'sample'
    scene_name = 'r4_new_f'
    annotations = np.load(f'{data_path}/{scene_name}/anno.npz')
    trajs_3d = annotations['trajs_3d'].astype(np.float32)
    cam_ints = annotations['intrinsics'].astype(np.float32)
    cam_exts = annotations['extrinsics'].astype(np.float32)
    num_frames = len(trajs_3d)
    num_frames = min(num_frames, 60)

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
    rr.spawn()
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    for index in range(num_frames):
        depth_16bit = cv2.imread(f'{data_path}/{scene_name}/depths/depth_{index:05d}.png', cv2.IMREAD_ANYDEPTH)
        img = cv2.imread(f'{data_path}/{scene_name}/rgbs/rgb_{index:05d}.jpg')
        seg_mask = cv2.imread(f'{data_path}/{scene_name}/masks/mask_{index:05d}.png')
        static_mask = (seg_mask == 0).all(-1)
        h, w = depth_16bit.shape

        trajs = trajs_3d[index]
        cam_intrinsic = cam_ints[index]
        cam_extrinsic = cam_exts[index]

        depth = depth_16bit.astype(np.float32) / 65535.0 * 1000.0
        depth_inv = inverse_projection(depth, cam_intrinsic, cam_extrinsic)

        uv, Z = reprojection(trajs, cam_intrinsic, cam_extrinsic)

        uv = np.round(uv).astype(np.int32)
        for i in range(len(uv)):
            u, v = uv[i]
            z = Z[i]
            if 0 < u < w and 0 < v < h:
                d = depth[int(v), int(u)]
                if d > z - 0.15 and d < z + 0.15:
                    img[int(v), int(u), :] = np.array([255, 255, 255])

        rr.set_time_sequence("frame", index)

        rr.log("world/trajs", rr.Points3D(trajs))
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

        delta_trajs = trajs - trajs_3d[0]
        non_static = (delta_trajs > 0).any(axis=-1)

        rr.log("world/trajs_init", rr.Points3D(trajs_3d[0]), static=True)
        rr.log("world/trajs_delta", rr.Arrows3D(origins=trajs_3d[0][non_static], vectors=delta_trajs[non_static]))

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        channels = [griddata(uv, delta_trajs[:, i], (grid_y, grid_x), method='cubic', fill_value=0) for i in range(3)]
        delta_image = np.stack(channels, axis=-1)
        delta_image[static_mask] = 0

        rr.log("world/camera/delta_img", rr.Image(delta_image))

        # for i in range(3):
        #     min_val = np.min(dense_image[:, :, i])
        #     max_val = np.max(dense_image[:, :, i])
        #     dense_image[:, :, i] = (dense_image[:, :, i] - min_val) / (max_val - min_val)

        # from image_utils import Im
        # Im(dense_image).save()

    # breakpoint()