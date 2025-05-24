import cv2
import numpy as np
import os
import json
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def backproject(depth, intrinsics, instance_mask, NOCS_convention=False):
    intrinsics_inv = np.linalg.inv(intrinsics)
    # image_shape = depth.shape
    # width = image_shape[1]
    # height = image_shape[0]

    # x = np.arange(width)
    # y = np.arange(height)

    # non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = depth > 0
    final_instance_mask = np.logical_and(instance_mask, non_zero_mask)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    # shape: height * width
    # mesh_grid = np.meshgrid(x, y) #[height, width, 2]
    # mesh_grid = np.reshape(mesh_grid, [2, -1])
    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
    xyz = np.transpose(xyz)  # [num_pixel, 3]

    z = depth[idxs[0], idxs[1]]

    # print(np.amax(z), np.amin(z))
    pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
    if NOCS_convention:
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]

    return pts, idxs


def visualize_sphere_o3d(center, color=[1, 0, 0], size=0.03):
    # center
    center_o3d = o3d.geometry.TriangleMesh.create_sphere()
    center_o3d.compute_vertex_normals()
    center_o3d.scale(size, [0, 0, 0])
    center_o3d.translate(center)
    center_o3d.paint_uniform_color(color)
    return center_o3d

def visualize_points_o3d(points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

output_dir = "outputs/"
obs_files = [f for f in os.listdir(output_dir) if f.endswith(".npz")]
obs_files.sort()
vis_o3d = []
 
rot_diff = R.from_euler('xyz', [180, 0, 180], degrees=True).as_matrix()
for obs_file_id in range(1, len(obs_files)-2):
    obs_file = f"obs_{obs_file_id * 10}.npz"
    obs = np.load(os.path.join(output_dir, obs_file))
    intrinsics = obs["frontview_intrinsics"]
    color = obs["frontview_image"][:, :, :]
    depth = obs["frontview_depth"][:, :, 0] 

    world_camera_pose =  obs["world_camera_pose"]
    world_camera_pose2 = obs["world_camera_pose2"]
    file_camera_pose = obs["file_camera_pose"]
    world_camera_rot = world_camera_pose[:3, :3]
    world_camera_rot2 = world_camera_pose2[:3, :3]
    world_camera_tra = world_camera_pose[:3, 3]
    world_camera_tra2 = world_camera_pose2[:3, 3]
    # world_camera_rot = rot_diff @ world_camera_rot
    # rot_diff = world_camera_rot2 @ world_camera_rot.T
    # rot_deg = R.from_matrix(rot_diff).as_euler('xyz', degrees=True)
    # tra_diff = world_camera_tra2 - world_camera_tra
    # print(rot_deg)
    # print(tra_diff)
    pose = world_camera_pose2
    
    points, idxs = backproject(depth, intrinsics, depth>0)
    points = points @ pose[:3, :3].T + pose[:3, 3]
    point_colors = color[idxs[0], idxs[1], :] / 255.0
    pcd = visualize_points_o3d(points, point_colors)
    vis_o3d.append(pcd)
    # o3d.visualization.draw_geometries([pcd])

o3d.visualization.draw(vis_o3d)