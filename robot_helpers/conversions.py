import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
def map_cloud_to_grid(voxel_size, points, values):
    # grid = np.zeros((40, 40, 40), dtype=np.float32)
    grid = np.zeros((60, 60, 60), dtype=np.float32)##学習時に変更

    indices = np.round(points / voxel_size).astype(int)
    grid[tuple(indices.T)] = values.squeeze()
    return grid


def grid_to_map_cloud(voxel_size, grid, threshold=0.0):
    grid_shape = grid.shape
    grid_indices = np.indices(grid_shape).reshape(3, -1).T 
    grid_values = grid.flatten() 

    mask = grid_values >= threshold
    indices = grid_indices[mask]
    values = grid_values[mask]

    points = indices * voxel_size
    values = values[:, None] 
    return points, values





def map_clouds_to_grid_instance(voxel_size, points, instance_ids, values=None, grid_shape=(60, 60, 60)):
    unique_ids = np.unique(instance_ids)
    grid = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2], len(unique_ids)), dtype=np.float32)
    indices = np.round(points / voxel_size).astype(int)
    valid_mask = np.all((indices >= 0) & (indices < np.array(grid_shape)), axis=1)
    indices = indices[valid_mask]
    instance_ids = instance_ids[valid_mask]
    if values is not None:
        values = values[valid_mask]

    for i, uid in enumerate(unique_ids):
        mask = (instance_ids == uid)
        idx = indices[mask]
        grid[..., i][tuple(idx.T)] = values[mask].squeeze()

    pts_list, colors_list = [], []
    rng = np.random.RandomState(0)
    cmap = rng.rand(len(unique_ids), 3)

    for i in range(len(unique_ids)):
        grid_i = grid[i]
        mask = grid_i > 0.0
        if not np.any(mask):
            continue
        idx = np.argwhere(mask)
        pts = idx.astype(np.float32) 
        col = np.tile(cmap[i], (pts.shape[0], 1))
        pts_list.append(pts)
        colors_list.append(col)


##visualize all instances together
    # all_pts = np.vstack(pts_list)
    # all_colors = np.vstack(colors_list)
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(all_pts[:, 0], all_pts[:, 1], all_pts[:, 2], c=all_colors, s=1)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_xlim(0,60)
    # ax.set_ylim(0,60)
    # ax.set_zlim(0,60)
    # plt.title('All Instances')
    # plt.show()

    return grid, unique_ids


def grid_to_map_clouds_instance(voxel_size, occupancy_voxel_instance, threshold=0.0):
    X, Y, Z, num_instances = occupancy_voxel_instance.shape

    points_list, values_list, ids_list = [], [], []

    for i in range(num_instances):
        grid_i = occupancy_voxel_instance[:, :, :, i]  
        mask = grid_i >= threshold
        if not np.any(mask):
            continue

        idx = np.argwhere(mask)  # (K, 3)
        pts = idx.astype(np.float32) * float(voxel_size)
        vals = grid_i[mask].astype(np.float32)

        points_list.append(pts)
        values_list.append(vals)
        ids_list.append(np.full(len(vals), i, dtype=np.int32))

    if points_list:
        points = np.vstack(points_list)
        values = np.concatenate(values_list)
        instance_ids = np.concatenate(ids_list)
    else:
        points = np.empty((0, 3), dtype=np.float32)
        values = np.empty((0,), dtype=np.float32)
        instance_ids = np.empty((0,), dtype=np.int32)

    return points, values, instance_ids
