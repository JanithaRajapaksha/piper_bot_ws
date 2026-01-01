import numpy as np
import open3d as o3d

# Load npy file
points = np.load("captured_masks/cloud_56_1_1767251150.npy")  # shape (N,3) or (N,6)

# If points contain normals or colors
xyz = points[:, :3]

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# Optional: estimate normals if not present
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.05, max_nn=30
    )
)

# Visualize
o3d.visualization.draw_geometries([pcd])
