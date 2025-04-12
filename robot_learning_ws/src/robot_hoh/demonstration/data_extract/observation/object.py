import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R

def load_point_cloud(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        if len(points) == 0:
            raise ValueError("Point cloud is empty.")
        return points
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return None

def compute_position(points):
    return np.mean(points, axis=0)

def compute_orientation(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    rotation_matrix = pca.components_.T
    quat = R.from_matrix(rotation_matrix).as_quat()
    return quat

def compute_velocity(points_current, points_previous=None, dt=1/30):
    centroid_current = compute_position(points_current)
    if points_previous is None:
        return np.array([0.0, 0.0, 0.0])  # No previous frame, velocity is zero
    centroid_previous = compute_position(points_previous)
    velocity = (centroid_current - centroid_previous) / dt
    return velocity

# Load current and optional previous point clouds
points_current = load_point_cloud('data/0/Cleaned/object_frame488.ply')
points_previous = load_point_cloud('data/0/Cleaned/object_frame487.ply') if 'data/0/Cleaned/object_frame487.ply' else None

# Compute object state
if points_current is not None:
    position = compute_position(points_current)
    orientation = compute_orientation(points_current)
    velocity = compute_velocity(points_current, points_previous)

    # Display results
    print("Object Position [x, y, z]:", position)
    print("Object Orientation [qx, qy, qz, qw]:", orientation)
    print("Object Linear Velocity [vx, vy, vz]:", velocity)
else:
    print("No valid point cloud data available.")
