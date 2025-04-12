import open3d as o3d
import numpy as np
import scipy.spatial.transform as tf
import time
import json

previous_pos = None
previous_quat = None
previous_time = None

# Function to generate 7-dimensional action vector from point cloud
def generate_action_vector(points):
    center = np.mean(points, axis=0)
    eef_delta = center - previous_pos if previous_pos is not None else np.zeros(3)

    try:
        cov = np.cov(points.T) if points.shape[0] > 2 else np.eye(3)
        _, eigvecs = np.linalg.eigh(cov)
        rotation_matrix = eigvecs[:, ::-1]
        eef_quat = tf.Rotation.from_matrix(rotation_matrix).as_quat()
        eef_rotvec = tf.Rotation.from_quat(eef_quat).as_rotvec()
    except np.linalg.LinAlgError:
        print("Error: Rotation estimation failed.")
        eef_rotvec = np.zeros(3)

    gripper_action = -1.0 if center[2] < np.median(points[:, 2]) else 1.0
    
    return np.concatenate([eef_delta, eef_rotvec, [gripper_action]])

# Simulate reading multiple point clouds
results = []

for i in range(2):
    pcd = o3d.io.read_point_cloud(f"data/0/Cleaned/giver_frame{i+488}.ply")
    points = np.asarray(pcd.points)
    current_time = time.time()

    action = generate_action_vector(points)

    actions = {
        "action_vector": action.tolist()
    }
    results.append(actions)

    previous_pos, previous_time = np.mean(points, axis=0), current_time

with open("action_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Action results saved to action_results.json")
