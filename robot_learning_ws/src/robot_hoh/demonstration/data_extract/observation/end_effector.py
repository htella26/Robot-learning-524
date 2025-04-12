import open3d as o3d
import numpy as np
import scipy.spatial.transform as tf
import time
import json

previous_pos = None
previous_quat = None
previous_time = None

# Function to extract joint and gripper data from point cloud
def extract_joint_and_gripper_data(points):
    center = np.mean(points, axis=0)
    gripper_qpos = [center[0] * 0.01, -center[0] * 0.01]
    gripper_qvel = np.gradient(gripper_qpos)
    joint_pos = np.mean(points[:7], axis=0)
    joint_vel = np.gradient(joint_pos)
    joint_pos_cos = np.cos(joint_pos)
    joint_pos_sin = np.sin(joint_pos)
    return gripper_qpos, gripper_qvel, joint_pos, joint_pos_cos, joint_pos_sin, joint_vel

# Function to estimate pose from point cloud
def estimate_pose(points):
    if points.size == 0:
        return np.array([np.nan, np.nan, np.nan]), tf.Rotation.identity().as_quat()
    
    eef_pos = np.nanmean(points, axis=0)
    try:
        cov = np.cov(points.T) if points.shape[0] > 2 else np.eye(3)
        eigvals, eigvecs = np.linalg.eigh(cov)
        rotation_matrix = eigvecs[:, ::-1]
        eef_quat = tf.Rotation.from_matrix(rotation_matrix).as_quat()
    except np.linalg.LinAlgError:
        print("Error: Eigenvalues did not converge. Using identity rotation.")
        eef_quat = tf.Rotation.identity().as_quat()
    return eef_pos, eef_quat

# Function to estimate velocity
def estimate_velocity(current_pos, current_quat, current_time):
    global previous_pos, previous_quat, previous_time
    if previous_pos is None or previous_time is None:
        previous_pos, previous_quat, previous_time = current_pos, current_quat, current_time
        return np.zeros(3), np.zeros(3)

    dt = current_time - previous_time
    lin_vel = (current_pos - previous_pos) / dt
    rot_diff = tf.Rotation.from_quat(current_quat).inv() * tf.Rotation.from_quat(previous_quat)
    ang_vel = rot_diff.as_rotvec() / dt

    previous_pos, previous_quat, previous_time = current_pos, current_quat, current_time
    return lin_vel, ang_vel

# Simulate reading multiple point clouds
results = []

for i in range(2):
    pcd = o3d.io.read_point_cloud(f"data/0/Cleaned/giver_frame{i+488}.ply")
    points = np.asarray(pcd.points)
    current_time = time.time()
    eef_pos, eef_quat = estimate_pose(points)
    eef_vel_lin, eef_vel_ang = estimate_velocity(eef_pos, eef_quat, current_time)
    
    gripper_qpos, gripper_qvel, joint_pos, joint_pos_cos, joint_pos_sin, joint_vel = extract_joint_and_gripper_data(points)

    observation = {
        "robot0_gripper_qpos": gripper_qpos,
        "robot0_gripper_qvel": gripper_qvel.tolist(),
        "robot0_joint_pos": joint_pos.tolist(),
        "robot0_joint_pos_cos": joint_pos_cos.tolist(),
        "robot0_joint_pos_sin": joint_pos_sin.tolist(),
        "robot0_joint_vel": joint_vel.tolist(),
        "robot0_eef_pos": eef_pos.tolist(),
        "robot0_eef_quat": eef_quat.tolist(),
        "robot0_eef_vel_lin": eef_vel_lin.tolist(),
        "robot0_eef_vel_ang": eef_vel_ang.tolist()
    }
    results.append(observation)

with open("eef_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Results saved to eef_results.json")
