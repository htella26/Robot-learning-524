import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import scipy.spatial.transform as tf
import time
import json
import h5py

previous_pos = None
previous_quat = None
previous_time = None


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
        eef_quat = tf.Rotation.identity().as_quat()
    return eef_pos, eef_quat

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
        return np.zeros(3)
    centroid_previous = compute_position(points_previous)
    velocity = (centroid_current - centroid_previous) / dt
    return velocity

def extract_joint_and_gripper_data(points):
    center = np.mean(points, axis=0)
    gripper_qpos = [center[0] * 0.01, -center[0] * 0.01]
    gripper_qvel = np.gradient(gripper_qpos)
    joint_pos = np.mean(points[:7], axis=0)
    joint_vel = np.gradient(joint_pos)
    return gripper_qpos, gripper_qvel, joint_pos, np.cos(joint_pos), np.sin(joint_pos), joint_vel

def generate_action_vector(points):
    global previous_pos
    center = compute_position(points)
    eef_delta = center - previous_pos if previous_pos is not None else np.zeros(3)
    eef_delta = np.clip(eef_delta / np.linalg.norm(eef_delta) if np.linalg.norm(eef_delta) != 0 else eef_delta, -1, 1)
    try:
        pca = PCA(n_components=3)
        pca.fit(points)
        rotation_matrix = pca.components_.T
        eef_quat = R.from_matrix(rotation_matrix).as_quat()
        eef_rotvec = R.from_quat(eef_quat).as_rotvec()
        eef_rotvec = np.clip(eef_rotvec / np.linalg.norm(eef_rotvec) if np.linalg.norm(eef_rotvec) != 0 else eef_rotvec, -1, 1)
    except Exception:
        eef_rotvec = np.zeros(3)
    gripper_action = -1.0 if center[2] < np.median(points[:, 2]) else 1.0
    return np.concatenate([eef_delta, eef_rotvec, [gripper_action]])

results = {
    "data": {

        },
    "mask": {},
     "env_args": {
        "env_name": "Lift",
        "env_version": "1.4.1",
        "type": 1,
        "env_kwargs": {
            "has_renderer": False,
            "has_offscreen_renderer": False,
            "ignore_done": True,
            "use_object_obs": True,
            "use_camera_obs": False,
            "control_freq": 20,
            "controller_configs": {
                "type": "OSC_POSE",
                "input_max": 1,
                "input_min": -1,
                "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                "kp": 150,
                "damping": 1,
                "impedance_mode": "fixed",
                "kp_limits": [0, 300],
                "damping_limits": [0, 10],
                "position_limits": None,
                "orientation_limits": None,
                "uncouple_pos_ori": True,
                "control_delta": True,
                "interpolation": None,
                "ramp_ratio": 0.2
            },
            "robots": ["Panda"],
            "camera_depths": False,
            "camera_heights": 84,
            "camera_widths": 84,
            "reward_shaping": False
        }
      },
 
    }

# Add the env_kwargs to results['data']
results['data']['env_args'] = results['env_args']

previous_obs = None  # Initialize previous_obs

# Create an HDF5 file to store the results
hdf5_filename = "modified_results.hdf5"
with h5py.File(hdf5_filename, 'w') as hdf5_file:
    # Create groups to mirror the dictionary structure
    data_group = hdf5_file.create_group("data")
    mask_group = hdf5_file.create_group("mask")

    for i in range(60):  # Update the range as needed
        points_current = load_point_cloud(f'data/0/Cleaned/giver_frame{i+489}.ply')
        object_points_current = load_point_cloud(f'data/0/Cleaned/object_frame{i+489}.ply')
        object_points_previous = load_point_cloud(f'data/0/Cleaned/object_frame{i+488}.ply') if i > 0 else None

        if points_current is None or object_points_current is None:
            continue

        current_time = time.time()
        position = compute_position(object_points_current)
        orientation = compute_orientation(object_points_current)
        velocity = compute_velocity(object_points_current, object_points_previous)
        eef_pos, eef_quat = estimate_pose(points_current)
        eef_vel_lin, eef_vel_ang = estimate_velocity(eef_pos, eef_quat, current_time)
        gripper_qpos, gripper_qvel, joint_pos, joint_pos_cos, joint_pos_sin, joint_vel = extract_joint_and_gripper_data(points_current)
        action_vector = generate_action_vector(points_current)

        rewards = np.array([1 if i == 59 else 0])
        dones = np.array([1 if i == 59 else 0])

        next_obs = {
            "object": position.tolist() + orientation.tolist() + velocity.tolist(),
            "robot0_eef_pos": eef_pos.tolist(),
            "robot0_eef_quat": eef_quat.tolist(),
            "robot0_eef_vel_ang": eef_vel_ang.tolist(),
            "robot0_eef_vel_lin": eef_vel_lin.tolist(),
            "robot0_gripper_qpos": gripper_qpos,
            "robot0_gripper_qvel": gripper_qvel.tolist(),
            "robot0_joint_pos": joint_pos.tolist(),
            "robot0_joint_pos_cos": joint_pos_cos.tolist(),
            "robot0_joint_pos_sin": joint_pos_sin.tolist(),
            "robot0_joint_vel": joint_vel.tolist(),
        }

        obs = next_obs.copy() if i > 0 else {key: [0] * len(value) for key, value in next_obs.items()}

        # demo_key = f"demo_{i}"
        demo_key = f"demo_{i}"
        demo_group = data_group.create_group(demo_key)

        # Store basic arrays
        demo_group.create_dataset('actions', data=action_vector)
        demo_group.create_dataset('dones', data=dones)
        demo_group.create_dataset('rewards', data=rewards)
        demo_group.create_dataset('states', data=np.array([]))

        # Create sub-groups for obs and next_obs
        obs_group = demo_group.create_group('obs')
        for key, value in obs.items():
            obs_group.create_dataset(key, data=value)

        next_obs_group = demo_group.create_group('next_obs')
        for key, value in next_obs.items():
            next_obs_group.create_dataset(key, data=value)

        results['data'][demo_key] = {
            "actions": action_vector.tolist(),
            "dones": dones.tolist(),
            "obs": obs,
            "next_obs": next_obs,
            "rewards": rewards.tolist(),
            "states": []
        }

    # Add the mask as an empty dictionary
    mask_group.attrs['mask'] = json.dumps({})

# Save results to a JSON file for inspection
with open("modified_combined_results.json", "w") as json_file:
    json.dump(results, json_file, indent=4)

print("Modified results saved to modified_combined_results.json and modified_results.hdf5")
