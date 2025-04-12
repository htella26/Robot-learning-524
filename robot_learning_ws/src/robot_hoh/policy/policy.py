# -*- coding: utf-8 -*-
"""run_policy.py

Run a trained policy and visualize the rollout.
"""

import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy
import torch
import urllib.request

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

# --------------------------
# Download policy checkpoint
# --------------------------
ckpt_path = "lift_ph_low_dim_epoch_1000_succ_100.pth"
urllib.request.urlretrieve(
    "http://downloads.cs.stanford.edu/downloads/rt_benchmark/model_zoo/lift/bc_rnn/lift_ph_low_dim_epoch_1000_succ_100.pth",
    filename=ckpt_path
)
assert os.path.exists(ckpt_path)

# --------------------------
# Load trained policy
# --------------------------
device = TorchUtils.get_torch_device(try_to_use_cuda=True)
policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt_path, device=device, verbose=True)

# --------------------------
# Create rollout environment
# --------------------------
env, _ = FileUtils.env_from_checkpoint(
    ckpt_dict=ckpt_dict,
    render=False,
    render_offscreen=True,
    verbose=True,
)

# --------------------------
# Rollout function
# --------------------------
def rollout(policy, env, horizon, render=False, video_writer=None, video_skip=5, camera_names=None):
    policy.start_episode()
    obs = env.reset_to(env.get_state())
    total_reward = 0.
    success = False

    for step_i in range(horizon):
        act = policy(ob=obs)
        next_obs, r, done, _ = env.step(act)
        total_reward += r
        success = env.is_success()["task"]

        if render:
            env.render(mode="human", camera_name=camera_names[0])

        if video_writer and step_i % video_skip == 0:
            frames = [
                env.render(mode="rgb_array", height=512, width=512, camera_name=cam)
                for cam in camera_names
            ]
            video_writer.append_data(np.concatenate(frames, axis=1))

        if done or success:
            break
        obs = deepcopy(next_obs)

    return {"Return": total_reward, "Horizon": step_i + 1, "Success_Rate": float(success)}

# --------------------------
# Run policy and save video to project folder
# --------------------------
output_folder = "/home/hambal/robot-learning-practice/videos"
os.makedirs(output_folder, exist_ok=True)
video_path = os.path.join(output_folder, "rollout.mp4")
video_writer = imageio.get_writer(video_path, fps=20)

stats = rollout(
    policy=policy,
    env=env,
    horizon=400,
    render=False,
    video_writer=video_writer,
    video_skip=5,
    camera_names=["agentview"]
)

video_writer.close()
print(f"Rollout video saved to {video_path}")
print("Rollout statistics:", stats)
