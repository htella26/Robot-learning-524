env = {
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