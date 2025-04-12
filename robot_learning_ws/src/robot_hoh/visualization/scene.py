import os
import cv2
import numpy as np
import open3d as o3d
import subprocess

# Define dataset path
# data_path = "data"
data_path  = os.path.abspath("data") 
folders = [str(i) for i in range(15)]  # ['0', '1', ..., '14']

output_dir = "visualization"
os.makedirs(output_dir, exist_ok=True)

def load_point_cloud(file_path, color):
    """Load a point cloud from a PLY file and assign a color."""
    if not os.path.exists(file_path):
        return None
    pcd = o3d.io.read_point_cloud(file_path)
    pcd.paint_uniform_color(color)  # Assign color for visualization
    return pcd

def generate_scene(folder_path):
    """Generate a list of scenes from point clouds in a given folder."""
    frames = []
    
    # Get sorted frame numbers
    frame_numbers = sorted(set(int(f.split("_frame")[1].split(".")[0]) for f in os.listdir(folder_path) if "frame" in f))

    for frame_num in frame_numbers:
        giver_pcd = load_point_cloud(os.path.join(folder_path, f"giver_frame{frame_num}.ply"), [1, 0, 0])  # Red
        receiver_pcd = load_point_cloud(os.path.join(folder_path, f"receiver_frame{frame_num}.ply"), [0, 1, 0])  # Green
        object_pcd = load_point_cloud(os.path.join(folder_path, f"object_frame{frame_num}.ply"), [0, 0, 1])  # Blue

        # Combine valid point clouds
        scene = [pcd for pcd in [giver_pcd, receiver_pcd, object_pcd] if pcd is not None]
        frames.append(scene)

    return frames


def save_scene_as_video(frames, folder_idx):
    """Render frames, save images, and create a video using ffmpeg."""
    image_folder = f"visualization/temp/temp_images_{folder_idx}"
    scene_folder = f"visualization/temp/scene_{folder_idx}"
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(scene_folder, exist_ok=True)

    image_files = []
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    for i, scene in enumerate(frames):
        vis.clear_geometries()
        for pcd in scene:
            vis.add_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        img_path = os.path.join(image_folder, f"frame_{i:04d}.png")
        vis.capture_screen_image(img_path)
        image_files.append(img_path)

        # print(f"Captured frame {i + 1}/{len(frames)}")

    vis.destroy_window()

    # Replace OpenCV VideoWriter with ffmpeg subprocess
    if image_files:
        video_path = os.path.join(scene_folder, f"scene_{folder_idx}.mp4")

        ffmpeg_command = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-framerate', '10',
            '-i', os.path.join(image_folder, 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            video_path
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f"Video created successfully with ffmpeg: {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed: {e}")

# Process each folder and generate videos
for folder in folders:
    folder_path = os.path.join(data_path, folder, "Cleaned")
    if os.path.exists(folder_path):
        print(f"Processing folder: {folder_path}")
        frames = generate_scene(folder_path)
        if frames:
            save_scene_as_video(frames, folder)
    else:
        print(f"Skipping missing folder: {folder_path}")
