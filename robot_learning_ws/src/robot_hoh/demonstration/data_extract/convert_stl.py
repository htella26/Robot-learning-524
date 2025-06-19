from PIL import Image
import numpy as np
import trimesh
import os

# Path to your image and output STL file
image_path = "../assets/basket_02.png"
output_path = "../assets/basket_02.stl"

# Load image and preprocess
img = Image.open(image_path).convert("L")
img = img.resize((128, 128))  # Smaller grid for performance
img_array = np.array(img) / 255.0  # Normalize to 0-1

# Create a heightmap surface mesh
height_scale = 0.02  # 2cm max height
X, Y = np.meshgrid(np.linspace(-0.04, 0.04, img_array.shape[1]),
                   np.linspace(-0.04, 0.04, img_array.shape[0]))
Z = img_array * height_scale

# Build vertices and faces
vertices = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
faces = []
rows, cols = img_array.shape
for i in range(rows - 1):
    for j in range(cols - 1):
        idx = i * cols + j
        faces.append([idx, idx + 1, idx + cols])
        faces.append([idx + 1, idx + cols + 1, idx + cols])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Export to STL
mesh.export(output_path)
print(f"STL exported to: {output_path}")
