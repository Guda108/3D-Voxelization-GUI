# 3D Voxelization GUI

Graphical Python tools for converting 3D models and point clouds into voxel center coordinates.  
The project includes two desktop applications:

1. **STL to Voxels GUI**: converts `.stl` 3D meshes into voxel centers.
2. **Point Cloud to Voxels GUI**: converts point cloud files such as `.ply`, `.pcd`, `.xyz`, `.csv`, and `.txt` into voxel centers.

The generated voxel data can be exported as:

- `.csv`
- `.npy`
- `.txt`

---

## Features

- Graphical user interface built with `customtkinter`.
- 3D preview using `matplotlib`.
- STL mesh voxelization using `trimesh`.
- Point cloud processing using `open3d`.
- Export of voxel centers with coordinates `x`, `y`, `z`, and `voxel_size`.
- Optional point cloud downsampling.
- Optional statistical outlier removal for point clouds.
- Option to fill the interior of STL voxelized models.
- Support for standalone executable generation using PyInstaller.

---

## Repository structure

```text
3D-Voxelization-GUI/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── GUI_cloud_v4.py
│   └── GUI_stl_csv_v2.py
│
├── examples/
│   ├── Castle_Full.stl
│   ├── output_voxels_castle.csv
│   └── Castle_Full_voxels.npy
│
└── build/
    └── pyinstaller_commands.txt
	
	
Requirements

Recommended Python version:

Python 3.11