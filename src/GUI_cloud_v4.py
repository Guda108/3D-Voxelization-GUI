# This script must be executed with Python 3.11 because of Open3D compatibility.

import os
import csv
import queue
import threading
import traceback
import numpy as np
import pandas as pd
import open3d as o3d
import customtkinter as ctk

from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ==========================================
# FILE LOADING
# ==========================================
def load_point_cloud_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"]:
        pcd = o3d.io.read_point_cloud(filepath)
        if pcd.is_empty():
            raise ValueError("The point cloud is empty or could not be loaded.")
        return pcd

    elif ext in [".csv", ".txt"]:
        try:
            if ext == ".csv":
                df = pd.read_csv(filepath)
            else:
                df = pd.read_csv(filepath, sep=None, engine="python")
        except Exception:
            df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")

        cols_lower = [str(c).strip().lower() for c in df.columns]

        if all(c in cols_lower for c in ["x", "y", "z"]):
            x_idx = cols_lower.index("x")
            y_idx = cols_lower.index("y")
            z_idx = cols_lower.index("z")
            pts = df.iloc[:, [x_idx, y_idx, z_idx]].to_numpy(dtype=float)
        else:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 3:
                raise ValueError("The file must contain x,y,z columns or at least 3 numeric columns.")
            pts = numeric_df.iloc[:, :3].to_numpy(dtype=float)

        if pts.shape[0] == 0:
            raise ValueError("No valid points were found.")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    else:
        raise ValueError(f"Unsupported format: {ext}")


# ==========================================
# EXPORT
# ==========================================
def ensure_dir_for_file(filepath):
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


def save_voxel_centers_csv(points, voxel_size, filepath):
    ensure_dir_for_file(filepath)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "voxel_size"])
        for p in points:
            writer.writerow([p[0], p[1], p[2], voxel_size])


def save_voxel_centers_txt(points, voxel_size, filepath):
    ensure_dir_for_file(filepath)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# x y z voxel_size\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {voxel_size}\n")


def save_voxel_centers_npy(points, voxel_size, filepath):
    ensure_dir_for_file(filepath)
    data = {
        "centers": np.asarray(points, dtype=float),
        "voxel_size": float(voxel_size),
        "count": int(len(points))
    }
    np.save(filepath, data, allow_pickle=True)


# ==========================================
# VOXELIZATION
# ==========================================
def get_voxel_centers_from_grid(voxel_grid):
    voxels = voxel_grid.get_voxels()
    origin = np.asarray(voxel_grid.origin, dtype=float)
    voxel_size = float(voxel_grid.voxel_size)

    centers = []
    for v in voxels:
        center = origin + (np.array(v.grid_index, dtype=float) + 0.5) * voxel_size
        centers.append(center)

    if len(centers) == 0:
        return np.empty((0, 3), dtype=float)

    return np.asarray(centers, dtype=float)


def process_point_cloud_to_voxels(
    pcd,
    voxel_size,
    downsample_first=False,
    downsample_size=None,
    remove_outliers=False,
    nb_neighbors=20,
    std_ratio=2.0
):
    working_pcd = pcd

    if downsample_first and downsample_size is not None and downsample_size > 0:
        working_pcd = working_pcd.voxel_down_sample(downsample_size)

    if remove_outliers:
        working_pcd, _ = working_pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        working_pcd,
        voxel_size=float(voxel_size)
    )

    centers = get_voxel_centers_from_grid(voxel_grid)
    return working_pcd, voxel_grid, centers


# ==========================================
# GUI
# ==========================================
class PointCloudVoxelGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Point Cloud to Voxels")
        self.geometry("900x650")
        self.minsize(900, 650)
        self.resizable(True, True)

        self.worker_queue = queue.Queue()
        self.worker_thread = None

        self.loaded_pcd = None
        self.cleaned_pcd = None
        self.loaded_voxel_grid = None
        self.loaded_voxel_centers = None

        self.input_file_var = ctk.StringVar()
        self.voxel_size_var = ctk.StringVar(value="0.05")
        self.downsample_size_var = ctk.StringVar(value="0.03")
        self.nb_neighbors_var = ctk.StringVar(value="20")
        self.std_ratio_var = ctk.StringVar(value="2.0")

        self.use_downsample_var = ctk.BooleanVar(value=False)
        self.use_outlier_removal_var = ctk.BooleanVar(value=False)

        self.export_csv_var = ctk.BooleanVar(value=True)
        self.export_npy_var = ctk.BooleanVar(value=True)
        self.export_txt_var = ctk.BooleanVar(value=False)

        self.csv_path_var = ctk.StringVar()
        self.npy_path_var = ctk.StringVar()
        self.txt_path_var = ctk.StringVar()

        self.build_ui()
        self.after(100, self.poll_worker_queue)

    def build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =========================
        # LEFT PANEL
        # =========================
        left_panel = ctk.CTkScrollableFrame(self, width=280, corner_radius=10)
        left_panel.grid(row=0, column=0, sticky="nsw", padx=(10, 6), pady=10)
        left_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_panel,
            text="Controls",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 10))

        # File
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        file_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(file_frame, text="File").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ctk.CTkEntry(file_frame, textvariable=self.input_file_var, height=30).grid(
            row=1, column=0, sticky="ew", padx=8, pady=4
        )
        ctk.CTkButton(file_frame, text="Browse", height=30, command=self.select_input_file).grid(
            row=2, column=0, sticky="ew", padx=8, pady=(4, 8)
        )

        # Parameters
        param_frame = ctk.CTkFrame(left_panel)
        param_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        param_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(param_frame, text="Parameters").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 6)
        )

        ctk.CTkLabel(param_frame, text="Voxel size").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkEntry(param_frame, textvariable=self.voxel_size_var, width=90, height=28).grid(
            row=1, column=1, sticky="w", padx=8, pady=4
        )

        ctk.CTkCheckBox(param_frame, text="Downsample", variable=self.use_downsample_var).grid(
            row=2, column=0, sticky="w", padx=8, pady=4
        )
        ctk.CTkEntry(param_frame, textvariable=self.downsample_size_var, width=90, height=28).grid(
            row=2, column=1, sticky="w", padx=8, pady=4
        )

        ctk.CTkCheckBox(param_frame, text="Remove outliers", variable=self.use_outlier_removal_var).grid(
            row=3, column=0, sticky="w", padx=8, pady=4
        )

        ctk.CTkLabel(param_frame, text="Neighbors").grid(row=4, column=0, sticky="w", padx=8, pady=4)
        ctk.CTkEntry(param_frame, textvariable=self.nb_neighbors_var, width=90, height=28).grid(
            row=4, column=1, sticky="w", padx=8, pady=4
        )

        ctk.CTkLabel(param_frame, text="Std ratio").grid(row=5, column=0, sticky="w", padx=8, pady=(4, 8))
        ctk.CTkEntry(param_frame, textvariable=self.std_ratio_var, width=90, height=28).grid(
            row=5, column=1, sticky="w", padx=8, pady=(4, 8)
        )

        # Export
        export_frame = ctk.CTkFrame(left_panel)
        export_frame.grid(row=3, column=0, sticky="ew", padx=4, pady=4)
        export_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(export_frame, text="Export").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 6)
        )

        ctk.CTkCheckBox(export_frame, text="CSV", variable=self.export_csv_var).grid(
            row=1, column=0, sticky="w", padx=8, pady=2
        )
        ctk.CTkButton(export_frame, text="CSV Path", height=28, command=self.pick_csv).grid(
            row=2, column=0, sticky="ew", padx=8, pady=2
        )

        ctk.CTkCheckBox(export_frame, text="NPY", variable=self.export_npy_var).grid(
            row=3, column=0, sticky="w", padx=8, pady=2
        )
        ctk.CTkButton(export_frame, text="NPY Path", height=28, command=self.pick_npy).grid(
            row=4, column=0, sticky="ew", padx=8, pady=2
        )

        ctk.CTkCheckBox(export_frame, text="TXT", variable=self.export_txt_var).grid(
            row=5, column=0, sticky="w", padx=8, pady=2
        )
        ctk.CTkButton(export_frame, text="TXT Path", height=28, command=self.pick_txt).grid(
            row=6, column=0, sticky="ew", padx=8, pady=(2, 8)
        )

        # Main action
        action_frame = ctk.CTkFrame(left_panel)
        action_frame.grid(row=4, column=0, sticky="ew", padx=4, pady=4)
        action_frame.grid_columnconfigure(0, weight=1)

        self.process_button = ctk.CTkButton(
            action_frame,
            text="Process to Voxels",
            height=36,
            command=self.start_processing
        )
        self.process_button.grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        # View
        view_frame = ctk.CTkFrame(left_panel)
        view_frame.grid(row=5, column=0, sticky="ew", padx=4, pady=4)
        view_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(view_frame, text="View").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 6)
        )

        ctk.CTkButton(
            view_frame,
            text="Original",
            height=30,
            command=lambda: self.update_embedded_plot("original")
        ).grid(row=1, column=0, sticky="ew", padx=6, pady=4)

        ctk.CTkButton(
            view_frame,
            text="Processed",
            height=30,
            command=lambda: self.update_embedded_plot("processed")
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ctk.CTkButton(
            view_frame,
            text="Voxels",
            height=30,
            command=lambda: self.update_embedded_plot("voxels")
        ).grid(row=2, column=0, sticky="ew", padx=6, pady=(4, 8))

        ctk.CTkButton(
            view_frame,
            text="Clear",
            height=30,
            command=self.clear_memory
        ).grid(row=2, column=1, sticky="ew", padx=6, pady=(4, 8))

        # Progress
        progress_frame = ctk.CTkFrame(left_panel)
        progress_frame.grid(row=6, column=0, sticky="ew", padx=4, pady=4)
        progress_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(progress_frame, text="Progress").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(
            progress_frame,
            text="No active process.",
            anchor="w",
            justify="left"
        )
        self.progress_label.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        # =========================
        # RIGHT PANEL
        # =========================
        right_panel = ctk.CTkFrame(self, corner_radius=10)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 10), pady=10)
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_rowconfigure(2, weight=0)

        ctk.CTkLabel(
            right_panel,
            text="3D Preview",
            font=ctk.CTkFont(size=18, weight="bold")
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 6))

        plot_frame = ctk.CTkFrame(right_panel)
        plot_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(1, weight=0)
        plot_frame.grid_rowconfigure(2, weight=0)
        plot_frame.grid_columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(5.2, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("No data")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Matplotlib toolbar: Home, Back, Forward, Pan, Zoom, Save.
        toolbar_frame = ctk.CTkFrame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(4, 0))
        toolbar_frame.grid_columnconfigure(0, weight=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x", expand=True)

        # Rotate mode button
        self.rotate_button = ctk.CTkButton(
            plot_frame,
            text="Rotate Mode",
            height=28,
            command=self.enable_rotate_mode
        )
        self.rotate_button.grid(row=2, column=0, sticky="ew", padx=0, pady=(4, 0))

        # Compact log
        log_frame = ctk.CTkFrame(right_panel)
        log_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        log_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(log_frame, text="Output").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )

        self.logbox = ctk.CTkTextbox(log_frame, height=110)
        self.logbox.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.logbox.insert("end", "Application started.\n")
        self.logbox.configure(state="disabled")

    def log(self, text):
        self.logbox.configure(state="normal")
        self.logbox.insert("end", text + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def set_progress(self, value, text):
        self.progress_bar.set(value)
        self.progress_label.configure(text=text)

    def enable_rotate_mode(self):
        """
        Disable toolbar modes such as Pan or Zoom.
        After this, left mouse drag can rotate the 3D plot again.
        """
        if hasattr(self, "toolbar"):
            mode = str(self.toolbar.mode).lower()

            if "pan" in mode:
                self.toolbar.pan()
            elif "zoom" in mode:
                self.toolbar.zoom()

        self.log("Rotate mode enabled. Drag with the left mouse button over the preview to rotate.")

    def select_input_file(self):
        path = filedialog.askopenfilename(
            title="Select point cloud",
            filetypes=[
                ("Point Clouds", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts *.csv *.txt"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        self.input_file_var.set(path)
        base = os.path.splitext(path)[0]
        self.csv_path_var.set(base + "_voxels.csv")
        self.npy_path_var.set(base + "_voxels.npy")
        self.txt_path_var.set(base + "_voxels.txt")
        self.log(f"Selected file: {path}")

    def pick_csv(self):
        path = filedialog.asksaveasfilename(
            title="Select CSV output path",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if path:
            self.csv_path_var.set(path)

    def pick_npy(self):
        path = filedialog.asksaveasfilename(
            title="Select NPY output path",
            defaultextension=".npy",
            filetypes=[("NPY", "*.npy")]
        )
        if path:
            self.npy_path_var.set(path)

    def pick_txt(self):
        path = filedialog.asksaveasfilename(
            title="Select TXT output path",
            defaultextension=".txt",
            filetypes=[("TXT", "*.txt")]
        )
        if path:
            self.txt_path_var.set(path)

    def clear_memory(self):
        self.loaded_pcd = None
        self.cleaned_pcd = None
        self.loaded_voxel_grid = None
        self.loaded_voxel_centers = None

        self.ax.clear()
        self.ax.set_title("No data")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()

        self.set_progress(0, "No active process.")
        self.log("Data in memory has been cleared.")

    def validate_inputs(self):
        filepath = self.input_file_var.get().strip()
        if not filepath:
            raise ValueError("Please select a file.")
        if not os.path.isfile(filepath):
            raise ValueError("The selected file does not exist.")

        voxel_size = float(self.voxel_size_var.get())
        downsample_size = float(self.downsample_size_var.get())
        nb_neighbors = int(self.nb_neighbors_var.get())
        std_ratio = float(self.std_ratio_var.get())

        if voxel_size <= 0:
            raise ValueError("voxel_size must be greater than zero.")
        if downsample_size <= 0:
            raise ValueError("downsample_size must be greater than zero.")
        if nb_neighbors <= 0:
            raise ValueError("nb_neighbors must be greater than zero.")
        if std_ratio <= 0:
            raise ValueError("std_ratio must be greater than zero.")

        return {
            "filepath": filepath,
            "voxel_size": voxel_size,
            "use_downsample": self.use_downsample_var.get(),
            "downsample_size": downsample_size,
            "use_outlier_removal": self.use_outlier_removal_var.get(),
            "nb_neighbors": nb_neighbors,
            "std_ratio": std_ratio,
            "export_csv": self.export_csv_var.get(),
            "export_npy": self.export_npy_var.get(),
            "export_txt": self.export_txt_var.get(),
            "csv_path": self.csv_path_var.get().strip(),
            "npy_path": self.npy_path_var.get().strip(),
            "txt_path": self.txt_path_var.get().strip(),
        }

    def start_processing(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.log("A process is already running.")
            return

        try:
            cfg = self.validate_inputs()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e))
            return

        self.process_button.configure(state="disabled")
        self.set_progress(0.02, "Preparing process...")
        self.log("Starting processing...")

        self.worker_thread = threading.Thread(
            target=self.processing_worker,
            args=(cfg,),
            daemon=True
        )
        self.worker_thread.start()

    def processing_worker(self, cfg):
        try:
            self.worker_queue.put(("progress", 0.10, "Loading point cloud..."))
            pcd = load_point_cloud_file(cfg["filepath"])
            pts = np.asarray(pcd.points)

            if len(pts) == 0:
                raise ValueError("The loaded point cloud is empty.")

            self.worker_queue.put(("log", f"Loaded points: {len(pts)}"))

            self.worker_queue.put(("progress", 0.35, "Applying processing..."))
            processed_pcd, voxel_grid, centers = process_point_cloud_to_voxels(
                pcd=pcd,
                voxel_size=cfg["voxel_size"],
                downsample_first=cfg["use_downsample"],
                downsample_size=cfg["downsample_size"],
                remove_outliers=cfg["use_outlier_removal"],
                nb_neighbors=cfg["nb_neighbors"],
                std_ratio=cfg["std_ratio"]
            )

            self.loaded_pcd = pcd
            self.cleaned_pcd = processed_pcd
            self.loaded_voxel_grid = voxel_grid
            self.loaded_voxel_centers = centers

            self.worker_queue.put(("log", f"Processed points: {len(np.asarray(processed_pcd.points))}"))
            self.worker_queue.put(("log", f"Occupied voxels: {len(centers)}"))

            self.worker_queue.put(("progress", 0.70, "Exporting results..."))

            if cfg["export_csv"] and cfg["csv_path"]:
                save_voxel_centers_csv(centers, cfg["voxel_size"], cfg["csv_path"])
                self.worker_queue.put(("log", f"CSV saved: {cfg['csv_path']}"))

            if cfg["export_npy"] and cfg["npy_path"]:
                save_voxel_centers_npy(centers, cfg["voxel_size"], cfg["npy_path"])
                self.worker_queue.put(("log", f"NPY saved: {cfg['npy_path']}"))

            if cfg["export_txt"] and cfg["txt_path"]:
                save_voxel_centers_txt(centers, cfg["voxel_size"], cfg["txt_path"])
                self.worker_queue.put(("log", f"TXT saved: {cfg['txt_path']}"))

            self.worker_queue.put(("progress", 1.0, "Process completed."))
            self.worker_queue.put(("done_success",))

        except Exception as e:
            self.worker_queue.put(("log", "ERROR: " + str(e)))
            self.worker_queue.put(("log", traceback.format_exc()))
            self.worker_queue.put(("progress", 0.0, "Process error."))
            self.worker_queue.put(("done",))

    def poll_worker_queue(self):
        try:
            while True:
                item = self.worker_queue.get_nowait()
                kind = item[0]

                if kind == "log":
                    self.log(item[1])

                elif kind == "progress":
                    _, value, text = item
                    self.set_progress(value, text)

                elif kind == "done_success":
                    self.process_button.configure(state="normal")
                    self.log("Process completed successfully.")
                    self.update_embedded_plot("voxels")

                elif kind == "done":
                    self.process_button.configure(state="normal")

        except queue.Empty:
            pass

        self.after(100, self.poll_worker_queue)

    def update_embedded_plot(self, mode="original"):
        self.ax.clear()

        if mode == "original":
            if self.loaded_pcd is None:
                self.ax.set_title("No original point cloud loaded")
                self.canvas.draw()
                return

            pts = np.asarray(self.loaded_pcd.points)
            self._scatter_points(pts, title="Original Point Cloud", marker="o")

        elif mode == "processed":
            if self.cleaned_pcd is None:
                self.ax.set_title("No processed point cloud available")
                self.canvas.draw()
                return

            pts = np.asarray(self.cleaned_pcd.points)
            self._scatter_points(pts, title="Processed Point Cloud", marker="o")

        elif mode == "voxels":
            if self.loaded_voxel_centers is None or len(self.loaded_voxel_centers) == 0:
                self.ax.set_title("No generated voxels available")
                self.canvas.draw()
                return

            pts = np.asarray(self.loaded_voxel_centers)
            self._scatter_points(pts, title="Voxel Centers", marker="s", point_size=10)

        self.canvas.draw()

    def _scatter_points(self, pts, title="3D View", marker="o", point_size=None):
        if pts.shape[0] == 0:
            self.ax.set_title("No points")
            return

        max_points = 25000
        if pts.shape[0] > max_points:
            idx = np.random.choice(pts.shape[0], max_points, replace=False)
            pts_plot = pts[idx]
            self.log(f"Visualization reduced to {max_points} points for better performance.")
        else:
            pts_plot = pts

        x = pts_plot[:, 0]
        y = pts_plot[:, 1]
        z = pts_plot[:, 2]

        if point_size is None:
            point_size = 2 if marker == "o" else 8

        self.ax.scatter(x, y, z, c=z, s=point_size, marker=marker, depthshade=True)

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self._set_equal_axes(x, y, z)

    def _set_equal_axes(self, x, y, z):
        x_mid = (np.max(x) + np.min(x)) / 2
        y_mid = (np.max(y) + np.min(y)) / 2
        z_mid = (np.max(z) + np.min(z)) / 2

        max_range = max(
            np.max(x) - np.min(x),
            np.max(y) - np.min(y),
            np.max(z) - np.min(z)
        ) / 2

        if max_range == 0:
            max_range = 1.0

        self.ax.set_xlim(x_mid - max_range, x_mid + max_range)
        self.ax.set_ylim(y_mid - max_range, y_mid + max_range)
        self.ax.set_zlim(z_mid - max_range, z_mid + max_range)


if __name__ == "__main__":
    app = PointCloudVoxelGUI()
    app.mainloop()