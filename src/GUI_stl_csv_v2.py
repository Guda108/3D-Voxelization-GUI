# This script processes STL files into voxel centers and exports them to CSV, NPY, or TXT.
# Recommended Python version: Python 3.11

import os
import csv
import queue
import threading
import traceback
import numpy as np
import trimesh
import customtkinter as ctk

from tkinter import filedialog, messagebox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ==========================================
# UTILITIES
# ==========================================
def safe_float(value, field_name="value"):
    try:
        return float(value)
    except ValueError:
        raise ValueError(f"{field_name} must be numeric.")


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def load_stl_mesh(filepath):
    mesh = trimesh.load_mesh(filepath)

    if mesh.is_empty:
        raise ValueError("The loaded mesh is empty.")

    return mesh


def centers_from_voxelgrid(voxel_grid):
    points = voxel_grid.points

    if points is None:
        return np.empty((0, 3), dtype=float)

    return np.asarray(points, dtype=float)


def save_csv(centers, voxel_size, filepath):
    ensure_parent_dir(filepath)

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "voxel_size"])

        for p in centers:
            writer.writerow([p[0], p[1], p[2], voxel_size])


def save_npy(centers, voxel_size, filepath):
    ensure_parent_dir(filepath)

    data = {
        "centers": np.asarray(centers, dtype=float),
        "voxel_size": float(voxel_size),
        "count": int(len(centers)),
    }

    np.save(filepath, data, allow_pickle=True)


def save_txt(centers, voxel_size, filepath):
    ensure_parent_dir(filepath)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# x y z voxel_size\n")

        for p in centers:
            f.write(f"{p[0]} {p[1]} {p[2]} {voxel_size}\n")


def process_stl_to_voxels(mesh, voxel_size, fill_interior=True):
    voxel_grid = mesh.voxelized(pitch=float(voxel_size))

    if fill_interior:
        try:
            voxel_grid = voxel_grid.fill()
        except Exception:
            pass

    centers = centers_from_voxelgrid(voxel_grid)

    return voxel_grid, centers


# ==========================================
# GUI
# ==========================================
class STLVoxelGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("STL to Voxels")
        self.geometry("900x650")
        self.minsize(900, 650)
        self.resizable(True, True)

        self.worker_queue = queue.Queue()
        self.worker_thread = None

        self.loaded_mesh = None
        self.loaded_voxel_grid = None
        self.loaded_centers = None

        self.input_file_var = ctk.StringVar()
        self.voxel_size_var = ctk.StringVar(value="3.0")

        self.fill_interior_var = ctk.BooleanVar(value=True)

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

        ctk.CTkLabel(file_frame, text="STL File").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 4)
        )

        ctk.CTkEntry(
            file_frame,
            textvariable=self.input_file_var,
            height=30
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=4)

        ctk.CTkButton(
            file_frame,
            text="Browse",
            height=30,
            command=self.select_input_file
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))

        # Parameters
        param_frame = ctk.CTkFrame(left_panel)
        param_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        param_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            param_frame,
            text="Parameters"
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 6))

        ctk.CTkLabel(param_frame, text="Voxel size").grid(
            row=1, column=0, sticky="w", padx=8, pady=4
        )

        ctk.CTkEntry(
            param_frame,
            textvariable=self.voxel_size_var,
            width=90,
            height=28
        ).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ctk.CTkCheckBox(
            param_frame,
            text="Fill interior",
            variable=self.fill_interior_var
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 8))

        # Export
        export_frame = ctk.CTkFrame(left_panel)
        export_frame.grid(row=3, column=0, sticky="ew", padx=4, pady=4)
        export_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(export_frame, text="Export").grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 6)
        )

        ctk.CTkCheckBox(
            export_frame,
            text="CSV",
            variable=self.export_csv_var
        ).grid(row=1, column=0, sticky="w", padx=8, pady=2)

        ctk.CTkButton(
            export_frame,
            text="CSV Path",
            height=28,
            command=self.pick_csv
        ).grid(row=2, column=0, sticky="ew", padx=8, pady=2)

        ctk.CTkCheckBox(
            export_frame,
            text="NPY",
            variable=self.export_npy_var
        ).grid(row=3, column=0, sticky="w", padx=8, pady=2)

        ctk.CTkButton(
            export_frame,
            text="NPY Path",
            height=28,
            command=self.pick_npy
        ).grid(row=4, column=0, sticky="ew", padx=8, pady=2)

        ctk.CTkCheckBox(
            export_frame,
            text="TXT",
            variable=self.export_txt_var
        ).grid(row=5, column=0, sticky="w", padx=8, pady=2)

        ctk.CTkButton(
            export_frame,
            text="TXT Path",
            height=28,
            command=self.pick_txt
        ).grid(row=6, column=0, sticky="ew", padx=8, pady=(2, 8))

        # Main action
        action_frame = ctk.CTkFrame(left_panel)
        action_frame.grid(row=4, column=0, sticky="ew", padx=4, pady=4)
        action_frame.grid_columnconfigure(0, weight=1)

        self.process_button = ctk.CTkButton(
            action_frame,
            text="Process STL to Voxels",
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
            text="Original STL",
            height=30,
            command=lambda: self.update_embedded_plot("mesh")
        ).grid(row=1, column=0, sticky="ew", padx=6, pady=4)

        ctk.CTkButton(
            view_frame,
            text="Voxels",
            height=30,
            command=lambda: self.update_embedded_plot("voxels")
        ).grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ctk.CTkButton(
            view_frame,
            text="Clear",
            height=30,
            command=self.clear_memory
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(4, 8))

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

        # Matplotlib navigation toolbar:
        # Home, Back, Forward, Pan, Zoom, Configure subplots, and Save.
        toolbar_frame = ctk.CTkFrame(plot_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew", padx=0, pady=(4, 0))
        toolbar_frame.grid_columnconfigure(0, weight=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x", expand=True)

        # Rotate mode button:
        # This disables Pan/Zoom from the toolbar so the mouse can rotate the 3D view again.
        self.rotate_button = ctk.CTkButton(
            plot_frame,
            text="Rotate Mode",
            height=28,
            command=self.enable_rotate_mode
        )
        self.rotate_button.grid(row=2, column=0, sticky="ew", padx=0, pady=(4, 0))

        # Log
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

    # ==========================================
    # UI HELPERS
    # ==========================================
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
        After this, the left mouse button can rotate the 3D plot again.
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
            title="Select STL file",
            filetypes=[
                ("STL files", "*.stl"),
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
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.csv_path_var.set(path)

    def pick_npy(self):
        path = filedialog.asksaveasfilename(
            title="Select NPY output path",
            defaultextension=".npy",
            filetypes=[
                ("NumPy files", "*.npy"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.npy_path_var.set(path)

    def pick_txt(self):
        path = filedialog.asksaveasfilename(
            title="Select TXT output path",
            defaultextension=".txt",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if path:
            self.txt_path_var.set(path)

    def clear_memory(self):
        self.loaded_mesh = None
        self.loaded_voxel_grid = None
        self.loaded_centers = None

        self.ax.clear()
        self.ax.set_title("No data")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.canvas.draw()

        self.set_progress(0, "No active process.")
        self.log("Data in memory has been cleared.")

    def validate_inputs(self):
        stl_path = self.input_file_var.get().strip()

        if not stl_path:
            raise ValueError("Please select an STL file.")

        if not os.path.isfile(stl_path):
            raise ValueError("The selected STL file does not exist.")

        voxel_size = safe_float(self.voxel_size_var.get().strip(), "Voxel size")

        if voxel_size <= 0:
            raise ValueError("Voxel size must be greater than zero.")

        export_csv = self.export_csv_var.get()
        export_npy = self.export_npy_var.get()
        export_txt = self.export_txt_var.get()

        if export_csv and not self.csv_path_var.get().strip():
            raise ValueError("Please select a CSV output path.")

        if export_npy and not self.npy_path_var.get().strip():
            raise ValueError("Please select an NPY output path.")

        if export_txt and not self.txt_path_var.get().strip():
            raise ValueError("Please select a TXT output path.")

        return {
            "stl_path": stl_path,
            "voxel_size": voxel_size,
            "fill_interior": self.fill_interior_var.get(),
            "export_csv": export_csv,
            "export_npy": export_npy,
            "export_txt": export_txt,
            "csv_path": self.csv_path_var.get().strip(),
            "npy_path": self.npy_path_var.get().strip(),
            "txt_path": self.txt_path_var.get().strip(),
        }

    # ==========================================
    # PROCESSING
    # ==========================================
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
        self.log("Starting STL processing...")

        self.worker_thread = threading.Thread(
            target=self.processing_worker,
            args=(cfg,),
            daemon=True
        )
        self.worker_thread.start()

    def processing_worker(self, cfg):
        try:
            self.worker_queue.put(("progress", 0.10, "Loading STL mesh..."))

            mesh = load_stl_mesh(cfg["stl_path"])

            self.worker_queue.put(("log", f"Mesh loaded: {cfg['stl_path']}"))
            self.worker_queue.put(("log", f"Vertices: {len(mesh.vertices)}"))
            self.worker_queue.put(("log", f"Faces: {len(mesh.faces)}"))
            self.worker_queue.put(("log", f"Bounds min: {mesh.bounds[0]}"))
            self.worker_queue.put(("log", f"Bounds max: {mesh.bounds[1]}"))

            self.worker_queue.put(("progress", 0.40, "Voxelizing mesh..."))

            voxel_grid, centers = process_stl_to_voxels(
                mesh=mesh,
                voxel_size=cfg["voxel_size"],
                fill_interior=cfg["fill_interior"]
            )

            self.loaded_mesh = mesh
            self.loaded_voxel_grid = voxel_grid
            self.loaded_centers = centers

            self.worker_queue.put(("log", f"Occupied voxels: {len(centers)}"))

            self.worker_queue.put(("progress", 0.70, "Exporting results..."))

            if cfg["export_csv"]:
                save_csv(centers, cfg["voxel_size"], cfg["csv_path"])
                self.worker_queue.put(("log", f"CSV saved: {cfg['csv_path']}"))

            if cfg["export_npy"]:
                save_npy(centers, cfg["voxel_size"], cfg["npy_path"])
                self.worker_queue.put(("log", f"NPY saved: {cfg['npy_path']}"))

            if cfg["export_txt"]:
                save_txt(centers, cfg["voxel_size"], cfg["txt_path"])
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

    # ==========================================
    # VISUALIZATION
    # ==========================================
    def update_embedded_plot(self, mode="mesh"):
        self.ax.clear()

        if mode == "mesh":
            if self.loaded_mesh is None:
                stl_path = self.input_file_var.get().strip()

                if not stl_path:
                    self.ax.set_title("No STL file selected")
                    self.canvas.draw()
                    return

                try:
                    self.loaded_mesh = load_stl_mesh(stl_path)
                except Exception as e:
                    self.ax.set_title("Could not load STL")
                    self.log(f"Error loading STL: {e}")
                    self.canvas.draw()
                    return

            self._plot_mesh(self.loaded_mesh)

        elif mode == "voxels":
            if self.loaded_centers is None or len(self.loaded_centers) == 0:
                self.ax.set_title("No generated voxels available")
                self.canvas.draw()
                return

            self._scatter_points(
                self.loaded_centers,
                title="Voxel Centers",
                marker="s",
                point_size=10
            )

        self.canvas.draw()

    def _plot_mesh(self, mesh):
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)

        if vertices.shape[0] == 0 or faces.shape[0] == 0:
            self.ax.set_title("Empty mesh")
            return

        max_faces = 8000

        if faces.shape[0] > max_faces:
            idx = np.random.choice(faces.shape[0], max_faces, replace=False)
            faces_plot = faces[idx]
            self.log(f"Mesh visualization reduced to {max_faces} faces for better performance.")
        else:
            faces_plot = faces

        self.ax.plot_trisurf(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2],
            triangles=faces_plot,
            linewidth=0.2,
            alpha=0.85
        )

        self.ax.set_title("Original STL Mesh")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self._set_equal_axes(
            vertices[:, 0],
            vertices[:, 1],
            vertices[:, 2]
        )

    def _scatter_points(self, pts, title="3D View", marker="o", point_size=None):
        pts = np.asarray(pts, dtype=float)

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

        self.ax.scatter(
            x,
            y,
            z,
            c=z,
            s=point_size,
            marker=marker,
            depthshade=True
        )

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self._set_equal_axes(x, y, z)

    def _set_equal_axes(self, x, y, z):
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

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
    app = STLVoxelGUI()
    app.mainloop()