import os
import sys


def get_project_root(current_file_path):
	current_dir = os.path.dirname(os.path.abspath(current_file_path))
	project_root = os.path.dirname(current_dir)
	if project_root not in sys.path:
		sys.path.append(project_root)
	return project_root


def get_models_dir(project_root):
	return os.path.join(project_root, "All_3D_Models")


def get_stl_files(models_dir):
	if not os.path.exists(models_dir):
		raise FileNotFoundError(f"Directory not found: {models_dir}")

	stl_files = [f for f in os.listdir(models_dir) if f.lower().endswith(".stl")]
	if not stl_files:
		raise FileNotFoundError("No .stl files found in the All_3D_Models directory.")

	return stl_files
