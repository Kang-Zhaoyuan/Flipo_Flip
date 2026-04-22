import os


def _is_likely_ascii_stl(stl_path, probe_bytes=512):
	"""Heuristic check for ASCII STL by probing file header text."""
	try:
		with open(stl_path, "rb") as f:
			header = f.read(probe_bytes)
	except OSError:
		return False

	header_l = header.lower()
	return header_l.startswith(b"solid") and (b"facet" in header_l)


def ensure_mujoco_compatible_stl(stl_path):
	"""Convert ASCII STL to binary STL for MuJoCo if needed.

	Returns:
		tuple: (mesh_path_to_use, converted_flag)
	"""
	if not _is_likely_ascii_stl(stl_path):
		return stl_path, False

	try:
		from stl import Mode, mesh
	except ImportError:
		print("[WARN] Selected STL looks like ASCII. MuJoCo may fail to decode it.")
		print("[WARN] Install numpy-stl to auto-convert: pip install numpy-stl")
		return stl_path, False

	converted_path = os.path.splitext(stl_path)[0] + "_mujoco_binary.stl"
	try:
		src_mesh = mesh.Mesh.from_file(stl_path)
		src_mesh.save(converted_path, mode=Mode.BINARY)
		print(f"[INFO] Converted ASCII STL -> Binary STL: {converted_path}")
		return converted_path, True
	except Exception as e:
		print(f"[WARN] Failed to convert ASCII STL with numpy-stl: {e}")
		return stl_path, False
