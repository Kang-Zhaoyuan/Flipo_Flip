import mujoco
import numpy as np


def get_mesh_dimensions_and_scale(stl_path, target_length=0.35):
	"""Load mesh once and return raw/scaled dimensions and scale factor.

	Returns:
		tuple: (raw_w, raw_l, raw_h, scale_factor, width, length, height)
	"""
	temp_xml = f"""
<mujoco>
	<asset>
		<mesh name=\"flipo_mesh\" file=\"{stl_path}\" scale=\"1 1 1\"/>
	</asset>
	<worldbody>
		<body pos=\"0 0 0\">
			<geom type=\"mesh\" mesh=\"flipo_mesh\"/>
		</body>
	</worldbody>
</mujoco>
"""

	temp_model = mujoco.MjModel.from_xml_string(temp_xml)
	verts = np.asarray(temp_model.mesh_vert, dtype=float)
	if verts.ndim == 1:
		verts = verts.reshape(-1, 3)

	vert_adr = temp_model.mesh_vertadr[0]
	vert_num = temp_model.mesh_vertnum[0]
	mesh_verts = verts[vert_adr : vert_adr + vert_num]

	raw_w = np.max(mesh_verts[:, 0]) - np.min(mesh_verts[:, 0])
	raw_l = np.max(mesh_verts[:, 1]) - np.min(mesh_verts[:, 1])
	raw_h = np.max(mesh_verts[:, 2]) - np.min(mesh_verts[:, 2])

	if raw_l <= 0:
		raise RuntimeError("Mesh length on Y-axis is zero; cannot derive scale_factor.")

	scale_factor = target_length / raw_l
	width = raw_w * scale_factor
	length = raw_l * scale_factor
	height = raw_h * scale_factor
	return raw_w, raw_l, raw_h, scale_factor, width, length, height
