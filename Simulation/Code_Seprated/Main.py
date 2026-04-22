import os
import sys

import mujoco

from Simulation.Code_Seprated.config import (
	ENABLE_PLOTS,
	ENABLE_TELEMETRY,
	H_RATIO,
	L_RATIO,
	OUTPUT_DIR_NAME,
	TARGET_LENGTH,
)
from Simulation.Code_Seprated.flick import compute_flick_velocities_from_omega
from Simulation.Code_Seprated.geometry import get_mesh_dimensions_and_scale
from Simulation.Code_Seprated.io_cli import (
	get_playback_speed,
	input_omega,
	print_flick_computation,
	print_geometry_info,
	print_simulation_ready,
	select_material,
	select_model,
)
from Simulation.Code_Seprated.mesh_handler import ensure_mujoco_compatible_stl
from Simulation.Code_Seprated.model_xml import build_final_xml
from Simulation.Code_Seprated.paths import get_models_dir, get_project_root, get_stl_files
from Simulation.Code_Seprated.plot_motion import generate_motion_plots
from Simulation.Code_Seprated.simulation import run_simulation
from Simulation.Code_Seprated.telemetry import make_run_id, save_telemetry_csv


def main():
	project_root = get_project_root(__file__)
	models_dir = get_models_dir(project_root)

	try:
		stl_files = get_stl_files(models_dir)
	except FileNotFoundError as e:
		print(str(e))
		raise SystemExit

	selected_stl = select_model(stl_files)
	stl_path = os.path.join(models_dir, selected_stl).replace("\\", "/")
	mesh_path, converted_for_mujoco = ensure_mujoco_compatible_stl(stl_path)

	selected_material_name, selected_density = select_material()

	try:
		(
			raw_w,
			raw_l,
			raw_h,
			scale_factor,
			geom_width,
			geom_length,
			geom_height,
		) = get_mesh_dimensions_and_scale(mesh_path, target_length=TARGET_LENGTH)
	except Exception as e:
		print(f"Failed to load model geometry. Error: {e}")
		raise SystemExit

	print_geometry_info(raw_w, raw_l, raw_h, geom_width, geom_length, geom_height)

	omega = input_omega()
	v_x, v_z = compute_flick_velocities_from_omega(
		omega=omega,
		y_length=geom_length,
		h_ratio=H_RATIO,
		l_ratio=L_RATIO,
	)
	print_flick_computation(H_RATIO, L_RATIO, geom_length, v_x, v_z)

	playback_speed = get_playback_speed()
	target_width = geom_width
	target_height = 1.5 * target_width

	print_simulation_ready(
		selected_stl=selected_stl,
		converted_for_mujoco=converted_for_mujoco,
		material_name=f"{selected_material_name} ({selected_density} kg/m^3)",
		target_width=target_width,
		playback_speed=playback_speed,
	)

	final_xml = build_final_xml(
		mesh_path=mesh_path,
		scale_factor=scale_factor,
		target_height=target_height,
		selected_density=selected_density,
	)
	model = mujoco.MjModel.from_xml_string(final_xml)
	data = mujoco.MjData(model)
	samples = run_simulation(model, data, v_x=v_x, v_z=v_z, omega=omega, playback_speed=playback_speed)

	if ENABLE_TELEMETRY:
		output_dir = os.path.join(project_root, OUTPUT_DIR_NAME)
		run_id = make_run_id(prefix="flipo")
		csv_path = save_telemetry_csv(samples=samples, output_dir=output_dir, run_id=run_id)
		print(f"[INFO] Telemetry CSV saved: {csv_path}")

		if ENABLE_PLOTS:
			meta_text = (
				f"model={selected_stl}; omega={omega:.3f}; H/L={H_RATIO:.4f}/{L_RATIO:.4f}; "
				f"material={selected_material_name}; density={selected_density}"
			)
			figure_paths = generate_motion_plots(
				samples=samples,
				output_dir=output_dir,
				run_id=run_id,
				meta_text=meta_text,
			)
			for fig_path in figure_paths:
				print(f"[INFO] Figure saved: {fig_path}")


if __name__ == "__main__":
	main()
