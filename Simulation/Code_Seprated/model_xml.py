from Simulation.Code_Seprated.config import CONTACT_SETTINGS


def build_final_xml(mesh_path, scale_factor, target_height, selected_density):
	return f"""
<mujoco>
  <option timestep=\"0.002\" gravity=\"0 0 -9.81\"/>

  <visual>
	<quality shadowsize=\"4096\"/>
  </visual>

  <asset>
	<mesh name=\"flipo_mesh\" file=\"{mesh_path}\" scale=\"{scale_factor} {scale_factor} {scale_factor}\"/>
  </asset>

  <worldbody>
	<light diffuse=\".5 .5 .5\" pos=\"0 0 5\" dir=\"0 0 -1\"/>

	<geom type=\"plane\" size=\"3 3 0.1\" rgba=\".8 .8 .8 1\"
		  solimp=\"{CONTACT_SETTINGS['solimp']}\" solref=\"{CONTACT_SETTINGS['solref']}\"/>

	<body pos=\"0 0 {target_height}\">
	  <joint type=\"free\" damping=\"{CONTACT_SETTINGS['joint_damping']}\"/>

	  <geom type=\"mesh\" mesh=\"flipo_mesh\" rgba=\"0.2 0.6 0.8 1\"
			contype=\"0\" conaffinity=\"0\" density=\"{selected_density}\"/>

	  <geom type=\"mesh\" mesh=\"flipo_mesh\" rgba=\"1 1 1 0\"
			mass=\"0\" condim=\"{CONTACT_SETTINGS['condim']}\" friction=\"{CONTACT_SETTINGS['friction']}\"/>
	</body>
  </worldbody>
</mujoco>
"""
