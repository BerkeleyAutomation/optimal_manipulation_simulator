# INSTRUCTIONS
To run the script, execute:
"python tools/generate_grasps_fanuc.py cfg/generate_grasps_fanuc.yaml"
from the command line.

The script outputs a set of pose matrices containing the transformation from the chessboard to the gripper in .csv format for:
	- mid_pose: the pregrasp position of the gripper
	- end_pose: the position of the gripper when the fingers should close
	- lift_pose: the position of the gripper to lift the object

You can modify the parameters of the script using the definitions below.

# PARAMETER DEFINITIONS
You can modify the following parameters below in a .yaml configuration file to generate grasps.
All distances are specified in units of meters

object_key: the name of the object to run
stp_id: a tag for the stable pose of the object, the following are valid
	- cliimbing_hold: 2
	- endstop_holder: 2
	- gearbox: 5
	- mount2: 0
	- pipe_connector: 6
	- turbine_housing: 1
	- vase: 2
num_grasps: the number of grasps to generate pose matrices for
debug: (0 or 1) whether or not to show the hacky debug display for the generated grasps

delta_x: the distance along the x axis in the object frame of reference from the template to the platform corner
delta_y: the distance along the y axis in the object frame of reference from the template to the platform corner
delta_z: the distance along the z axis in the object frame of reference from the template to the platform corner

cb_platform_x: the distance along the x axis in the object frame of reference from the platform corner to the chessboard reference point
cb_platform_y: the distance along the y axis in the object frame of reference from the platform corner to the chessboard reference point
cb_platform_z: the distance along the z axis in the object frame of reference from the platform corner to the chessboard reference point

delta_pregrasp: the distance along the grasp approach axis for the mid pose
delta_lift: the distance along the table normal for the lift pose
robot_base_angle_thresh: the largest angle that the gripper approach axis can make with the chessboard x axis (to prevent weird arm configurations)

table_clearance: the min allowable height from the table
angle_from_table: the angle from the table in units of pi

force_closure_thresh: the threshold value for using force closure or not (0 == any grasp permitted, 1 == only force closure grasps)

output_dir: where to write the .csv files
