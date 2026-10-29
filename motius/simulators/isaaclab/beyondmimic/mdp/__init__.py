"""Native tracking terms and the exact required Isaac Lab MDP primitives."""
from isaaclab.envs.mdp import JointPositionActionCfg, action_rate_l2, base_ang_vel, base_lin_vel, generated_commands, joint_pos_limits, joint_pos_rel, joint_vel_rel, last_action, push_by_setting_velocity, randomize_rigid_body_material, time_out, undesired_contacts
from .commands import MotionCommandCfg
from .events import randomize_joint_default_pos, randomize_rigid_body_com
from .observations import robot_body_pos_b, robot_body_ori_b, motion_anchor_pos_b, motion_anchor_ori_b
from .rewards import motion_global_anchor_position_error_exp, motion_global_anchor_orientation_error_exp, motion_relative_body_position_error_exp, motion_relative_body_orientation_error_exp, motion_global_body_linear_velocity_error_exp, motion_global_body_angular_velocity_error_exp
from .terminations import bad_anchor_pos_z_only, bad_anchor_ori, bad_motion_body_pos_z_only
