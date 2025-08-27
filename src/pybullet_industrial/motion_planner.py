"""
Motion Planning Module for PyBullet Industrial

This module provides OMPL-based motion planning capabilities for industrial robots.
It integrates with the existing RobotBase and CollisionChecker classes.
"""

import sys
import os
import time
import numpy as np
import pybullet as p
from typing import List, Tuple, Optional, Dict, Any
from itertools import product

# Try to import OMPL bindings
try:
    # Try multiple paths to find OMPL Python bindings
    ompl_paths = [
        # Relative path from current package
        # os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'test_motion_planner', 'ompl', 'py-bindings'),
        # Docker OMPL installation paths
        # OMPL py-bindings directory (created during build)
        # '/app/test_motion_planner/ompl/py-bindings',
        # '/app/.venv/lib/python3.10/site-packages',
        #'/home/rickymaggio/Documents/experiment/test_simulation/XeleritCoderBackend/test_motion_planner/ompl/py-bindings',
        '/app/tmp/ompl/py-bindings',
        '/tmp/ompl/py-bindings',
        '/usr/local/ompl/py-bindings'
        # Alternative system paths
        # '/usr/lib/python3/dist-packages',
        # '/usr/local/lib/python3/site-packages',
    ]
    print(f"NEW OMPL paths: {ompl_paths}")
    ompl_found = False
    for ompl_path in ompl_paths:
        if os.path.exists(ompl_path):
            print(f"Found OMPL path: {ompl_path}")
            if ompl_path not in sys.path:
                sys.path.insert(0, ompl_path)
            ompl_found = True
            break
    
    if not ompl_found:
        print("Warning: OMPL path not found in common locations")
        print("Attempting to import OMPL from system Python path...")
    import ompl
    from ompl import base as ob
    from ompl import geometric as og
    OMPL_AVAILABLE = True
    print("✓ OMPL Python bindings successfully imported")
    
except ImportError as e:
    OMPL_AVAILABLE = False
    print(f"Warning: OMPL not available. Motion planning will not work.")
    print(f"Import error: {e}")
    print("Please install OMPL with Python bindings to use motion planning features.")
    print("In Docker, ensure OMPL is built and installed system-wide.")


class MotionPlanner:
    """
    OMPL-based motion planner for industrial robots and tools.
    
    This class provides motion planning capabilities using OMPL (Open Motion Planning Library)
    and integrates with the existing pybullet_industrial RobotBase and EndeffectorTool.
    
    Args:
        subject: RobotBase or EndeffectorTool instance to plan for
        collision_checker: CollisionChecker instance for collision detection
        planning_time: Default planning time in seconds
        interpolation_steps: Number of interpolation steps for path smoothing
    """
    
    def __init__(self, subject, collision_checker=None, planning_time: float = 5.0, 
                 interpolation_steps: int = 500):
        if not OMPL_AVAILABLE:
            raise ImportError("OMPL is not available. Please install OMPL with Python bindings.")
        
        self.subject = subject
        self.collision_checker = collision_checker
        self.planning_time = planning_time
        self.interpolation_steps = interpolation_steps
        
        # Determine if subject is robot or tool
        self._determine_subject_type()
        
        # Set dimensions based on subject type
        if self.subject_type == "tool":
            # Tool mode: 7D state space (position + quaternion)
            self.num_dimensions = 7
            self.active_joint_indices = []  # No joint indices needed for tool mode
        else:
            # Robot mode: joint-based state space
            self.active_joint_indices = self._get_active_joint_indices()
            self.num_dimensions = len(self.active_joint_indices)
        
        # Setup OMPL state space
        self.space = self._create_state_space()
        self.simple_setup = og.SimpleSetup(self.space)
        
        # Setup collision detection
        self._setup_collision_detection()
        
        # Default planner
        self.set_planner("RRT")
    
    def _determine_subject_type(self):
        """Determine if the subject is a robot or tool and set up accordingly."""
        from pybullet_industrial.robot_base import RobotBase
        from pybullet_industrial.endeffector_tool import EndeffectorTool
        
        if isinstance(self.subject, RobotBase):
            self.subject_type = "robot"
            self.robot = self.subject
            self.tool = None
            print("Motion planner initialized with RobotBase")
        elif isinstance(self.subject, EndeffectorTool):
            self.subject_type = "tool"
            self.tool = self.subject
            # Get the coupled robot from the tool
            if self.tool.is_coupled():
                self.robot = self.tool._coupled_robot
                print("Motion planner initialized with EndeffectorTool (coupled to robot)")
            else:
                raise ValueError("Tool must be coupled to a robot for motion planning")
        else:
            raise TypeError("Subject must be either RobotBase or EndeffectorTool")
    
    def change_subject(self, new_subject):
        """
        Change the subject of the motion planner (robot or tool).
        
        Args:
            new_subject: New RobotBase or EndeffectorTool instance
        """
        print(f"Changing motion planner subject from {self.subject_type} to new subject...")
        
        # Store the new subject
        self.subject = new_subject
        
        # Re-determine subject type and set up
        self._determine_subject_type()
        
        # Set dimensions based on subject type
        if self.subject_type == "tool":
            # Tool mode: 7D state space (position + quaternion)
            self.num_dimensions = 7
            self.active_joint_indices = []  # No joint indices needed for tool mode
        else:
            # Robot mode: joint-based state space
            self.active_joint_indices = self._get_active_joint_indices()
            self.num_dimensions = len(self.active_joint_indices)
        
        # Recreate state space
        self.space = self._create_state_space()
        self.simple_setup = og.SimpleSetup(self.space)
        
        # Re-setup collision detection
        self._setup_collision_detection()
        
        print(f"Motion planner now using {self.subject_type} with {self.num_dimensions} dimensions")
    
    def _get_active_joint_indices(self) -> List[int]:
        """Get indices of active (non-fixed) joints from the robot."""
        if self.subject_type == "tool":
            # For tool mode, we don't need joint indices - we'll use position/orientation state space
            return []
        else:
            # For robot mode, get active joint indices
            active_joints = []
            for i in range(self.robot.number_of_joints):
                joint_info = p.getJointInfo(self.robot.urdf, i)
                if joint_info[2] != p.JOINT_FIXED:  # Not fixed joint
                    active_joints.append(i)
            return active_joints
    
    def _create_state_space(self) -> ob.RealVectorStateSpace:
        """Create OMPL state space with appropriate dimensions for robot or tool."""
        if self.subject_type == "tool":
            # For tool mode: 7D state space [x, y, z, qx, qy, qz, qw] (position + quaternion)
            space = ob.RealVectorStateSpace(7)
            bounds = ob.RealVectorBounds(7)
            
            # Position bounds (x, y, z) - reasonable workspace limits
            # These bounds should be set based on the robot's reachable workspace
            workspace_bounds = self._get_workspace_bounds()
            
            # Position bounds
            bounds.setLow(0, workspace_bounds['x_min'])  # x
            bounds.setHigh(0, workspace_bounds['x_max'])
            bounds.setLow(1, workspace_bounds['y_min'])  # y
            bounds.setHigh(1, workspace_bounds['y_max'])
            bounds.setLow(2, workspace_bounds['z_min'])  # z
            bounds.setHigh(2, workspace_bounds['z_max'])
            
            # Quaternion bounds (qx, qy, qz, qw) - normalized quaternions
            bounds.setLow(3, -1.0)  # qx
            bounds.setHigh(3, 1.0)
            bounds.setLow(4, -1.0)  # qy
            bounds.setHigh(4, 1.0)
            bounds.setLow(5, -1.0)  # qz
            bounds.setHigh(5, 1.0)
            bounds.setLow(6, -1.0)  # qw
            bounds.setHigh(6, 1.0)
            
            print(f"Creating TOOL state space with 7 dimensions (position + quaternion)")
            print(f"Workspace bounds: X[{workspace_bounds['x_min']:.2f}, {workspace_bounds['x_max']:.2f}], "
                  f"Y[{workspace_bounds['y_min']:.2f}, {workspace_bounds['y_max']:.2f}], "
                  f"Z[{workspace_bounds['z_min']:.2f}, {workspace_bounds['z_max']:.2f}]")
            
        else:
            # For robot mode: joint-based state space
            space = ob.RealVectorStateSpace(self.num_dimensions)
            bounds = ob.RealVectorBounds(self.num_dimensions)
            joint_names, joint_indices = self.robot.get_moveable_joints()
            
            print(f"Creating ROBOT state space with {self.num_dimensions} dimensions")
            print(f"Active joint indices: {self.active_joint_indices}")
            
            for i, (joint_name, joint_idx) in enumerate(zip(joint_names, joint_indices)):
                if joint_idx in self.active_joint_indices:
                    # Get joint limits from robot
                    lower_limit = self.robot._lower_joint_limit[joint_idx]
                    upper_limit = self.robot._upper_joint_limit[joint_idx]
                    
                    # Handle infinite limits
                    if not np.isfinite(lower_limit):
                        lower_limit = -np.pi
                    if not np.isfinite(upper_limit):
                        upper_limit = np.pi
                        
                    bounds.setLow(i, lower_limit)
                    bounds.setHigh(i, upper_limit)
                    
                    print(f"  Joint {joint_name} (idx {joint_idx}): [{lower_limit:.3f}, {upper_limit:.3f}]")
        
        space.setBounds(bounds)
        return space
    
    def _get_workspace_bounds(self) -> dict:
        """Get reasonable workspace bounds for tool planning."""
        # Default workspace bounds - these should be customized based on your robot
        # For now, using conservative bounds that should work for most 6-DOF robots
        
        # Get current robot position as reference
        base_pos, _ = p.getBasePositionAndOrientation(self.robot.urdf)
        
        # Convert PyBullet Vector to numpy array
        base_pos = np.array(base_pos)
        
        # Conservative workspace bounds relative to robot base
        # These bounds assume the robot can reach ±1.5m in each direction from base
        # You should adjust these based on your specific robot's reach
        bounds = {
            'x_min': base_pos[0] - 1.5,
            'x_max': base_pos[0] + 1.5,
            'y_min': base_pos[1] - 1.5,
            'y_max': base_pos[1] + 1.5,
            'z_min': base_pos[2] - 0.5,  # Lower bound closer to ground
            'z_max': base_pos[2] + 2.0,  # Upper bound higher for overhead reach
        }
        
        return bounds
    
    def _setup_collision_detection(self):
        """Setup collision detection for OMPL."""
        if self.collision_checker is not None:
            # Disable internal collision checking, only check external collisions
            print("Disabling internal collision checking, only checking external collisions")
            self.collision_checker.enable_internal_collision = False
            self.collision_checker.enable_external_collision = True
            
            # Check if robot has coupled end-effector tools and add them to collision checker
            self._add_coupled_tools_to_collision_checker()
            
            # Use existing collision checker
            self.simple_setup.setStateValidityChecker(
                ob.StateValidityCheckerFn(self._check_state_validity_with_collision_checker)
            )
        else:
            # Simple collision detection
            self.simple_setup.setStateValidityChecker(
                ob.StateValidityCheckerFn(self._check_state_validity_simple)
            )
    
    def _add_coupled_tools_to_collision_checker(self):
        """Add any coupled end-effector tools to the collision checker."""
        # Get the number of constraints by trying to get constraint info for increasing IDs
        # until we get None (no more constraints)
        constraint_id = 0
        max_constraints_to_check = 100  # Safety limit
        
        while constraint_id < max_constraints_to_check:
            try:
                constraint_info = p.getConstraintInfo(constraint_id)
                if constraint_info is not None:
                    # Check if this constraint connects our robot to another body
                    if constraint_info[1] == self.robot.urdf:  # Robot is body A
                        tool_body_id = constraint_info[2]  # Tool is body B
                        
                        # Use the new tool coupling handler
                        self._handle_tool_coupling(tool_body_id)
                            
                constraint_id += 1
            except Exception as e:
                # No more constraints or invalid constraint ID
                break
    
    def _normalize_quaternion_state(self, state) -> ob.State:
        """Normalize quaternion components of a state to ensure they represent a valid quaternion."""
        if self.subject_type == "tool":
            # Extract quaternion components [qx, qy, qz, qw]
            qx, qy, qz, qw = state[3], state[4], state[5], state[6]
            
            # Normalize quaternion
            quat_norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if quat_norm > 0:
                state[3] = qx / quat_norm
                state[4] = qy / quat_norm
                state[5] = qz / quat_norm
                state[6] = qw / quat_norm
        
        return state
    
    def _check_state_validity_with_collision_checker(self, state) -> bool:
        """Check state validity using the collision checker."""
        # Normalize quaternion if in tool mode
        state = self._normalize_quaternion_state(state)
        
        if self.subject_type == "tool":
            # Tool mode: state is [x, y, z, qx, qy, qz, qw]
            # Convert tool pose to robot joint configuration using IK
            tool_pose = self._ompl_state_to_tool_pose(state)
            joint_state = self._tool_pose_to_joints(tool_pose)
            if joint_state is None:
                print(f"  State validity check (tool mode): INVALID - IK failed")
                return False
        else:
            # Robot mode: state is joint angles
            joint_state = self._ompl_state_to_dict(state)
        
        # Set robot to the given state
        self.robot.set_joint_position(joint_state)
        
        # Check if collision-free
        is_valid = self.collision_checker.is_collision_free()
        
        # Debug information
        print(f"  State validity check (collision checker): {is_valid}")
        if not is_valid:
            if self.subject_type == "tool":
                print(f"    Tool pose: {tool_pose}")
            print(f"    Joint state: {joint_state}")
            print(f"    Collision points: {len(p.getContactPoints(self.robot.urdf))}")
            
            # Check internal collisions
            internal_collisions = self.collision_checker.get_internal_collisions()
            if internal_collisions:
                print(f"    Internal collisions: {internal_collisions}")
            
            # Check external collisions
            external_collisions = self.collision_checker.get_external_collisions()
            if external_collisions:
                print(f"    External collisions: {external_collisions}")
            
            # Check if internal collision checking is enabled
            print(f"    Internal collision enabled: {self.collision_checker.enable_internal_collision}")
            print(f"    External collision enabled: {self.collision_checker.enable_external_collision}")
        
        return is_valid
    
    def _check_state_validity_simple(self, state) -> bool:
        """Simple state validity check without collision checker."""
        # Normalize quaternion if in tool mode
        state = self._normalize_quaternion_state(state)
        
        if self.subject_type == "tool":
            # Tool mode: state is [x, y, z, qx, qy, qz, qw]
            # Convert tool pose to robot joint configuration using IK
            tool_pose = self._ompl_state_to_tool_pose(state)
            joint_state = self._tool_pose_to_joints(tool_pose)
            if joint_state is None:
                print(f"  State validity check (simple, tool mode): INVALID - IK failed")
                return False
        else:
            # Robot mode: state is joint angles
            joint_state = self._ompl_state_to_dict(state)
        
        # Set robot to the given state
        self.robot.set_joint_position(joint_state)
        
        # Basic collision check using PyBullet
        contact_points = p.getContactPoints(self.robot.urdf)
        is_valid = len(contact_points) == 0
        
        # Debug information
        print(f"  State validity check (simple): {is_valid}")
        if not is_valid:
            if self.subject_type == "tool":
                print(f"    Tool pose: {tool_pose}")
            print(f"    Joint state: {joint_state}")
            print(f"    Collision points: {len(contact_points)}")
            for i, contact in enumerate(contact_points[:3]):  # Show first 3 collisions
                print(f"      Contact {i+1}: Body {contact[1]} <-> Body {contact[2]}")
        
        return is_valid
    
    def _ompl_state_to_dict(self, state) -> Dict[str, float]:
        """Convert OMPL state to joint dictionary (robot mode only)."""
        if self.subject_type == "tool":
            raise ValueError("_ompl_state_to_dict is for robot mode only. Use _ompl_state_to_tool_pose for tool mode.")
        
        joint_names, joint_indices = self.robot.get_moveable_joints()
        joint_dict = {}
        
        print(f"  Converting OMPL state to joint dict:")
        print(f"    Active joint indices: {self.active_joint_indices}")
        print(f"    OMPL state values: {[state[i] for i in range(self.num_dimensions)]}")
        
        for i, (joint_name, joint_idx) in enumerate(zip(joint_names, joint_indices)):
            if joint_idx in self.active_joint_indices:
                active_idx = self.active_joint_indices.index(joint_idx)
                joint_dict[joint_name] = state[active_idx]
                print(f"    {joint_name} (idx {joint_idx}): {state[active_idx]}")
        
        print(f"    Final joint dict: {joint_dict}")
        return joint_dict
    
    def _ompl_state_to_tool_pose(self, state) -> Tuple[np.ndarray, np.ndarray]:
        """Convert OMPL state to tool pose (tool mode only)."""
        if self.subject_type != "tool":
            raise ValueError("_ompl_state_to_tool_pose is for tool mode only.")
        
        # State is [x, y, z, qx, qy, qz, qw]
        position = np.array([state[0], state[1], state[2]])
        orientation = np.array([state[3], state[4], state[5], state[6]])  # quaternion [qx, qy, qz, qw]
        
        # Normalize quaternion to ensure it's valid
        quat_norm = np.linalg.norm(orientation)
        if quat_norm > 0:
            orientation = orientation / quat_norm
        
        print(f"  Converting OMPL state to tool pose:")
        print(f"    Position: {position}")
        print(f"    Orientation (quaternion): {orientation}")
        
        return position, orientation
    
    def _tool_pose_to_joints(self, tool_pose: Tuple[np.ndarray, np.ndarray]) -> Optional[Dict[str, float]]:
        """Convert tool pose to robot joint configuration using inverse kinematics."""
        position, orientation = tool_pose
        
        # Use PyBullet's inverse kinematics to get joint configuration for the tool pose
        # Note: We need to use the robot's end effector that the tool is attached to
        try:
            # Get the end effector ID that the tool is attached to
            if hasattr(self.tool, '_coupling_frame_name') and self.tool._coupling_frame_name:
                endeffector_name = self.tool._coupling_frame_name
            else:
                # Fallback to default end effector
                endeffector_name = None
            
            # Use the robot's IK method which handles the PyBullet call internally
            if endeffector_name:
                # Use the robot's reset_endeffector_pose to get joint configuration
                # This method uses IK but doesn't actually move the robot
                # We'll temporarily store current joint state, call IK, then restore
                current_joints = self.robot.get_joint_position()
                
                # Call IK through the robot's method
                joint_poses = self.robot.server.calculateInverseKinematics(
                    self.robot.urdf,
                    self.robot._convert_endeffector(endeffector_name),
                    position,
                    targetOrientation=orientation,
                    lowerLimits=self.robot._lower_joint_limit,
                    upperLimits=self.robot._upper_joint_limit
                )
            else:
                # Use default end effector
                joint_poses = self.robot.server.calculateInverseKinematics(
                    self.robot.urdf,
                    self.robot._default_endeffector_id,
                    position,
                    targetOrientation=orientation,
                    lowerLimits=self.robot._lower_joint_limit,
                    upperLimits=self.robot._upper_joint_limit
                )
            
            if joint_poses is not None and len(joint_poses) > 0:
                # Convert to joint dictionary
                joint_names, joint_indices = self.robot.get_moveable_joints()
                joint_dict = {}
                
                # Map the IK results to joint names
                for i, joint_name in enumerate(joint_names):
                    if i < len(joint_poses):
                        joint_dict[joint_name] = joint_poses[i]
                
                print(f"    IK successful: {joint_dict}")
                return joint_dict
            else:
                print(f"    IK failed: No solution found")
                return None
                
        except Exception as e:
            print(f"    IK error: {e}")
            return None
    
    def _dict_to_ompl_state(self, joint_dict: Dict[str, float]) -> ob.State:
        """Convert joint dictionary to OMPL state (robot mode only)."""
        if self.subject_type == "tool":
            raise ValueError("_dict_to_ompl_state is for robot mode only. Use _tool_pose_to_ompl_state for tool mode.")
        
        state = ob.State(self.space)
        joint_names, joint_indices = self.robot.get_moveable_joints()
        
        for i, (joint_name, joint_idx) in enumerate(zip(joint_names, joint_indices)):
            if joint_idx in self.active_joint_indices:
                active_idx = self.active_joint_indices.index(joint_idx)
                if joint_name in joint_dict:
                    state[active_idx] = joint_dict[joint_name]
        
        return state
    
    def _tool_pose_to_ompl_state(self, tool_pose: Tuple[np.ndarray, np.ndarray]) -> ob.State:
        """Convert tool pose to OMPL state (tool mode only)."""
        if self.subject_type != "tool":
            raise ValueError("_tool_pose_to_ompl_state is for tool mode only.")
        
        position, orientation = tool_pose
        
        # Ensure orientation is a numpy array and normalize it
        orientation = np.array(orientation)
        quat_norm = np.linalg.norm(orientation)
        if quat_norm > 0:
            orientation = orientation / quat_norm
        
        # Create OMPL state [x, y, z, qx, qy, qz, qw]
        state = ob.State(self.space)
        state[0] = position[0]  # x
        state[1] = position[1]  # y
        state[2] = position[2]  # z
        state[3] = orientation[0]  # qx
        state[4] = orientation[1]  # qy
        state[5] = orientation[2]  # qz
        state[6] = orientation[3]  # qw
        
        return state
    
    def set_planner(self, planner_name: str):
        """
        Set the motion planning algorithm.
        
        Args:
            planner_name: Name of the planner (RRT, RRTstar, BITstar, etc.)
        """
        if planner_name == "RRT":
            self.planner = og.RRT(self.simple_setup.getSpaceInformation())
        elif planner_name == "RRTstar":
            self.planner = og.RRTstar(self.simple_setup.getSpaceInformation())
        elif planner_name == "BITstar":
            self.planner = og.BITstar(self.simple_setup.getSpaceInformation())
        elif planner_name == "RRTConnect":
            self.planner = og.RRTConnect(self.simple_setup.getSpaceInformation())
        elif planner_name == "PRM":
            self.planner = og.PRM(self.simple_setup.getSpaceInformation())
        elif planner_name == "EST":
            self.planner = og.EST(self.simple_setup.getSpaceInformation())
        elif planner_name == "FMT":
            self.planner = og.FMT(self.simple_setup.getSpaceInformation())
        elif planner_name == "InformedRRTstar":
            self.planner = og.InformedRRTstar(self.simple_setup.getSpaceInformation())
        else:
            raise ValueError(f"Unknown planner: {planner_name}")
        
        self.simple_setup.setPlanner(self.planner)
    
    def plan(self, goal, start_joints: Optional[Dict[str, float]] = None,
             planning_time: Optional[float] = None, 
             goal_type: str = "auto") -> Tuple[bool, List[Dict[str, float]]]:
        """
        Plan a path to the goal configuration.
        
        Args:
            goal: Goal specification. Can be:
                - Dict[str, float]: Joint configuration
                - Tuple[np.ndarray, np.ndarray]: (position, orientation) for end-effector
                - np.ndarray: Position only for end-effector
            start_joints: Starting joint configuration (uses current if None)
            planning_time: Planning time limit (uses default if None)
            goal_type: Type of goal ("joints", "pose", "position", or "auto" for automatic detection)
            
        Returns:
            Tuple of (success, path) where path is a list of joint configurations
        """
        if planning_time is None:
            planning_time = self.planning_time
        
        # Get start state
        if start_joints is None:
            start_joints = self.robot.get_joint_position()
        
        # Determine goal type and convert to joint configuration
        goal_joints = self._resolve_goal(goal, goal_type)
        if goal_joints is None:
            return False, []
        
        # Convert to OMPL states
        start_state = self._dict_to_ompl_state(start_joints)
        goal_state = self._dict_to_ompl_state(goal_joints)
        
        # Set start and goal
        self.simple_setup.setStartAndGoalStates(start_state, goal_state)
        
        # Solve
        solved = self.simple_setup.solve(planning_time)
        
        if solved:
            # Get solution path
            path = self.simple_setup.getSolutionPath()
            path.interpolate(self.interpolation_steps)
            
            # Convert to joint dictionaries
            joint_path = []
            for state in path.getStates():
                joint_dict = self._ompl_state_to_dict(state)
                joint_path.append(joint_dict)
            
            return True, joint_path
        else:
            return False, []
    
    def _resolve_goal(self, goal, goal_type: str = "auto") -> Optional[Dict[str, float]]:
        """
        Resolve goal specification to joint configuration.
        
        Args:
            goal: Goal specification (joints, pose, or position)
            goal_type: Type of goal ("joints", "pose", "position", or "auto")
            
        Returns:
            Joint configuration dictionary or None if resolution failed
        """
        if goal_type == "auto":
            # Auto-detect goal type
            if isinstance(goal, dict):
                goal_type = "joints"
            elif isinstance(goal, (tuple, list)) and len(goal) == 2:
                goal_type = "pose"
            elif isinstance(goal, np.ndarray):
                goal_type = "position"
            else:
                raise ValueError(f"Cannot auto-detect goal type for: {type(goal)}")
        
        if goal_type == "joints":
            # Goal is already in joint space
            return goal
        elif goal_type == "pose":
            # Goal is (position, orientation)
            if len(goal) != 2:
                raise ValueError("Pose goal must be (position, orientation)")
            position, orientation = goal
            return self._pose_to_joints(position, orientation)
        elif goal_type == "position":
            # Goal is position only (use current orientation)
            current_pose = self.robot.get_endeffector_pose()
            current_orientation = current_pose[1]  # Get current orientation
            return self._pose_to_joints(goal, current_orientation)
        else:
            raise ValueError(f"Unknown goal type: {goal_type}")
    
    def _pose_to_joints(self, position: np.ndarray, orientation: np.ndarray) -> Optional[Dict[str, float]]:
        """
        Convert end-effector pose to joint configuration.
        
        Args:
            position: Target position
            orientation: Target orientation (quaternion)
            
        Returns:
            Joint configuration or None if IK fails
        """
        try:
            # Store current state
            current_joints = self.robot.get_joint_position()
            
            # Try to set the end-effector to the target pose
            self.robot.set_endeffector_pose(position, orientation)
            
            # Get the resulting joint configuration
            goal_joints = self.robot.get_joint_position()
            
            # Restore original state
            self.robot.set_joint_position(current_joints)
            
            return goal_joints
            
        except Exception as e:
            print(f"Inverse kinematics failed: {e}")
            return None
    
    def plan_to_joints(self, goal_joints: Dict[str, float], 
                      start_joints: Optional[Dict[str, float]] = None,
                      planning_time: Optional[float] = None) -> Tuple[bool, List[Dict[str, float]]]:
        """
        Plan a path to a joint configuration.
        
        Args:
            goal_joints: Target joint configuration
            start_joints: Starting joint configuration (uses current if None)
            planning_time: Planning time limit (uses default if None)
            
        Returns:
            Tuple of (success, path) where path is a list of joint configurations
        """
        return self.plan(goal_joints, start_joints, planning_time, goal_type="joints")
    
    def plan_to_pose(self, target_position: np.ndarray, 
                    target_orientation: Optional[np.ndarray] = None,
                    endeffector_name: Optional[str] = None,
                    planning_time: Optional[float] = None) -> Tuple[bool, List[Dict[str, float]]]:
        """
        Plan a path to a target end-effector pose.
        
        Args:
            target_position: Target position
            target_orientation: Target orientation (quaternion)
            endeffector_name: Name of end-effector (ignored, uses default)
            planning_time: Planning time limit
            
        Returns:
            Tuple of (success, path)
        """
        if target_orientation is None:
            # Use current orientation if not specified
            current_pose = self.robot.get_endeffector_pose()
            target_orientation = current_pose[1]
        
        return self.plan((target_position, target_orientation), 
                        planning_time=planning_time, goal_type="pose")
    
    def plan_to_position(self, target_position: np.ndarray,
                        planning_time: Optional[float] = None) -> Tuple[bool, List[Dict[str, float]]]:
        """
        Plan a path to a target end-effector position (keeping current orientation).
        
        Args:
            target_position: Target position
            planning_time: Planning time limit
            
        Returns:
            Tuple of (success, path)
        """
        return self.plan(target_position, planning_time=planning_time, goal_type="position")
    
    def plan_to_tool_pose(self, target_position: np.ndarray, 
                         target_orientation: Optional[np.ndarray] = None,
                         planning_time: Optional[float] = None) -> Tuple[bool, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Plan a path to a target tool pose (tool-centric planning).
        
        Args:
            target_position: Target position for the tool
            target_orientation: Target orientation for the tool (quaternion)
            planning_time: Planning time limit
            
        Returns:
            Tuple of (success, path) where path is a list of (position, orientation) tuples
        """
        if self.subject_type != "tool":
            raise ValueError("plan_to_tool_pose can only be used when subject is a tool")
        
        if target_orientation is None:
            # Use current tool orientation if not specified
            current_pose = self.tool.get_tool_pose()
            target_orientation = current_pose[1]
        
        # Get current tool pose as start
        start_position, start_orientation = self.tool.get_tool_pose()
        
        # Convert PyBullet Vector objects to numpy arrays
        start_position = np.array(start_position)
        start_orientation = np.array(start_orientation)
        target_position = np.array(target_position)
        target_orientation = np.array(target_orientation)

        distance = np.linalg.norm(target_position - start_position)
        num_steps = max(int(distance * 500 / 0.7), 60)

        # Create a simple linear interpolation path in tool pose space
        path = []
        
        for i in range(num_steps + 1):
            t = i / num_steps
            
            # Interpolate position
            interp_position = start_position + t * (target_position - start_position)
            
            # Interpolate orientation (SLERP for quaternions)
            interp_orientation = self._slerp_quaternion(start_orientation, target_orientation, t)
            
            path.append((interp_position, interp_orientation))
        
        return True, path
    
    def plan_to_tool_position(self, target_position: np.ndarray,
                            planning_time: Optional[float] = None) -> Tuple[bool, List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Plan a path to a target tool position (keeping current orientation).
        
        Args:
            target_position: Target position for the tool
            planning_time: Planning time limit
            
        Returns:
            Tuple of (success, path) where path is a list of (position, orientation) tuples
        """
        if self.subject_type != "tool":
            raise ValueError("plan_to_tool_position can only be used when subject is a tool")
        
        # Get current tool pose
        current_position, current_orientation = self.tool.get_tool_pose()
        
        # Convert PyBullet Vector objects to numpy arrays
        current_position = np.array(current_position)
        current_orientation = np.array(current_orientation)
        target_position = np.array(target_position)
        
        # Create a simple linear interpolation path in tool position space
        num_steps = self.interpolation_steps
        path = []
        
        for i in range(num_steps + 1):
            t = i / num_steps
            
            # Interpolate position only
            interp_position = current_position + t * (target_position - current_position)
            
            # Keep current orientation
            path.append((interp_position, current_orientation))
        
        return True, path
    
    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation (SLERP) between two quaternions.
        
        Args:
            q1: First quaternion
            q2: Second quaternion
            t: Interpolation parameter (0 to 1)
            
        Returns:
            Interpolated quaternion
        """
        # Ensure quaternions are normalized
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Calculate dot product
        dot = np.dot(q1, q2)
        
        # If quaternions are very close, use linear interpolation
        if abs(dot) > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        
        # Ensure shortest path
        if dot < 0:
            q2 = -q2
            dot = -dot
        
        # Calculate angle
        theta = np.arccos(np.clip(dot, -1, 1))
        
        # SLERP formula
        if theta < 1e-6:
            result = q1
        else:
            result = (np.sin((1 - t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)
        
        return result / np.linalg.norm(result)
    
    def execute_path(self, path, dynamics: bool = False, step_delay: float = 0.01):
        """
        Execute a planned path using the appropriate interface (robot or tool).
        
        Args:
            path: List of joint configurations (for robot) or (position, orientation) tuples (for tool)
            dynamics: Whether to use dynamics simulation
            step_delay: Delay between steps
        """
        if self.subject_type == "robot":
            # Robot-centric execution - path should be joint configurations
            if not isinstance(path[0], dict):
                raise ValueError("For robot planning, path must be a list of joint dictionaries")
            
            for joint_config in path:
                if dynamics:
                    # Use position control with dynamics
                    for joint_name, position in joint_config.items():
                        joint_idx = self.robot._joint_name_to_index[joint_name]
                        p.setJointMotorControl2(
                            self.robot.urdf, joint_idx, 
                            p.POSITION_CONTROL, position, 
                            force=self.robot.max_joint_force[joint_idx]
                        )
                else:
                    # Direct state setting
                    self.robot.set_joint_position(joint_config)
                
                p.stepSimulation()
                time.sleep(step_delay)
                
        elif self.subject_type == "tool":
            # Tool-centric execution - path should be (position, orientation) tuples
            if not isinstance(path[0], (tuple, list)) or len(path[0]) != 2:
                raise ValueError("For tool planning, path must be a list of (position, orientation) tuples")
            
            for position, orientation in path:
                # Use tool's set_tool_pose method for movement
                self.tool.set_tool_pose(position, orientation)
                p.stepSimulation()
                time.sleep(step_delay)
    
    # def draw_tool_path(self, path):
    #     if not isinstance(path[0], (tuple, list)) or len(path[0]) != 2:
    #             raise ValueError("For tool planning, path must be a list of (position, orientation) tuples")
        
    #     from pybullet_industrial.utility import draw_path
    #     path = []
    #     for position, orientation in path:
    #         path.append(position)
    #     path = np.array(path)
    #     draw_path(path, color=[0, 0, 1], width=2.0)
    
    def execute_tool_path(self, path: List[Tuple[np.ndarray, np.ndarray]], 
                         step_delay: float = 0.01):
        """
        Execute a path defined in tool poses (tool-centric execution).
        
        Args:
            path: List of (position, orientation) tuples for the tool
            step_delay: Delay between steps
        """
        if self.subject_type != "tool":
            raise ValueError("execute_tool_path can only be used when subject is a tool")
        
        for position, orientation in path:
            # Use tool's set_tool_pose method
            self.tool.set_tool_pose(position, orientation)
            p.stepSimulation()
            time.sleep(step_delay)
    
    def get_available_planners(self) -> List[str]:
        """Get list of available planners."""
        return ["RRT", "RRTstar", "BITstar", "RRTConnect", "PRM", "EST", "FMT", "InformedRRTstar"]

    def add_end_effector_tool(self, tool_body_id: int):
        """Add an end-effector tool to the collision checker.
        
        This method adds the tool to collision checking but does NOT ignore
        collisions between the robot and tool, as this can interfere with
        PyBullet's physical coupling constraints.
        
        Args:
            tool_body_id (int): The PyBullet body ID of the tool
        """
        if self.collision_checker is not None:
            # Add tool to collision checker for external obstacle detection
            existing_bodies = [body['body_id'] for body in self.collision_checker._bodies_information]
            if tool_body_id not in existing_bodies:
                print(f"Adding end-effector tool (body {tool_body_id}) to collision checker")
                self.collision_checker.add_body_id(tool_body_id)
                
                # Note: We do NOT ignore collisions between robot and tool
                # This preserves PyBullet's physical coupling while still
                # allowing the tool to be considered in collision checking
                # for external obstacles
            else:
                print(f"Tool (body {tool_body_id}) already in collision checker")
        else:
            print(f"Warning: No collision checker available, cannot add tool {tool_body_id}")

    def _handle_tool_coupling(self, tool_body_id: int):
        """Handle tool coupling in a way that preserves PyBullet physics.
        
        This method ensures that tools are properly integrated with the
        motion planner without breaking PyBullet's physical coupling constraints.
        
        Args:
            tool_body_id (int): The PyBullet body ID of the tool
        """
        if self.collision_checker is not None:
            # Add tool to collision checker
            self.add_end_effector_tool(tool_body_id)
            
            # Ensure the tool is treated as part of the robot for collision checking
            # but without interfering with the physical coupling
            print(f"Tool {tool_body_id} coupled and added to collision checking")
        else:
            print(f"Warning: No collision checker available for tool {tool_body_id}")

    def configure_tool_collision_checking(self, enable_tool_collision_checking: bool = True, 
                                        ignore_robot_tool_collisions: bool = False):
        """Configure how tools are handled in collision checking.
        
        Args:
            enable_tool_collision_checking (bool): Whether to include tools in collision checking
            ignore_robot_tool_collisions (bool): Whether to ignore collisions between robot and tool
                                                (use with caution as it can interfere with PyBullet physics)
        """
        if self.collision_checker is None:
            print("Warning: No collision checker available")
            return
            
        if not enable_tool_collision_checking:
            print("Tool collision checking disabled")
            return
            
        if ignore_robot_tool_collisions:
            print("Warning: Ignoring robot-tool collisions may interfere with PyBullet physics")
            # This is the old behavior that can cause tools to float
            # Only use if you understand the implications
            pass
        else:
            print("Tool collision checking enabled (preserving PyBullet physics)")
            
        # Store the configuration for future tool additions
        self._tool_collision_config = {
            'enable_tool_collision_checking': enable_tool_collision_checking,
            'ignore_robot_tool_collisions': ignore_robot_tool_collisions
        }


# Extension to RobotBase class
def add_motion_planning_to_robot_base():
    """Add motion planning methods to RobotBase class."""
    
    def create_motion_planner(self, collision_checker=None, **kwargs):
        """Create a motion planner for this robot."""
        return MotionPlanner(self, collision_checker, **kwargs)
    
    def plan_to_joints(self, goal_joints: Dict[str, float], **kwargs):
        """Plan to a joint configuration."""
        planner = self.create_motion_planner()
        return planner.plan_to_joints(goal_joints, **kwargs)
    
    def plan_to_pose(self, target_position: np.ndarray, 
                    target_orientation: Optional[np.ndarray] = None,
                    endeffector_name: Optional[str] = None, **kwargs):
        """Plan to an end-effector pose."""
        planner = self.create_motion_planner()
        return planner.plan_to_pose(target_position, target_orientation, endeffector_name, **kwargs)
    
    def plan_to_position(self, target_position: np.ndarray, **kwargs):
        """Plan to an end-effector position (keeping current orientation)."""
        planner = self.create_motion_planner()
        return planner.plan_to_position(target_position, **kwargs)
    
    def plan(self, goal, **kwargs):
        """Plan to a goal (auto-detects goal type)."""
        planner = self.create_motion_planner()
        return planner.plan(goal, **kwargs)
    
    def execute_planned_path(self, path: List[Dict[str, float]], **kwargs):
        """Execute a planned path."""
        planner = self.create_motion_planner()
        return planner.execute_path(path, **kwargs)
    
    def add_end_effector_tool_to_planner(self, tool_body_id: int):
        """Add an end-effector tool to the motion planner's collision checker."""
        planner = self.create_motion_planner()
        return planner.add_end_effector_tool(tool_body_id)
    
    def switch_to_tool_planning(self, tool):
        """
        Switch the motion planner to use tool-centric planning.
        
        Args:
            tool: EndeffectorTool instance to switch to
        """
        planner = self.create_motion_planner()
        planner.change_subject(tool)
        return planner
    
    # Add methods to RobotBase
    from pybullet_industrial.robot_base import RobotBase
    RobotBase.create_motion_planner = create_motion_planner
    RobotBase.plan_to_joints = plan_to_joints
    RobotBase.plan_to_pose = plan_to_pose
    RobotBase.plan_to_position = plan_to_position
    RobotBase.plan = plan
    RobotBase.execute_planned_path = execute_planned_path
    RobotBase.add_end_effector_tool_to_planner = add_end_effector_tool_to_planner
    RobotBase.switch_to_tool_planning = switch_to_tool_planning


# Auto-extend RobotBase when module is imported
if OMPL_AVAILABLE:
    add_motion_planning_to_robot_base() 