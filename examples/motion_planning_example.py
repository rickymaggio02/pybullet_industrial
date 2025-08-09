"""
Motion Planning Example for PyBullet Industrial

This example demonstrates how to use the new motion planning functionality
with the existing pybullet_industrial package.

Usage:
    python motion_planning_example.py [example_name]

Available examples:
    joint          - Joint space planning
    pose           - End-effector pose planning
    position       - End-effector position planning
    auto_joint     - Auto-detection joint goal
    auto_pose      - Auto-detection pose goal
    auto_position  - Auto-detection position goal
    robotbase      - RobotBase method examples
    rrtstar        - RRTstar planner example
    tool           - End-effector tool planning example
    switching      - Switch between robot and tool planning
    all            - Run all examples (default)
    debug          - Run with extra debugging
"""

import numpy as np
import pybullet as p
import pybullet_data
import sys
import os
import time
import argparse

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from pybullet_industrial import RobotBase, CollisionChecker, MotionPlanner


def debug_robot_state(robot, state_name="Current"):
    """Debug robot state and configuration."""
    print(f"\n=== {state_name} Robot State Debug ===")
    
    # Get joint information
    joint_names, joint_indices = robot.get_moveable_joints()
    print(f"Moveable joints: {list(zip(joint_names, joint_indices))}")
    
    # Get current joint state
    current_joints = robot.get_joint_position()
    print(f"Current joint values: {current_joints}")
    
    # Get joint limits
    lower_limits, upper_limits = robot.get_joint_limits()
    print(f"Joint limits:")
    for joint_name in joint_names:
        print(f"  {joint_name}: [{lower_limits[joint_name]:.3f}, {upper_limits[joint_name]:.3f}]")
    
    # Check if current state is within limits
    print(f"\nJoint limit validation:")
    within_limits = True
    for joint_name in joint_names:
        value = current_joints[joint_name]
        lower = lower_limits[joint_name]
        upper = upper_limits[joint_name]
        if value < lower or value > upper:
            print(f"  ✗ {joint_name}: {value:.3f} outside limits [{lower:.3f}, {upper:.3f}]")
            within_limits = False
        else:
            print(f"  ✓ {joint_name}: {value:.3f} within limits [{lower:.3f}, {upper:.3f}]")
    
    # Get end-effector pose
    try:
        end_effector_pose = robot.get_endeffector_pose()
        print(f"\nEnd-effector pose:")
        print(f"  Position: {end_effector_pose[0]}")
        print(f"  Orientation: {end_effector_pose[1]}")
    except Exception as e:
        print(f"  Error getting end-effector pose: {e}")
    
    # Check for collisions
    collision_points = p.getContactPoints(robot.urdf)
    if collision_points:
        print(f"\n⚠️  Collisions detected: {len(collision_points)} contact points")
        for i, contact in enumerate(collision_points[:3]):  # Show first 3
            print(f"  Contact {i+1}: Body {contact[1]} <-> Body {contact[2]}")
    else:
        print(f"\n✓ No collisions detected")
    
    return within_limits


def debug_goal_state(robot, goal, goal_type="joints"):
    """Debug goal state validity."""
    print(f"\n=== Goal State Debug ({goal_type}) ===")
    
    if goal_type == "joints":
        # Check joint limits for goal
        joint_names, joint_indices = robot.get_moveable_joints()
        lower_limits, upper_limits = robot.get_joint_limits()
        
        print(f"Goal joint values: {goal}")
        print(f"Joint limit validation:")
        
        within_limits = True
        for joint_name in joint_names:
            if joint_name in goal:
                value = goal[joint_name]
                lower = lower_limits[joint_name]
                upper = upper_limits[joint_name]
                if value < lower or value > upper:
                    print(f"  ✗ {joint_name}: {value:.3f} outside limits [{lower:.3f}, {upper:.3f}]")
                    within_limits = False
                else:
                    print(f"  ✓ {joint_name}: {value:.3f} within limits [{lower:.3f}, {upper:.3f}]")
            else:
                print(f"  ? {joint_name}: Not specified in goal")
        
        return within_limits
    
    elif goal_type == "pose":
        position, orientation = goal
        print(f"Goal position: {position}")
        print(f"Goal orientation: {orientation}")
        
        # Check if position is reasonable
        if np.any(np.abs(position) > 10):
            print(f"⚠️  Warning: Position values seem very large: {position}")
        
        # Check if orientation is normalized
        orientation_norm = np.linalg.norm(orientation)
        if abs(orientation_norm - 1.0) > 0.1:
            print(f"⚠️  Warning: Orientation not normalized: {orientation_norm}")
        
        return True
    
    elif goal_type == "position":
        print(f"Goal position: {goal}")
        
        # Check if position is reasonable
        if np.any(np.abs(goal) > 10):
            print(f"⚠️  Warning: Position values seem very large: {goal}")
        
        return True
    
    return True


def preview_goal_position(robot, goal, goal_type="joints", preview_time=1.0):
    """
    Preview the goal position by moving the robot there, waiting, then returning to start.
    
    Args:
        robot: RobotBase instance
        goal: Goal specification
        goal_type: Type of goal ("joints", "pose", "position", or "auto")
        preview_time: Time to wait at goal position in seconds
    """
    print(f"\n=== Previewing Goal Position ===")
    
    # Store current state
    current_joints = robot.get_joint_position()
    print(f"Current position: {current_joints}")
    
    # Resolve goal to joint configuration
    if goal_type == "joints":
        goal_joints = goal
        # For joint goals, use reset_joint_position
        print("Moving to goal position for preview (joint space)...")
        robot.reset_joint_position(goal_joints)
        
        # Wait at goal position
        print(f"Waiting {preview_time} seconds at goal position...")
        time.sleep(preview_time)
        
        # Move back to start position using reset_joint_position
        print("Moving back to start position...")
        robot.reset_joint_position(current_joints)
        
    elif goal_type == "pose":
        position, orientation = goal
        print(f"Target position: {position}")
        print(f"Target orientation: {orientation}")
        try:
            # For pose goals, use reset_endeffector_pose
            print("Moving to goal position for preview (end-effector pose)...")
            robot.reset_endeffector_pose(position, orientation)
            
            # Wait at goal position
            print(f"Waiting {preview_time} seconds at goal position...")
            time.sleep(preview_time)
            
            # Move back to start position using reset_joint_position
            print("Moving back to start position...")
            robot.reset_joint_position(current_joints)
            
        except Exception as e:
            print(f"⚠️  Could not preview pose goal: {e}")
            return
            
    elif goal_type == "position":
        try:
            # For position goals, use reset_endeffector_pose with current orientation
            current_pose = robot.get_endeffector_pose()
            current_orientation = current_pose[1]
            print("Moving to goal position for preview (end-effector position)...")
            robot.reset_endeffector_pose(goal, current_orientation)
            
            # Wait at goal position
            print(f"Waiting {preview_time} seconds at goal position...")
            time.sleep(preview_time)
            
            # Move back to start position using reset_joint_position
            print("Moving back to start position...")
            robot.reset_joint_position(current_joints)
            
        except Exception as e:
            print(f"⚠️  Could not preview position goal: {e}")
            return
            
    else:  # auto
        if isinstance(goal, dict):
            # Joint goal - use reset_joint_position
            goal_joints = goal
            print("Moving to goal position for preview (joint space)...")
            robot.reset_joint_position(goal_joints)
            
            # Wait at goal position
            print(f"Waiting {preview_time} seconds at goal position...")
            time.sleep(preview_time)
            
            # Move back to start position using reset_joint_position
            print("Moving back to start position...")
            robot.reset_joint_position(current_joints)
            
        elif isinstance(goal, (tuple, list)) and len(goal) == 2:
            # Pose goal - use reset_endeffector_pose
            position, orientation = goal
            try:
                print("Moving to goal position for preview (end-effector pose)...")
                robot.reset_endeffector_pose(position, orientation)
                
                # Wait at goal position
                print(f"Waiting {preview_time} seconds at goal position...")
                time.sleep(preview_time)
                
                # Move back to start position using reset_joint_position
                print("Moving back to start position...")
                robot.reset_joint_position(current_joints)
                
            except Exception as e:
                print(f"⚠️  Could not preview pose goal: {e}")
                return
                
        elif isinstance(goal, np.ndarray):
            # Position goal - use reset_endeffector_pose with current orientation
            try:
                current_pose = robot.get_endeffector_pose()
                current_orientation = current_pose[1]
                print("Moving to goal position for preview (end-effector position)...")
                robot.reset_endeffector_pose(goal, current_orientation)
                
                # Wait at goal position
                print(f"Waiting {preview_time} seconds at goal position...")
                time.sleep(preview_time)
                
                # Move back to start position using reset_joint_position
                print("Moving back to start position...")
                robot.reset_joint_position(current_joints)
                
            except Exception as e:
                print(f"⚠️  Could not preview position goal: {e}")
                return
        else:
            print(f"⚠️  Unknown goal type for preview: {type(goal)}")
            return
    
    print("Goal preview completed!")


def validate_and_reset_start_state(robot, test_name="Planning Test"):
    """
    Validate the current robot state and reset to a valid state if needed.
    
    Args:
        robot: RobotBase instance
        test_name: Name of the test for logging
        
    Returns:
        bool: True if start state is valid, False otherwise
    """
    print(f"\n=== {test_name} - Start State Validation ===")
    
    # Check current state
    current_valid = debug_robot_state(robot, "Current")
    
    if not current_valid:
        print(f"⚠️  Current state is invalid, attempting to reset...")
        
        # Try to reset to a safe state
        try:
            # Reset to all zeros (home position)
            safe_joints = {joint_name: 0.0 for joint_name, _ in robot.get_moveable_joints()}
            robot.set_joint_position(safe_joints)
            
            # Check if reset was successful
            reset_valid = debug_robot_state(robot, "Reset")
            
            if reset_valid:
                print(f"✓ Successfully reset to valid start state")
                return True
            else:
                print(f"✗ Failed to reset to valid state")
                return False
                
        except Exception as e:
            print(f"✗ Error during reset: {e}")
            return False
    else:
        print(f"✓ Current start state is valid")
        return True


def debug_planning_attempt(planner, goal, goal_type="auto", planning_time=5.0, preview_goal=True, test_name="Planning"):
    """Debug a planning attempt with detailed information."""
    print(f"\n=== {test_name} Debug ===")
    print(f"Goal type: {goal_type}")
    print(f"Goal: {goal}")
    print(f"Planning time: {planning_time}s")
    
    # Validate and reset start state if needed
    start_valid = validate_and_reset_start_state(planner.robot, test_name)
    if not start_valid:
        print(f"✗ Cannot proceed with planning - invalid start state")
        return False, []
    
    # Preview goal position if requested
    if preview_goal:
        preview_goal_position(planner.robot, goal, goal_type)
    
    # Check if OMPL is available
    if not hasattr(planner, 'simple_setup'):
        print("✗ OMPL not properly initialized")
        return False, []
    print("OMPL initialized")
    
    # Get current robot state
    print("Getting current robot state")
    current_joints = planner.robot.get_joint_position()
    print(f"Start joints: {current_joints}")
    
    # Try to resolve goal
    try:
        goal_joints = planner._resolve_goal(goal, goal_type)
        if goal_joints is None:
            print("✗ Failed to resolve goal to joint configuration")
            return False, []
        print(f"Resolved goal joints: {goal_joints}")
    except Exception as e:
        print(f"✗ Error resolving goal: {e}")
        return False, []
    
    # Check if resolved goal is valid
    debug_goal_state(planner.robot, goal_joints, "joints")
    
    # Attempt planning
    print(f"\nStarting planning...")
    start_time = time.time()
    
    try:
        success, path = planner.plan(goal, planning_time=planning_time, goal_type=goal_type)
        planning_time_actual = time.time() - start_time
        
        print(f"Planning completed in {planning_time_actual:.2f}s")
        
        if success:
            print(f"✓ Planning successful!")
            print(f"  Path length: {len(path)} waypoints")
            print(f"  First waypoint: {path[0] if path else 'None'}")
            print(f"  Last waypoint: {path[-1] if path else 'None'}")
            
            # Validate path
            print(f"\nPath validation:")
            valid_path = True
            for i, waypoint in enumerate(path):
                if not debug_goal_state(planner.robot, waypoint, "joints"):
                    print(f"  ✗ Waypoint {i} has invalid joint values")
                    valid_path = False
                    break
                if i > 0:  # Skip first waypoint
                    # Check for large joint changes
                    prev_waypoint = path[i-1]
                    for joint_name in waypoint:
                        if joint_name in prev_waypoint:
                            change = abs(waypoint[joint_name] - prev_waypoint[joint_name])
                            if change > np.pi:  # Large joint change
                                print(f"  ⚠️  Large joint change at waypoint {i}: {joint_name} = {change:.3f}")
            
            if valid_path:
                print(f"  ✓ Path validation passed")
            
        else:
            print(f"✗ Planning failed!")
            
            # Try to diagnose the issue
            print(f"\nDiagnosing planning failure:")
            
            # Check if start state is valid
            print(f"Checking start state validity...")
            start_valid = debug_robot_state(planner.robot, "Start")
            
            # Check if goal state is reachable
            print(f"Checking goal state reachability...")
            goal_valid = debug_goal_state(planner.robot, goal_joints, "joints")
            
            if not start_valid:
                print(f"✗ Start state is invalid")
            if not goal_valid:
                print(f"✗ Goal state is invalid")
            if start_valid and goal_valid:
                print(f"✓ Both start and goal states are valid")
                print(f"  Possible issues: collision, planning time too short, or no path exists")
        
        return success, path
        
    except Exception as e:
        print(f"✗ Planning error: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def run_joint_planning_example(planner, robot):
    """Run joint space planning example."""
    print("\n" + "="*60)
    print("JOINT SPACE PLANNING EXAMPLE")
    print("="*60)
    
    # Define goal joint configuration
    goal_joints = {
        'q1': 0.5,
        'q2': 0.3,
        'q3': -1.0,
        'q4': 0.2,
        'q5': 0.1,
        'q6': 0.8
    }
    
    print(f"Goal joint configuration: {goal_joints}")
    
    # Debug goal state
    debug_goal_state(robot, goal_joints, "joints")
    
    # Plan to joint configuration with debugging
    success, path = debug_planning_attempt(planner, goal_joints, "joints", 10.0, preview_goal=True, test_name="Joint Space Planning")
    
    if success:
        print(f"Planning successful! Path has {len(path)} waypoints.")
        
        # Execute the path
        print("\n=== Executing Planned Path ===")
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        
        print("Path execution completed!")
    else:
        print("Planning failed!")
    
    return success


def run_pose_planning_example(planner, robot):
    """Run end-effector pose planning example."""
    print("\n" + "="*60)
    print("END-EFFECTOR POSE PLANNING EXAMPLE")
    print("="*60)
    
    # Define target pose
    target_position = np.array([0.8, 0.2, 0.5])
    target_orientation = np.array([0, 0, 0, 1])  # Identity quaternion
    
    print(f"Target position: {target_position}")
    print(f"Target orientation: {target_orientation}")
    
    debug_goal_state(robot, (target_position, target_orientation), "pose")
    success, path = debug_planning_attempt(planner, (target_position, target_orientation), "pose", preview_goal=True, test_name="Pose Planning")
    
    if success:
        print(f"Pose planning successful! Path has {len(path)} waypoints.")
        
        # Execute the path
        print("\n=== Executing Pose Path ===")
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        
        print("Pose path execution completed!")
    else:
        print("Pose planning failed!")
    
    return success


def run_position_planning_example(planner, robot):
    """Run end-effector position planning example."""
    print("\n" + "="*60)
    print("END-EFFECTOR POSITION PLANNING EXAMPLE")
    print("="*60)
    
    # Define target position (keeps current orientation)
    target_position_only = np.array([0.6, -0.3, 0.4])
    
    print(f"Target position: {target_position_only}")
    print("(Keeping current orientation)")
    
    debug_goal_state(robot, target_position_only, "position")
    success, path = debug_planning_attempt(planner, target_position_only, "position", preview_goal=True, test_name="Position Planning")
    
    if success:
        print(f"Position planning successful! Path has {len(path)} waypoints.")
        
        # Execute the path
        print("\n=== Executing Position Path ===")
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        
        print("Position path execution completed!")
    else:
        print("Position planning failed!")
    
    return success


def run_auto_detection_examples(planner, robot):
    """Run auto-detection examples."""
    print("\n" + "="*60)
    print("AUTO-DETECTION EXAMPLES")
    print("="*60)
    
    results = {}
    
    # Example 1: Joint space goal (dictionary)
    print("\n--- Auto-Detection Joint Goal ---")
    joint_goal = {
        'q1': -0.2,
        'q2': 0.4,
        'q3': -0.6,
        'q4': 0.1,
        'q5': -0.3,
        'q6': 0.5
    }
    success, path = debug_planning_attempt(planner, joint_goal, "auto", 5.0, preview_goal=True, test_name="Auto-Detection Joint Goal")
    results['auto_joint'] = success
    print(f"Joint goal planning: {'Success' if success else 'Failed'}")
    
    # Example 2: Pose goal (tuple of position and orientation)
    print("\n--- Auto-Detection Pose Goal ---")
    pose_goal = (np.array([0.7, 0.1, 0.6]), np.array([0, 0, 0, 1]))
    success, path = debug_planning_attempt(planner, pose_goal, "auto", 5.0, preview_goal=True, test_name="Auto-Detection Pose Goal")
    results['auto_pose'] = success
    print(f"Pose goal planning: {'Success' if success else 'Failed'}")
    
    # Example 3: Position goal (numpy array)
    print("\n--- Auto-Detection Position Goal ---")
    position_goal = np.array([0.5, -0.2, 0.3])
    success, path = debug_planning_attempt(planner, position_goal, "auto", 5.0, preview_goal=True, test_name="Auto-Detection Position Goal")
    results['auto_position'] = success
    print(f"Position goal planning: {'Success' if success else 'Failed'}")
    
    return results


def run_robotbase_examples(robot):
    """Run RobotBase method examples."""
    print("\n" + "="*60)
    print("ROBOTBASE METHOD EXAMPLES")
    print("="*60)
    
    results = {}
    
    # Create a different goal
    new_goal = {
        'q1': -0.3,
        'q2': 0.5,
        'q3': -0.8,
        'q4': -0.2,
        'q5': 0.4,
        'q6': -0.1
    }
    
    # Validate start state for RobotBase planning
    validate_and_reset_start_state(robot, "RobotBase Joint Planning")
    
    # Preview goal for RobotBase planning
    preview_goal_position(robot, new_goal, "joints")
    
    success, path = robot.plan_to_joints(new_goal, planning_time=8.0)
    results['robotbase_joint'] = success
    
    if success:
        print(f"RobotBase joint planning successful! Path has {len(path)} waypoints.")
        robot.execute_planned_path(path, step_delay=0.02)
        print("RobotBase joint path execution completed!")
    else:
        print("RobotBase joint planning failed!")
    
    # Try position planning with RobotBase
    target_pos = np.array([0.4, 0.3, 0.7])
    
    # Validate start state for RobotBase position planning
    validate_and_reset_start_state(robot, "RobotBase Position Planning")
    
    preview_goal_position(robot, target_pos, "position")
    
    success, path = robot.plan_to_position(target_pos, planning_time=5.0)
    results['robotbase_position'] = success
    
    if success:
        print(f"RobotBase position planning successful! Path has {len(path)} waypoints.")
        robot.execute_planned_path(path, step_delay=0.02)
        print("RobotBase position path execution completed!")
    else:
        print("RobotBase position planning failed!")
    
    # Try flexible planning with RobotBase
    flexible_goal = (np.array([0.3, -0.4, 0.5]), np.array([0, 0, 0, 1]))
    
    # Validate start state for RobotBase flexible planning
    validate_and_reset_start_state(robot, "RobotBase Flexible Planning")
    
    preview_goal_position(robot, flexible_goal, "pose")
    
    success, path = robot.plan(flexible_goal, planning_time=5.0)
    results['robotbase_flexible'] = success
    
    if success:
        print(f"RobotBase flexible planning successful! Path has {len(path)} waypoints.")
        robot.execute_planned_path(path, step_delay=0.02)
        print("RobotBase flexible path execution completed!")
    else:
        print("RobotBase flexible planning failed!")
    
    return results


def run_rrtstar_example(planner, robot):
    """Run RRTstar planner example."""
    print("\n" + "="*60)
    print("RRTSTAR PLANNER EXAMPLE")
    print("="*60)
    
    # Show available planners
    print(f"Available planners: {planner.get_available_planners()}")
    
    # Set planner to RRTstar
    planner.set_planner("RRTstar")
    print("Set planner to RRTstar")
    
    test_goal = {
        'q1': 0.2,
        'q2': -0.3,
        'q3': -0.5,
        'q4': 0.1,
        'q5': -0.2,
        'q6': 0.3
    }
    
    success, path = debug_planning_attempt(planner, test_goal, "joints", 5.0, preview_goal=True, test_name="RRTstar Planning")
    
    if success:
        print(f"RRTstar planning successful! Path has {len(path)} waypoints.")
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        print("RRTstar path execution completed!")
    else:
        print("RRTstar planning failed!")
    
    return success


def run_tool_planning_example(planner, robot):
    """Run example demonstrating motion planning with an attached end-effector tool."""

    """FIXED: The tool floating issue was caused by the collision checker ignoring
    collisions between the robot and tool, which interfered with PyBullet's physical
    coupling constraints. The solution is to:
    
    1. Configure tool collision checking to preserve PyBullet physics
    2. NOT ignore robot-tool collisions in the collision checker
    3. Let PyBullet handle the physical coupling naturally
    
    This ensures the tool stays properly attached to the robot while still being
    included in collision checking for external obstacles.
    """
    print("\n" + "="*60)
    print("END-EFFECTOR TOOL PLANNING EXAMPLE")
    print("="*60)
    
    # Configure tool collision checking to preserve PyBullet physics
    if planner is not None:
        print("1. Configuring tool collision checking...")
        planner.configure_tool_collision_checking(
            enable_tool_collision_checking=True,
            ignore_robot_tool_collisions=False  # This prevents tools from floating!
        )
        print("✓ Tool collision checking configured to preserve PyBullet physics")
    
    # Create and couple a tool
    print("\n2. Creating and coupling tool...")
    
    # Try to load the 3D printing head URDF if available
    tool_urdf = os.path.join(os.path.dirname(__file__), 'robot_descriptions/3d_printing_head.urdf')
    use_endeffector_tool = False
    tool = None
    
    if os.path.exists(tool_urdf):
        print(f"Loading tool URDF: {tool_urdf}")
        try:
            # Create Extruder instance using pybullet_industrial
            tool_position = [1.1, 0, 2.2]
            target_position = np.array([0.5, 0, 3.2])
            tool_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            # Import here to avoid circular imports
            import pybullet_industrial as pi
            from pybullet_industrial import Extruder, Plastic
            
            extruder_properties = {
                'maximum distance': 0.5,
                'opening angle': 0,
                'material': Plastic,
                'number of rays': 1
            }
            
            tool = Extruder(
                tool_urdf, tool_position, tool_orientation, p, extruder_properties)
            tool.couple(robot, 'printing_coupling_frame')
            use_endeffector_tool = True

            robot.reset_endeffector_pose(target_position, tool_orientation)
            tool.set_tool_pose(target_position, tool_orientation)
            for _ in range(50):
                p.stepSimulation()
            
            print(f"✓ Extruder tool coupled! Tool body ID: {tool.urdf}")
            
        except Exception as e:
            print(f"Failed to load Extruder: {e}")
            print("Falling back to simple tool...")
            use_endeffector_tool = False
    
    if not use_endeffector_tool:
        # Create a simple tool as fallback
        print("Creating simple tool as fallback...")
        
        col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2])
        vis_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2], 
                                       rgbaColor=[0.2, 0.8, 0.2, 1])
        tool_body_id = p.createMultiBody(baseMass=0.1, 
                                        baseCollisionShapeIndex=col_box_id,
                                        baseVisualShapeIndex=vis_box_id,
                                        basePosition=[0, 0, 0.5])
        
        # Attach tool to robot
        robot_end_effector_id = robot._default_endeffector_id
        constraint_id = p.createConstraint(robot.urdf, robot_end_effector_id,
                                         tool_body_id, -1,
                                         p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
        
        # Add tool to collision checker using the new method
        if planner is not None:
            planner.add_end_effector_tool(tool_body_id)
        
        print(f"✓ Simple tool created and attached! Tool body ID: {tool_body_id}")
    
    # Switch to tool-centric planning
    print("\n3. Switching to tool-centric planning...")
    
    if use_endeffector_tool and tool is not None:
        # Use the Extruder tool
        planner.change_subject(tool)
        print("✓ Motion planner now using Extruder tool")
    else:
        # Create a simple tool object for the simple tool
        class SimpleTool:
            def __init__(self, body_id, robot):
                self.urdf = body_id
                self._coupled_robot = robot
            
            def is_coupled(self):
                return True
            
            def get_tool_pose(self):
                pos, ori = p.getBasePositionAndOrientation(self.urdf)
                return np.array(pos), np.array(ori)
            
            def set_tool_pose(self, position, orientation):
                # For simple tool, we'll use robot's set_endeffector_pose
                self._coupled_robot.set_endeffector_pose(position, orientation)
        
        simple_tool = SimpleTool(tool_body_id, robot)
        planner.change_subject(simple_tool)
        print("✓ Motion planner now using SimpleTool")
    
    # Now plan using tool-centric approach
    print("\n4. Planning with tool-centric approach...")
    
    # Get current tool pose
    if use_endeffector_tool and tool is not None:
        current_tool_pos, current_tool_ori = tool.get_tool_pose()
    else:
        current_tool_pos, current_tool_ori = simple_tool.get_tool_pose()
    
    # Define target tool pose
    target_tool_pos = np.array([1.2, 0.3, 0.8])
    target_tool_ori = current_tool_ori  # Keep current orientation
    
    print(f"Current tool position: {current_tool_pos}")
    print(f"Target tool position: {target_tool_pos}")
    
    # Plan to tool position using tool-centric planning
    success, path = planner.plan_to_tool_position(target_tool_pos, planning_time=8.0)
    
    if success:
        print("✓ Tool planning successful!")
        print(f"  Path type: {type(path[0])} (tool poses)")
        print(f"  Path length: {len(path)} waypoints")
        
        # Execute the tool-centric path
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        
        # Verify tool reached target
        if use_endeffector_tool and tool is not None:
            final_tool_pos, _ = tool.get_tool_pose()
        else:
            final_tool_pos, _ = simple_tool.get_tool_pose()
        
        print(f"Final tool position: {final_tool_pos}")
        print(f"Target reached: {np.allclose(final_tool_pos, target_tool_pos, atol=0.1)}")
    else:
        print("✗ Tool planning failed!")
    
    # Clean up
    if use_endeffector_tool and tool is not None:
        tool.decouple()
        p.removeBody(tool.urdf)
        time.sleep(1)
    else:
        p.removeBody(tool_body_id)
        time.sleep(1)
    
    return success


def run_switching_planning_example(planner, robot):
    """Run example demonstrating switching between robot and tool planning."""
    print("\n" + "="*60)
    print("SWITCHING BETWEEN ROBOT AND TOOL PLANNING")
    print("="*60)
    
    # First, plan using robot-centric approach
    print("1. Planning with robot-centric approach...")
    
    goal_joints = {
        'q1': 0.5,
        'q2': 0.3,
        'q3': -0.8,
        'q4': 0.1,
        'q5': -0.2,
        'q6': 0.4
    }
    
    success, path = debug_planning_attempt(planner, goal_joints, "joints", 5.0, 
                                         preview_goal=False, test_name="Robot Planning")
    
    if success:
        print("✓ Robot planning successful!")
        print(f"  Path type: {type(path[0])} (joint configurations)")
        print(f"  Path length: {len(path)} waypoints")
        planner.execute_path(path, dynamics=False, step_delay=0.02)
    
    # Now create and couple a tool
    print("\n2. Creating and coupling tool...")
    
    # Create a simple tool
    col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2])
    vis_box_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.2], 
                                   rgbaColor=[0.2, 0.8, 0.2, 1])
    tool_body_id = p.createMultiBody(baseMass=0.1, 
                                    baseCollisionShapeIndex=col_box_id,
                                    baseVisualShapeIndex=vis_box_id,
                                    basePosition=[0, 0, 0.5])
    
    # Attach tool to robot
    robot_end_effector_id = robot._default_endeffector_id
    constraint_id = p.createConstraint(robot.urdf, robot_end_effector_id,
                                     tool_body_id, -1,
                                     p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
    
    # Switch to tool-centric planning
    print("3. Switching to tool-centric planning...")
    
    # Create a simple tool object for demonstration
    class SimpleTool:
        def __init__(self, body_id, robot):
            self.urdf = body_id
            self._coupled_robot = robot
        
        def is_coupled(self):
            return True
        
        def get_tool_pose(self):
            pos, ori = p.getBasePositionAndOrientation(self.urdf)
            return np.array(pos), np.array(ori)
        
        def set_tool_pose(self, position, orientation):
            # For simple tool, we'll use robot's set_endeffector_pose
            self._coupled_robot.set_endeffector_pose(position, orientation)
    
    simple_tool = SimpleTool(tool_body_id, robot)
    
    # Change the planner's subject to the tool
    planner.change_subject(simple_tool)
    
    # Now plan using tool-centric approach
    print("4. Planning with tool-centric approach...")
    
    # Get current tool pose
    current_tool_pos, current_tool_ori = simple_tool.get_tool_pose()
    
    # Define target tool pose
    target_tool_pos = np.array([1.2, 0.3, 0.8])
    target_tool_ori = current_tool_ori  # Keep current orientation
    
    print(f"Current tool position: {current_tool_pos}")
    print(f"Target tool position: {target_tool_pos}")
    
    # Plan to tool position using tool-centric planning
    success, path = planner.plan_to_tool_position(target_tool_pos, planning_time=8.0)
    
    if success:
        print("✓ Tool planning successful!")
        print(f"  Path type: {type(path[0])} (tool poses)")
        print(f"  Path length: {len(path)} waypoints")
        print(f"  First waypoint: position={path[0][0]}, orientation={path[0][1]}")
        
        # Execute the tool-centric path
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        
        # Verify tool reached target
        final_tool_pos, _ = simple_tool.get_tool_pose()
        print(f"Final tool position: {final_tool_pos}")
        print(f"Target reached: {np.allclose(final_tool_pos, target_tool_pos, atol=0.1)}")
    else:
        print("✗ Tool planning failed!")
    
    # Test tool pose planning as well
    print("\n5. Testing tool pose planning...")
    
    # Define a new target with different orientation
    target_tool_pos_2 = np.array([1.0, -0.2, 0.6])
    target_tool_ori_2 = p.getQuaternionFromEuler([0, np.pi/4, 0])  # 45 degree rotation around Y
    
    print(f"Target tool pose: position={target_tool_pos_2}, orientation={target_tool_ori_2}")
    
    success, path = planner.plan_to_tool_pose(target_tool_pos_2, target_tool_ori_2, planning_time=8.0)
    
    if success:
        print("✓ Tool pose planning successful!")
        print(f"  Path length: {len(path)} waypoints")
        
        # Execute the tool pose path
        planner.execute_path(path, dynamics=False, step_delay=0.02)
        
        # Verify final pose
        final_tool_pos, final_tool_ori = simple_tool.get_tool_pose()
        print(f"Final tool pose: position={final_tool_pos}, orientation={final_tool_ori}")
        print(f"Position reached: {np.allclose(final_tool_pos, target_tool_pos_2, atol=0.1)}")
    
    # Switch back to robot-centric planning
    print("\n6. Switching back to robot-centric planning...")
    planner.change_subject(robot)
    
    # Plan one more time with robot
    goal_joints_2 = {
        'q1': -0.3,
        'q2': 0.2,
        'q3': -0.5,
        'q4': 0.0,
        'q5': 0.1,
        'q6': -0.2
    }
    
    success, path = debug_planning_attempt(planner, goal_joints_2, "joints", 5.0, 
                                         preview_goal=False, test_name="Robot Planning (After Switch)")
    
    if success:
        print("✓ Robot planning after switch successful!")
        print(f"  Path type: {type(path[0])} (joint configurations)")
        print(f"  Path length: {len(path)} waypoints")
        planner.execute_path(path, dynamics=False, step_delay=0.02)
    
    # Clean up
    p.removeBody(tool_body_id)
    
    return success


def setup_environment():
    """Setup PyBullet environment and robot."""
    # Initialize PyBullet
    p.connect(p.GUI)
    # p.setGravity(0, 0, -9.81)
    # p.setTimeStep(1./240.)
    p.setPhysicsEngineParameter(numSolverIterations=5000)
    
    # Load ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    
    # Load a robot (using Comau as example)
    robot_urdf = "/home/rickymaggio/Documents/experiment/test_simulation/XeleritCoderBackend/package/pybullet_industrial/examples/robot_descriptions/comau_nj290_robot.urdf"
    
    if not os.path.exists(robot_urdf):
        print(f"Robot URDF not found: {robot_urdf}")
        print("Please update the path to your robot URDF file.")
        return None, None, None
    
    # Create robot instance
    start_position = np.array([0, 0, 0])
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot = RobotBase(robot_urdf, start_position, start_orientation, p)
    
    # Debug initial robot state
    debug_robot_state(robot, "Initial")
    
    # Create collision checker
    collision_checker = CollisionChecker([robot.urdf])

    # Create motion planner
    try:
        planner = MotionPlanner(robot, collision_checker, planning_time=5.0)
        print("Motion planner created successfully!")
    except ImportError as e:
        print(f"Could not create motion planner: {e}")
        print("Please ensure OMPL is installed with Python bindings.")
        return None, None, None
    
    # Add some obstacles
    obstacle_pos = [1.0, 0.5, 0.5]
    obstacle_size = [0.3, 0.3, 0.3]
    col_box_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_size)
    obstacle_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_box_id, basePosition=obstacle_pos)
    
    # Update collision checker with new obstacle
    collision_checker.add_body_id(obstacle_id)
    
    return robot, planner, collision_checker


def main():
    """Main example function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Motion Planning Examples for PyBullet Industrial')
    parser.add_argument('example', nargs='?', default='all', 
                       choices=['joint', 'pose', 'position', 'auto_joint', 'auto_pose', 'auto_position', 
                               'robotbase', 'rrtstar', 'all', 'debug', 'tool', 'switching'],
                       help='Example to run (default: all)')
    parser.add_argument('--no-preview', action='store_true', 
                       help='Disable goal position preview')
    parser.add_argument('--planning-time', type=float, default=5.0,
                       help='Planning time limit in seconds (default: 5.0)')
    
    args = parser.parse_args()
    
    print(f"Running example: {args.example}")
    print(f"Planning time: {args.planning_time}s")
    print(f"Goal preview: {'Disabled' if args.no_preview else 'Enabled'}")
    
    # Setup environment
    robot, planner, collision_checker = setup_environment()
    if robot is None:
        return
    
    # Get current joint state
    current_joints = robot.get_joint_position()
    print(f"Current joint state: {current_joints}")
    
    results = {}
    
    try:
        if args.example == 'joint' or args.example == 'all':
            results['joint'] = run_joint_planning_example(planner, robot)
        
        if args.example == 'pose' or args.example == 'all':
            results['pose'] = run_pose_planning_example(planner, robot)
        
        if args.example == 'position' or args.example == 'all':
            results['position'] = run_position_planning_example(planner, robot)
        
        if args.example == 'auto_joint' or args.example == 'auto_pose' or args.example == 'auto_position' or args.example == 'all':
            auto_results = run_auto_detection_examples(planner, robot)
            results.update(auto_results)
        
        if args.example == 'robotbase' or args.example == 'all':
            robotbase_results = run_robotbase_examples(robot)
            results.update(robotbase_results)
        
        if args.example == 'rrtstar' or args.example == 'all':
            results['rrtstar'] = run_rrtstar_example(planner, robot)
        
        if args.example == 'tool' or args.example == 'all':
            results['tool'] = run_tool_planning_example(planner, robot)
        
        if args.example == 'switching' or args.example == 'all':
            results['switching'] = run_switching_planning_example(planner, robot)
        
        if args.example == 'debug':
            # Run all examples with extra debugging
            print("\n" + "="*60)
            print("DEBUG MODE - RUNNING ALL EXAMPLES WITH EXTRA DEBUGGING")
            print("="*60)
            results['joint'] = run_joint_planning_example(planner, robot)
            results['pose'] = run_pose_planning_example(planner, robot)
            results['position'] = run_position_planning_example(planner, robot)
            auto_results = run_auto_detection_examples(planner, robot)
            results.update(auto_results)
            robotbase_results = run_robotbase_examples(robot)
            results.update(robotbase_results)
            results['rrtstar'] = run_rrtstar_example(planner, robot)
            results['tool'] = run_tool_planning_example(planner, robot)
            results['switching'] = run_switching_planning_example(planner, robot)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for test_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{test_name:20} : {status}")
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        print(f"\nOverall: {success_count}/{total_count} tests passed")
        
    except KeyboardInterrupt:
        print("\n\nExample interrupted by user")
    except Exception as e:
        print(f"\n\nError during example execution: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Example Completed ===")
    print("Press Enter to exit...")
    input()


if __name__ == "__main__":
    main() 