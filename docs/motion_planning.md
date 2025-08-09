# Motion Planning in PyBullet Industrial

This document describes the motion planning functionality that has been integrated into the PyBullet Industrial package.

## Overview

The motion planning module provides OMPL-based motion planning capabilities for industrial robots. It integrates seamlessly with the existing `RobotBase` and `CollisionChecker` classes.
Important: for some reason if gravity and timestep are set, there are some problems with the end effector (starts floating around).

## Features

- **Multiple Planning Algorithms**: RRT, RRTstar, BITstar, RRTConnect, PRM, EST, FMT, InformedRRTstar
- **Flexible Goal Specification**: Support for joint space, end-effector pose, and position-only goals
- **Joint Space Planning**: Plan paths to specific joint configurations
- **Task Space Planning**: Plan paths to end-effector poses or positions
- **Collision Avoidance**: Integrates with the existing collision detection system
- **Path Execution**: Execute planned paths with or without dynamics
- **Seamless Integration**: Works with existing RobotBase and CollisionChecker classes

## Requirements

- OMPL (Open Motion Planning Library) with Python bindings
- PyBullet Industrial package
- NumPy

## Installation

1. Install OMPL with Python bindings:
   ```bash
   # Clone and build OMPL
   git clone https://github.com/ompl/ompl.git
   cd ompl
   mkdir build && cd build
   cmake ../.. -DPYTHON_EXEC=/path/to/your/python
   make -j4 py_ompl
   ```

2. Install PyBullet Industrial:
   ```bash
   cd package/pybullet_industrial/src
   pip install -e .
   ```

## Usage

### Basic Usage

```python
import numpy as np
import pybullet as p
from pybullet_industrial import RobotBase, CollisionChecker, MotionPlanner

# Initialize PyBullet
p.connect(p.GUI)

# Create robot
robot = RobotBase("path/to/robot.urdf", [0, 0, 0], [0, 0, 0, 1], p)

# Create collision checker
collision_checker = CollisionChecker([robot.urdf])

# Create motion planner
planner = MotionPlanner(robot, collision_checker)

# Plan to joint configuration
goal_joints = {'q1': 0.5, 'q2': 0.3, 'q3': -1.0, 'q4': 0.2, 'q5': 0.1, 'q6': 0.8}
success, path = planner.plan(goal_joints)

if success:
    planner.execute_path(path)
```

### Using RobotBase Methods

The motion planning functionality is also available directly on RobotBase instances:

```python
# Plan to joint configuration
success, path = robot.plan_to_joints(goal_joints)

# Plan to end-effector pose
target_position = np.array([0.8, 0.2, 0.5])
target_orientation = np.array([0, 0, 0, 1])  # Quaternion
success, path = robot.plan_to_pose(target_position, target_orientation)

# Execute planned path
robot.execute_planned_path(path)
```

### Advanced Usage

```python
# Create planner with custom settings
planner = MotionPlanner(
    robot, 
    collision_checker, 
    planning_time=10.0,  # Default planning time
    interpolation_steps=1000  # Path interpolation steps
)

# Set different planner
planner.set_planner("RRTstar")

# Plan with custom parameters
success, path = planner.plan(
    goal_joints,
    start_joints=start_joints,  # Optional start configuration
    planning_time=15.0  # Override default planning time
)

# Execute with dynamics
planner.execute_path(path, dynamics=True, step_delay=0.01)
```

## API Reference

### MotionPlanner Class

#### Constructor
```python
MotionPlanner(robot, collision_checker=None, planning_time=5.0, interpolation_steps=500)
```

**Parameters:**
- `robot`: RobotBase instance
- `collision_checker`: CollisionChecker instance (optional)
- `planning_time`: Default planning time in seconds
- `interpolation_steps`: Number of path interpolation steps

#### Methods

##### `set_planner(planner_name)`
Set the motion planning algorithm.

**Parameters:**
- `planner_name`: String name of planner ("RRT", "RRTstar", "BITstar", etc.)

##### `plan(goal, start_joints=None, planning_time=None, goal_type="auto")`
Plan a path to a goal configuration with flexible goal specification.

**Parameters:**
- `goal`: Goal specification. Can be:
  - `Dict[str, float]`: Joint configuration
  - `Tuple[np.ndarray, np.ndarray]`: (position, orientation) for end-effector
  - `np.ndarray`: Position only for end-effector
- `start_joints`: Starting joint configuration (uses current if None)
- `planning_time`: Planning time limit (uses default if None)
- `goal_type`: Type of goal ("joints", "pose", "position", or "auto" for automatic detection)

**Returns:**
- Tuple of (success, path) where path is a list of joint configurations

##### `plan_to_joints(goal_joints, start_joints=None, planning_time=None)`
Plan a path to a joint configuration.

**Parameters:**
- `goal_joints`: Dictionary of joint names to target values
- `start_joints`: Starting joint configuration (uses current if None)
- `planning_time`: Planning time limit (uses default if None)

**Returns:**
- Tuple of (success, path) where path is a list of joint configurations

##### `plan_to_pose(target_position, target_orientation=None, endeffector_name=None, planning_time=None)`
Plan a path to an end-effector pose.

**Parameters:**
- `target_position`: Target position as numpy array
- `target_orientation`: Target orientation as quaternion (optional, uses current if None)
- `endeffector_name`: Name of end-effector (ignored, uses default)
- `planning_time`: Planning time limit (optional)

**Returns:**
- Tuple of (success, path)

##### `plan_to_position(target_position, planning_time=None)`
Plan a path to an end-effector position (keeping current orientation).

**Parameters:**
- `target_position`: Target position as numpy array
- `planning_time`: Planning time limit (optional)

**Returns:**
- Tuple of (success, path)

##### `execute_path(path, dynamics=False, step_delay=0.01)`
Execute a planned path.

**Parameters:**
- `path`: List of joint configurations
- `dynamics`: Whether to use dynamics simulation
- `step_delay`: Delay between simulation steps

##### `get_available_planners()`
Get list of available planner names.

**Returns:**
- List of planner names

### RobotBase Extensions

The following methods are automatically added to RobotBase instances:

#### `create_motion_planner(collision_checker=None, **kwargs)`
Create a motion planner for this robot.

#### `plan(goal, **kwargs)`
Plan to a goal (auto-detects goal type).

#### `plan_to_joints(goal_joints, **kwargs)`
Plan to a joint configuration.

#### `plan_to_pose(target_position, target_orientation=None, endeffector_name=None, **kwargs)`
Plan to an end-effector pose.

#### `plan_to_position(target_position, **kwargs)`
Plan to an end-effector position (keeping current orientation).

#### `execute_planned_path(path, **kwargs)`
Execute a planned path.

## Goal Specification Examples

### Joint Space Goals
```python
# Dictionary of joint values
goal_joints = {
    'q1': 0.5,
    'q2': 0.3,
    'q3': -1.0,
    'q4': 0.2,
    'q5': 0.1,
    'q6': 0.8
}
success, path = planner.plan(goal_joints)
```

### End-Effector Pose Goals
```python
# Tuple of (position, orientation)
target_position = np.array([0.8, 0.2, 0.5])
target_orientation = np.array([0, 0, 0, 1])  # Quaternion
pose_goal = (target_position, target_orientation)
success, path = planner.plan(pose_goal)
```

### End-Effector Position Goals
```python
# Position only (keeps current orientation)
target_position = np.array([0.6, -0.3, 0.4])
success, path = planner.plan(target_position)
```

### Auto-Detection
```python
# The planner automatically detects goal type
planner.plan(goal_joints)      # Detects as joint space
planner.plan(pose_goal)        # Detects as pose
planner.plan(target_position)  # Detects as position
```

## Examples

See `examples/motion_planning_example.py` for a complete working example.

## Troubleshooting

### OMPL Import Error
If you get an import error for OMPL:
1. Ensure OMPL is built with Python bindings
2. Check that the OMPL path is correctly set
3. Verify Python version compatibility

### Planning Failures
- Check that start and goal configurations are within joint limits
- Verify that the robot can reach the goal configuration
- Try increasing planning time
- Check for collisions in the environment

### Performance Issues
- Reduce interpolation steps for faster execution
- Use simpler planners for basic planning tasks
- Consider using dynamics=False for faster path execution

## Integration with Existing Code

The motion planning module is designed to work seamlessly with existing PyBullet Industrial code:

- Uses existing `RobotBase` joint management
- Integrates with `CollisionChecker` collision detection
- Follows the same patterns and conventions
- Maintains backward compatibility

## Future Enhancements

- Path optimization algorithms
- Multi-robot planning
- Task and motion planning integration
- Real-time planning capabilities