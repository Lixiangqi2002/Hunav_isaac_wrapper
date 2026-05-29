#!/usr/bin/env python3
"""
teleop_hunav_sim.py

Contains the TeleopHuNavSim class which combines:
- ROS 2 teleoperation for a differential robot.
- World loading via WorldBuilder.
- Agent management via HuNavManager.
"""
from isaacsim import SimulationApp

# Start Isaac Sim
CONFIG = {
    "width": 640,
    "height": 480,
    "sync_loads": True,
    "headless": False,
    "renderer": "RaytracedLighting",
}
simulation_app = SimulationApp(CONFIG)

import os
import re
import signal
import subprocess
import json
from pathlib import Path
import numpy as np
try:
    from plyfile import PlyData
except ImportError:
    PlyData = None
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from isaacsim.core.api import World
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
    DifferentialController,
)
import omni
import omni.graph.core as og
import omni.timeline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux
try:
    from omni.isaac.core.utils.semantics import add_update_semantics
except ImportError:
    try:
        from isaacsim.core.utils.semantics import add_update_semantics
    except ImportError:
        add_update_semantics = None
 
# Import the WorldBuilder and HuNavManager modules.
from .world_builder import WorldBuilder
from .hunav_manager import HuNavManager

scene = "hospital"
root = "/data/code/hunavsim_docker"

DEFAULT_CHOIS_ANIMATED_USD = (
    "/workspace/hunav_isaac_ws/src/animation_usd/sample_motion_001_with_cache.usd"
)
DEFAULT_CHOIS_ANIMATED_PRIM_PATH = "/World/CHOISActors/SampleMotion001"
DEFAULT_CHOIS_ANIMATED_USD_LOCAL = (
    "isaac_sim/hunav_isaac_ws/src/"
    f"{root}/animation_usd/sample_motion_001_with_cache.usd"
)
DEFAULT_CHOIS_ANIMATION_USD_ROOT = f"/workspace/hunav_isaac_ws/src/animation_usd/{scene}"
DEFAULT_CHOIS_ANIMATION_USD_ROOT_LOCAL = (
    f"{root}/isaac_sim/hunav_isaac_ws/src/animation_usd/{scene}"
)
DEFAULT_CHOIS_WAYPOINTS_NPY = (
    "/workspace/hunav_isaac_ws/src/animation_npy/chois_waypoints.npy"
)
DEFAULT_CHOIS_WAYPOINTS_NPY_LOCAL = (
    f"{root}/isaac_sim/hunav_isaac_ws/src/"
    "animation_npy/chois_waypoints.npy"
)
DEFAULT_CHOIS_ANIMATION_NPY_ROOT = f"/workspace/hunav_isaac_ws/src/animation_npy/{scene}"
DEFAULT_CHOIS_ANIMATION_NPY_ROOT_LOCAL = (
    f"{root}/isaac_sim/hunav_isaac_ws/src/animation_npy/{scene}"
)
DEFAULT_CLICKED_POINTS_NPY = "/workspace/hunav_isaac_ws/src/animation_usd/rviz_clicked_points.npy"
DEFAULT_GOAL_POSES_NPY = "/workspace/hunav_isaac_ws/src/animation_usd/rviz_goal_poses.npy"
DEFAULT_RECORD_POSE_TOPIC = "/hunav_record_pose"
DEFAULT_CHOIS_ACTOR_Z = 0.0
DEFAULT_CHOIS_DEBUG_ROOT = (
    f"{root}/isaac_sim/hunav_isaac_ws/src/animation_usd/debug"
)
DEFAULT_CHOIS_OUTPUT_ROOT = f"{root}/chois_bridge/outputs/chois_output"
DEFAULT_CHOIS_DEBUG_VISUALIZATION = False
DEFAULT_GT_VISIBLE_OBJECTS_TOPIC = "/gt_visible_objects"
DEFAULT_GT_VISIBLE_OBJECTS_FRONT_TOPIC = "/gt_visible_objects/front_left"
DEFAULT_GT_VISIBLE_OBJECTS_REAR_TOPIC = "/gt_visible_objects/rear_left"
DEFAULT_GT_CAMERA_PRIM_HINT = "sim_camera"
DEFAULT_GT_IMAGE_WIDTH = 640
DEFAULT_GT_IMAGE_HEIGHT = 480
DEFAULT_GT_PUBLISH_EVERY_N_STEPS = 2
DEFAULT_GT_OCCLUSION_OVERLAP_THRESHOLD = 0.7
DEFAULT_GT_OCCLUSION_DEPTH_MARGIN = 0.2
DEFAULT_CHOIS_STATE_TOPIC = "/chois/state"
DEFAULT_CHOIS_HUMAN_POSES_TOPIC = "/chois/human_poses"
DEFAULT_CHOIS_OBJECT_POSES_TOPIC = "/chois/object_poses"
DEFAULT_CHOIS_MARKERS_TOPIC = "/chois/markers"
DEFAULT_CHOIS_FRAME_ID = "map"
DEFAULT_CHOIS_PUBLISH_EVERY_N_STEPS = 1

def find_package_share_directory():
    """
    Find the package share directory containing worlds, scenarios, config, etc.
    Works both in development and installed package modes.
    """
    # Prefer the editable source tree when this module is imported from source.
    # This keeps large USD world tweaks from being shadowed by an older install/.
    current_file = Path(__file__)
    if current_file.parent.parent.name == "src":
        src_dir = current_file.parent.parent
        if (src_dir / "worlds").exists():
            return str(src_dir)

    # Try to find via ROS2 package first (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        share_dir = pkg_path / "share" / "hunav_isaac_wrapper"
        if share_dir.exists() and (share_dir / "worlds").exists():
            return str(share_dir)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Development mode fallback
    # Last fallback - check current working directory
    cwd = Path.cwd()
    if (cwd / "worlds").exists():
        return str(cwd)
    
    # If all else fails, return the old path calculation
    return os.path.dirname(os.path.dirname(__file__))


def find_robot_config_path(filename):
    """
    Find robot configuration file in development or installed package.
    
    Args:
        filename: Name of the robot config file (e.g., "nova_carter_ros2_sensors.usd")
    
    Returns:
        str: Absolute path to the robot config file
    """
    # Try to find via ROS2 package share directory (installed mode)
    try:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", "hunav_isaac_wrapper"],
            capture_output=True, text=True, check=True
        )
        pkg_path = Path(result.stdout.strip())
        robot_config = pkg_path / "share" / "hunav_isaac_wrapper" / "config" / "robots" / filename
        if robot_config.exists():
            return str(robot_config)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try development mode (relative to this file)
    current_file_dir = Path(__file__).parent
    workspace_root = current_file_dir.parent.parent
    robot_config = workspace_root / "config" / "robots" / filename
    if robot_config.exists():
        return str(robot_config)
    
    # Try alternative development paths
    dev_paths = [
        current_file_dir.parent / "config" / "robots" / filename,
        Path.cwd() / "src" / "config" / "robots" / filename,
        Path.cwd() / "config" / "robots" / filename,
    ]
    
    for path in dev_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(f"Robot config file not found: {filename}")

class TeleopHuNavSim(Node):
    """
    Combines:
    - Differential robot teleop (subscribing to /cmd_vel)
    - USD map loading (via WorldBuilder)
    - Agent management and update (via HuNavManager)
    """

    def __init__(self, map_name, hunav_config, robot_name, animated_usd_path=None):
        super().__init__("hunav_sim")
        self.map_name = map_name
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Assets root
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Nucleus root.")

        # Load USD stage
        self.builder = WorldBuilder(base_path=find_package_share_directory())
        if map_name:
            self.builder.load_map(map_name)

        self.chois_animation_npy_root = self._resolve_chois_animation_npy_root()
        self.chois_animation_usd_root = self._resolve_chois_animation_usd_root()
        self.custom_animated_usd_entries = self._resolve_custom_animated_usd_entries(
            animated_usd_path
        )
        self.custom_animated_actors = []
        self.custom_animated_assets = {}
        self.custom_animated_time_range = None
        self.custom_animation_time_seconds = None
        self.clicked_points = []
        self.goal_poses = []
        self.clicked_points_output_path = self._resolve_clicked_points_output_path()
        self.goal_poses_output_path = self._resolve_goal_poses_output_path()
        self.chois_debug_output_root = self._resolve_chois_debug_output_root()
        self.enable_chois_debug_visualization = self._resolve_chois_debug_visualization()
        self.custom_animated_debug_data = {}
        self.custom_animation_frame_index = 0

        # Create World object
        timestep = 1.0 / 20.0
        self.world = World(
            stage_units_in_meters=1, physics_dt=timestep, rendering_dt=timestep
        )
        self._apply_scene_semantics()
        self._ensure_hospital_lighting()

        if map_name == "empty_world":
            self.world.scene.add_default_ground_plane()

        for animated_entry in self.custom_animated_usd_entries:
            actor_prim = self._spawn_custom_animated_usd(
                animated_entry["usd_path"],
                animated_entry["prim_path"],
                animated_entry["placement"],
            )
            self.custom_animated_actors.append(actor_prim)
            if animated_entry["prim_path"] in self.custom_animated_assets:
                self.custom_animated_assets[animated_entry["prim_path"]]["asset_key"] = (
                    animated_entry["asset_key"]
                )
            if self.enable_chois_debug_visualization:
                self._initialize_custom_animated_debug(actor_prim, animated_entry)

        # Define configuration for each wheeled robot available
        robot_configs = {
            "jetbot": {
                "name": "Jetbot",
                "usd_relative_path": os.path.join(
                    "Isaac", "Robots", "Jetbot", "jetbot.usd"
                ),
                "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
                "wheel_radius": 0.0325,
                "wheel_base": 0.118,
            },
            "create3": {
                "name": "Create3",
                "usd_relative_path": os.path.join(
                    "Isaac", "Robots", "iRobot", "create_3.usd"
                ),
                "wheel_dof_names": ["left_wheel_joint", "right_wheel_joint"],
                "wheel_radius": 0.03575,
                "wheel_base": 0.233,
            },
            "carter": {
                "name": "Nova_Carter",
                "usd_relative_path": os.path.join(
                    "Isaac", "Robots", "Carter", "nova_carter_sensors.usd"
                ),
                "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
                "wheel_radius": 0.14,
                "wheel_base": 0.413,
            },
            "carter_ROS": {
                "name": "Nova_Carter",
                "usd_relative_path": find_robot_config_path("nova_carter_ros2_sensors.usd"),
                "wheel_dof_names": ["joint_wheel_left", "joint_wheel_right"],
                "wheel_radius": 0.14,
                "wheel_base": 0.413,
            },
        }

        if robot_name not in robot_configs:
            raise ValueError(f"Unsupported robot_name: {robot_name}")

        robot_config = robot_configs[robot_name]
        
        # Handle absolute vs relative paths for robot USD files
        if os.path.isabs(robot_config["usd_relative_path"]):
            # Absolute path (for custom robots like carter_ROS)
            robot_path = robot_config["usd_relative_path"]
        else:
            # Relative path (for built-in Isaac Sim robots)
            robot_path = os.path.join(assets_root_path, robot_config["usd_relative_path"])

        # Add robot to world
        robot_prim_path = f"/World/{robot_config['name']}"
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path=robot_prim_path,
                name="Robot",
                wheel_dof_names=robot_config["wheel_dof_names"],
                create_robot=True,
                usd_path=robot_path,
                position=[0.0, 0.0, 0.0],
                orientation=[0, 0, 0, 1],
            )
        )

        # Create differential drive controller
        self.diff_controller = DifferentialController(
            name="diff_drive_controller",
            wheel_radius=robot_config["wheel_radius"],
            wheel_base=robot_config["wheel_base"],
        )

        # ROS2 cmd_vel subscriber
        self.cmd_lin = 0.00
        self.cmd_ang = 0.00
        self.cmd_vel_sub = self.create_subscription(
            Twist, "/cmd_vel", self._cmd_vel_callback, 10
        )
        self.clicked_point_sub = self.create_subscription(
            PointStamped, "/clicked_point", self._clicked_point_callback, 10
        )
        self.goal_pose_sub = self.create_subscription(
            PoseStamped,
            os.environ.get("HUNAV_RECORD_POSE_TOPIC", DEFAULT_RECORD_POSE_TOPIC),
            self._goal_pose_callback,
            10,
        )
        self.gt_visible_objects_topic = os.environ.get(
            "HUNAV_GT_VISIBLE_OBJECTS_TOPIC",
            DEFAULT_GT_VISIBLE_OBJECTS_TOPIC,
        )
        self.gt_visible_objects_pub = self.create_publisher(
            String,
            self.gt_visible_objects_topic,
            10,
        )
        self.gt_visible_objects_front_topic = os.environ.get(
            "HUNAV_GT_VISIBLE_OBJECTS_FRONT_TOPIC",
            DEFAULT_GT_VISIBLE_OBJECTS_FRONT_TOPIC,
        )
        self.gt_visible_objects_rear_topic = os.environ.get(
            "HUNAV_GT_VISIBLE_OBJECTS_REAR_TOPIC",
            DEFAULT_GT_VISIBLE_OBJECTS_REAR_TOPIC,
        )
        self.gt_visible_objects_front_pub = self.create_publisher(
            String,
            self.gt_visible_objects_front_topic,
            10,
        )
        self.gt_visible_objects_rear_pub = self.create_publisher(
            String,
            self.gt_visible_objects_rear_topic,
            10,
        )
        self.gt_image_width = int(
            os.environ.get("HUNAV_GT_IMAGE_WIDTH", str(DEFAULT_GT_IMAGE_WIDTH))
        )
        self.gt_image_height = int(
            os.environ.get("HUNAV_GT_IMAGE_HEIGHT", str(DEFAULT_GT_IMAGE_HEIGHT))
        )
        self.gt_publish_every_n_steps = max(
            1,
            int(
                os.environ.get(
                    "HUNAV_GT_PUBLISH_EVERY_N_STEPS",
                    str(DEFAULT_GT_PUBLISH_EVERY_N_STEPS),
                )
            ),
        )
        self.gt_publish_step_counter = 0
        self.chois_publish_step_counter = 0
        self.gt_occlusion_overlap_threshold = float(
            os.environ.get(
                "HUNAV_GT_OCCLUSION_OVERLAP_THRESHOLD",
                str(DEFAULT_GT_OCCLUSION_OVERLAP_THRESHOLD),
            )
        )
        self.gt_occlusion_depth_margin = float(
            os.environ.get(
                "HUNAV_GT_OCCLUSION_DEPTH_MARGIN",
                str(DEFAULT_GT_OCCLUSION_DEPTH_MARGIN),
            )
        )
        self.gt_camera_prim_path = self._resolve_gt_camera_prim_path()
        self.gt_front_camera_prim_path = self._resolve_named_gt_camera_prim_path(
            "front_hawk/left/camera_left"
        )
        self.gt_rear_camera_prim_path = self._resolve_named_gt_camera_prim_path(
            "back_hawk/left/camera_left"
        ) or self._resolve_named_gt_camera_prim_path("rear_hawk/left/camera_left")
        if self.gt_camera_prim_path:
            self.get_logger().info(
                f"Publishing GT visible objects on {self.gt_visible_objects_topic} "
                f"using camera prim {self.gt_camera_prim_path}"
            )
        else:
            self.get_logger().warning(
                "GT visible-objects publisher enabled, but no camera prim was found yet. "
                "Set HUNAV_GT_CAMERA_PRIM_PATH to override."
            )
        if self.gt_front_camera_prim_path:
            self.get_logger().info(
                f"Publishing front-left GT visible objects on {self.gt_visible_objects_front_topic} "
                f"using camera prim {self.gt_front_camera_prim_path}"
            )
        if self.gt_rear_camera_prim_path:
            self.get_logger().info(
                f"Publishing rear-left GT visible objects on {self.gt_visible_objects_rear_topic} "
                f"using camera prim {self.gt_rear_camera_prim_path}"
            )
        self.chois_state_topic = os.environ.get(
            "HUNAV_CHOIS_STATE_TOPIC",
            DEFAULT_CHOIS_STATE_TOPIC,
        )
        self.chois_human_poses_topic = os.environ.get(
            "HUNAV_CHOIS_HUMAN_POSES_TOPIC",
            DEFAULT_CHOIS_HUMAN_POSES_TOPIC,
        )
        self.chois_object_poses_topic = os.environ.get(
            "HUNAV_CHOIS_OBJECT_POSES_TOPIC",
            DEFAULT_CHOIS_OBJECT_POSES_TOPIC,
        )
        self.chois_markers_topic = os.environ.get(
            "HUNAV_CHOIS_MARKERS_TOPIC",
            DEFAULT_CHOIS_MARKERS_TOPIC,
        )
        self.chois_frame_id = os.environ.get(
            "HUNAV_CHOIS_FRAME_ID",
            DEFAULT_CHOIS_FRAME_ID,
        )
        self.chois_publish_every_n_steps = max(
            1,
            int(
                os.environ.get(
                    "HUNAV_CHOIS_PUBLISH_EVERY_N_STEPS",
                    str(DEFAULT_CHOIS_PUBLISH_EVERY_N_STEPS),
                )
            ),
        )
        self.chois_state_pub = self.create_publisher(
            String,
            self.chois_state_topic,
            10,
        )
        self.chois_human_poses_pub = self.create_publisher(
            PoseArray,
            self.chois_human_poses_topic,
            10,
        )
        self.chois_object_poses_pub = self.create_publisher(
            PoseArray,
            self.chois_object_poses_topic,
            10,
        )
        self.chois_markers_pub = self.create_publisher(
            MarkerArray,
            self.chois_markers_topic,
            10,
        )
        if self.custom_animated_assets:
            self.get_logger().info(
                f"Publishing CHOIS state on {self.chois_state_topic}, "
                f"poses on {self.chois_human_poses_topic} and "
                f"{self.chois_object_poses_topic}, markers on "
                f"{self.chois_markers_topic}"
            )

        # Setup HuNavManager
        self.hunav = HuNavManager(
            node=self,
            world=self.world,
            config_file_path=hunav_config,
            robot_prim_path=robot_prim_path,
            robot=self.robot,
        )

        self.create_ros_clock_action_graph()

        self.hunav.initialize_agents()
        self.hunav.initialize_hunav_nodes()

    def _resolve_chois_animation_usd_root(self):
        candidates = [
            os.environ.get("HUNAV_CHOIS_ANIMATION_USD_ROOT"),
            DEFAULT_CHOIS_ANIMATION_USD_ROOT,
            DEFAULT_CHOIS_ANIMATION_USD_ROOT_LOCAL,
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _resolve_chois_animation_npy_root(self):
        candidates = [
            os.environ.get("HUNAV_CHOIS_ANIMATION_NPY_ROOT"),
            DEFAULT_CHOIS_ANIMATION_NPY_ROOT,
            DEFAULT_CHOIS_ANIMATION_NPY_ROOT_LOCAL,
        ]
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _chois_asset_key_from_usd_path(self, usd_path: str):
        usd_file = Path(usd_path)
        parent_name = usd_file.parent.name
        if parent_name.endswith("_with_cache"):
            return parent_name[: -len("_with_cache")]
        return usd_file.stem.replace("_with_cache", "")

    def _prim_safe_name(self, name: str):
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", name)
        sanitized = sanitized.strip("_")
        if not sanitized:
            sanitized = "CHOISActor"
        if sanitized[0].isdigit():
            sanitized = f"CHOIS_{sanitized}"
        return sanitized

    def _asset_key_from_scanned_usd_path(self, usd_path: str):
        if not self.chois_animation_usd_root:
            return self._chois_asset_key_from_usd_path(usd_path)

        usd_root = Path(self.chois_animation_usd_root) / "usd"
        try:
            relative_parts = Path(usd_path).resolve().relative_to(usd_root.resolve()).parts
        except Exception:
            return self._chois_asset_key_from_usd_path(usd_path)

        if self.chois_animation_npy_root:
            for part in relative_parts:
                candidate_npy = Path(self.chois_animation_npy_root) / f"{part}.npy"
                if candidate_npy.exists():
                    return part

        if len(relative_parts) >= 2:
            return relative_parts[1]
        if relative_parts:
            return relative_parts[0]
        return self._chois_asset_key_from_usd_path(usd_path)

    def _resolve_chois_waypoints_path(self, asset_key=None):
        candidates = []
        if asset_key and self.chois_animation_npy_root:
            candidates.append(str(Path(self.chois_animation_npy_root) / f"{asset_key}.npy"))
            candidates.extend(
                str(path)
                for path in Path(self.chois_animation_npy_root).rglob(f"{asset_key}.npy")
            )
        candidates.extend([
            os.environ.get("HUNAV_CHOIS_WAYPOINTS_NPY"),
            DEFAULT_CHOIS_WAYPOINTS_NPY,
            DEFAULT_CHOIS_WAYPOINTS_NPY_LOCAL,
        ])
        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate
        return None

    def _resolve_chois_case_output_dir(self, asset_key: str):
        if not asset_key:
            return None
        candidates = [
            os.environ.get("HUNAV_CHOIS_OUTPUT_ROOT"),
            DEFAULT_CHOIS_OUTPUT_ROOT,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            case_dir = Path(candidate) / str(self.map_name) / asset_key
            if case_dir.exists():
                return case_dir
        return None

    def _sample_smplx_object_local_xy_trajectory(self, asset_key: str, sample_count: int):
        if PlyData is None:
            return None

        case_dir = self._resolve_chois_case_output_dir(asset_key)
        if case_dir is None:
            return None

        object_dir = case_dir / "smplx_object"
        if not object_dir.exists():
            return None

        ply_files = sorted(object_dir.glob("*.ply"))
        if len(ply_files) < 2:
            return None

        if sample_count <= 1:
            sample_indices = [0]
        else:
            sample_indices = np.linspace(0, len(ply_files) - 1, sample_count)
            sample_indices = np.round(sample_indices).astype(int)

        sampled_points = []
        for file_index in sample_indices:
            ply = PlyData.read(str(ply_files[int(file_index)]))
            vertices = ply["vertex"]
            xyz = np.column_stack(
                [vertices["x"], vertices["y"], vertices["z"]]
            ).astype(np.float32)
            center = (xyz.min(axis=0) + xyz.max(axis=0)) * 0.5
            sampled_points.append(center[:2])

        return np.asarray(sampled_points, dtype=np.float32)

    def _sample_usd_local_xy_trajectory(self, usd_path: str, sample_count: int):
        asset_stage = Usd.Stage.Open(usd_path)
        if not asset_stage:
            return None

        root_prim = asset_stage.GetDefaultPrim()
        if not root_prim:
            root_prim = asset_stage.GetPseudoRoot()
        if not root_prim:
            return None

        sampled_prim = root_prim
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == "Animated_Object":
                sampled_prim = prim
                break

        start_time_code = float(asset_stage.GetStartTimeCode())
        end_time_code = float(asset_stage.GetEndTimeCode())
        if end_time_code < start_time_code:
            return None

        if sample_count <= 1:
            sample_time_codes = [start_time_code]
        else:
            sample_time_codes = np.linspace(start_time_code, end_time_code, sample_count)

        sampled_points = []
        for time_code in sample_time_codes:
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode(float(time_code)),
                [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
                useExtentsHint=True,
            )
            world_bound = bbox_cache.ComputeWorldBound(sampled_prim)
            aligned_range = world_bound.ComputeAlignedBox()
            center = aligned_range.GetMidpoint()
            sampled_points.append([float(center[0]), float(center[1])])

        return np.asarray(sampled_points, dtype=np.float32)

    def _resample_xy_by_arclength(self, points_xy, target_len: int):
        points_xy = np.asarray(points_xy, dtype=np.float32)
        if len(points_xy) == target_len:
            return points_xy.copy()
        if len(points_xy) < 2:
            return None

        segment_lengths = np.linalg.norm(np.diff(points_xy, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        total_length = float(cumulative[-1])
        if total_length <= 1e-8:
            return np.repeat(points_xy[:1], target_len, axis=0)

        target_dist = np.linspace(0.0, total_length, target_len, dtype=np.float32)
        resampled = np.empty((target_len, 2), dtype=np.float32)
        for dim in range(2):
            resampled[:, dim] = np.interp(target_dist, cumulative, points_xy[:, dim])
        return resampled

    def _search_min_rmse_yaw_alignment(self, local_xy, expected_xy):
        local_xy = np.asarray(local_xy, dtype=np.float32)
        expected_xy = np.asarray(expected_xy, dtype=np.float32)
        if len(local_xy) < 2 or len(expected_xy) < 2:
            return None

        def evaluate_candidate(candidate_xy: np.ndarray, yaw_deg: float):
            local_rs = self._resample_xy_by_arclength(candidate_xy, len(expected_xy))
            if local_rs is None:
                return None
            yaw_rad = float(np.radians(yaw_deg))
            rotation_2d = np.asarray(
                [
                    [np.cos(yaw_rad), -np.sin(yaw_rad)],
                    [np.sin(yaw_rad), np.cos(yaw_rad)],
                ],
                dtype=np.float32,
            )
            translation_xy = expected_xy[0] - (rotation_2d @ local_rs[0])
            aligned_xy = local_rs @ rotation_2d.T + translation_xy
            rmse = float(
                np.sqrt(np.mean(np.sum((aligned_xy - expected_xy) ** 2, axis=1)))
            )
            end_error = float(np.linalg.norm(aligned_xy[-1] - expected_xy[-1]))
            return {
                "yaw_deg": float(yaw_deg),
                "rotation_2d": rotation_2d,
                "translation_xy": translation_xy,
                "rmse": rmse,
                "end_error": end_error,
                "local_start": local_rs[0],
                "local_end": local_rs[-1],
                "aligned_xy": aligned_xy,
            }

        best_alignment = None
        coarse_yaws = np.arange(-180.0, 180.0, 5.0, dtype=np.float32)
        refine_offsets = np.arange(-5.0, 5.0001, 0.5, dtype=np.float32)
        axis_labels = ["x", "y"]
        axis_pairs = [(0, 1), (1, 0)]
        for axis_a, axis_b in axis_pairs:
            for sign_a in (-1.0, 1.0):
                for sign_b in (-1.0, 1.0):
                    candidate_base = np.empty((local_xy.shape[0], 2), dtype=np.float32)
                    candidate_base[:, 0] = local_xy[:, axis_a] * sign_a
                    candidate_base[:, 1] = local_xy[:, axis_b] * sign_b
                    for reverse in (False, True):
                        candidate_xy = candidate_base[::-1] if reverse else candidate_base

                        local_best = None
                        for yaw_deg in coarse_yaws:
                            candidate = evaluate_candidate(candidate_xy, float(yaw_deg))
                            if candidate is None:
                                continue
                            if local_best is None or candidate["rmse"] < local_best["rmse"]:
                                local_best = candidate
                            if best_alignment is None or candidate["rmse"] < best_alignment["rmse"]:
                                best_alignment = {
                                    **candidate,
                                    "projection": (
                                        f"{'+' if sign_a > 0 else '-'}{axis_labels[axis_a]},"
                                        f"{'+' if sign_b > 0 else '-'}{axis_labels[axis_b]}"
                                    ),
                                    "reversed": reverse,
                                }

                        if local_best is None:
                            continue

                        refine_yaws = (
                            (local_best["yaw_deg"] + refine_offsets + 180.0) % 360.0
                        ) - 180.0
                        for yaw_deg in np.unique(np.round(refine_yaws, 4)):
                            candidate = evaluate_candidate(candidate_xy, float(yaw_deg))
                            if candidate is None:
                                continue
                            if best_alignment is None or candidate["rmse"] < best_alignment["rmse"]:
                                best_alignment = {
                                    **candidate,
                                    "projection": (
                                        f"{'+' if sign_a > 0 else '-'}{axis_labels[axis_a]},"
                                        f"{'+' if sign_b > 0 else '-'}{axis_labels[axis_b]}"
                                    ),
                                    "reversed": reverse,
                                }

        return best_alignment

    def _projection_matrix_from_label(self, projection_label: str):
        projection_label = str(projection_label or "+x,+y")
        first_label, second_label = [part.strip() for part in projection_label.split(",")]

        def parse_axis(label: str):
            sign = -1.0 if label.startswith("-") else 1.0
            axis_name = label[1:] if label[:1] in "+-" else label
            axis_index = {"x": 0, "y": 1}.get(axis_name)
            if axis_index is None:
                raise ValueError(f"Unsupported projection axis label: {label}")
            return sign, axis_index

        sign_a, axis_a = parse_axis(first_label)
        sign_b, axis_b = parse_axis(second_label)

        projection = np.zeros((2, 2), dtype=np.float32)
        projection[0, axis_a] = sign_a
        projection[1, axis_b] = sign_b
        return projection

    def _compute_usd_endpoint_alignment(self, usd_path: str, expected_xy, asset_key=None):
        if expected_xy is None or len(expected_xy) < 2:
            return None

        sampled_points = None
        sampled_source = "animated_object_bbox"
        if asset_key:
            sampled_points = self._sample_smplx_object_local_xy_trajectory(
                asset_key, sample_count=len(expected_xy)
            )
            if sampled_points is not None:
                sampled_source = "smplx_object"

        if sampled_points is None:
            sampled_points = self._sample_usd_local_xy_trajectory(
                usd_path, sample_count=len(expected_xy)
            )
        if sampled_points is None or len(sampled_points) < 2:
            return None
        best_alignment = self._search_min_rmse_yaw_alignment(sampled_points, expected_xy)
        if best_alignment is None:
            return None

        translation_xy = best_alignment["translation_xy"]
        return {
            "position": Gf.Vec3d(
                float(translation_xy[0]),
                float(translation_xy[1]),
                DEFAULT_CHOIS_ACTOR_Z,
            ),
            "yaw_deg": best_alignment["yaw_deg"],
            "local_start": best_alignment["local_start"],
            "local_end": best_alignment["local_end"],
            "end_error": best_alignment["end_error"],
            "rmse": best_alignment["rmse"],
            "sampled_source": sampled_source,
            "projection": best_alignment.get("projection", "+x,+y"),
            "reversed": bool(best_alignment.get("reversed", False)),
        }

    def _estimate_aligned_placement_from_paths(self, expected_xy, usd_local_xy):
        if expected_xy is None or usd_local_xy is None:
            return None
        if len(expected_xy) < 2 or len(usd_local_xy) < 2:
            return None

        sample_idx = np.linspace(0, len(usd_local_xy) - 1, len(expected_xy))
        sample_idx = np.round(sample_idx).astype(int)
        usd_resampled = usd_local_xy[sample_idx]

        expected_center = expected_xy.mean(axis=0)
        usd_center = usd_resampled.mean(axis=0)
        expected_centered = expected_xy - expected_center
        usd_centered = usd_resampled - usd_center

        covariance = usd_centered.T @ expected_centered
        u_mat, _, vt_mat = np.linalg.svd(covariance)
        rotation = vt_mat.T @ u_mat.T
        if np.linalg.det(rotation) < 0:
            vt_mat[-1, :] *= -1
            rotation = vt_mat.T @ u_mat.T

        translation = expected_center - (usd_center @ rotation.T)
        yaw_deg = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        yaw_deg += float(os.environ.get("HUNAV_CHOIS_YAW_OFFSET_DEG", "0.0"))
        rmse = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (usd_resampled @ rotation.T + translation - expected_xy) ** 2,
                        axis=1,
                    )
                )
            )
        )
        return {
            "position": Gf.Vec3d(
                float(translation[0]),
                float(translation[1]),
                DEFAULT_CHOIS_ACTOR_Z,
            ),
            "yaw_deg": yaw_deg,
            "rmse": rmse,
        }

    def _load_custom_animated_placement(self, asset_key=None, usd_path=None):
        waypoint_path = self._resolve_chois_waypoints_path(asset_key)
        if not waypoint_path:
            print(
                f"[TeleopHuNavSim] No CHOIS waypoint file found for "
                f"{asset_key or 'default asset'}. Using default placement."
            )
            return {
                "position": Gf.Vec3d(0.0, 0.0, DEFAULT_CHOIS_ACTOR_Z),
                "yaw_deg": 0.0,
            }

        try:
            waypoints = np.asarray(np.load(waypoint_path), dtype=np.float32)
        except Exception as exc:
            print(
                f"[TeleopHuNavSim] Failed to load CHOIS waypoints from "
                f"{waypoint_path}: {exc}. Using default placement."
            )
            return {
                "position": Gf.Vec3d(0.0, 0.0, DEFAULT_CHOIS_ACTOR_Z),
                "yaw_deg": 0.0,
            }

        if waypoints.ndim != 2 or waypoints.shape[0] < 1 or waypoints.shape[1] < 2:
            print(
                f"[TeleopHuNavSim] Invalid CHOIS waypoint array shape "
                f"{waypoints.shape} in {waypoint_path}. Using default placement."
            )
            return {
                "position": Gf.Vec3d(0.0, 0.0, DEFAULT_CHOIS_ACTOR_Z),
                "yaw_deg": 0.0,
            }

        expected_xy = np.asarray(waypoints[:, :2], dtype=np.float32)
        if usd_path:
            endpoint_alignment = self._compute_usd_endpoint_alignment(
                usd_path, expected_xy, asset_key=asset_key
            )
            if endpoint_alignment is not None:
                print(
                    f"[TeleopHuNavSim] Using endpoint-aligned CHOIS placement from "
                    f"{waypoint_path}: position={endpoint_alignment['position']}, "
                    f"yaw_deg={endpoint_alignment['yaw_deg']:.3f}, "
                    f"rmse={endpoint_alignment['rmse']:.3f}, "
                    f"projection={endpoint_alignment['projection']}, "
                    f"reversed={endpoint_alignment['reversed']}, "
                    f"local_start=({float(endpoint_alignment['local_start'][0]):.3f}, "
                    f"{float(endpoint_alignment['local_start'][1]):.3f}), "
                    f"local_end=({float(endpoint_alignment['local_end'][0]):.3f}, "
                    f"{float(endpoint_alignment['local_end'][1]):.3f}), "
                    f"end_error={endpoint_alignment['end_error']:.3f}, "
                    f"source={endpoint_alignment['sampled_source']}"
                )
                return {
                    "position": endpoint_alignment["position"],
                    "yaw_deg": endpoint_alignment["yaw_deg"],
                    "projection": endpoint_alignment["projection"],
                    "reversed": endpoint_alignment["reversed"],
                }

        direction = np.zeros(2, dtype=np.float32)
        for delta in np.diff(expected_xy, axis=0)[:5]:
            if float(np.linalg.norm(delta)) > 1e-4:
                direction += delta
        yaw_deg = float(np.degrees(np.arctan2(direction[1], direction[0]))) if float(
            np.linalg.norm(direction)
        ) > 1e-4 else 0.0
        yaw_deg += float(os.environ.get("HUNAV_CHOIS_YAW_OFFSET_DEG", "0.0"))

        origin = Gf.Vec3d(
            float(expected_xy[0, 0]),
            float(expected_xy[0, 1]),
            DEFAULT_CHOIS_ACTOR_Z,
        )
        print(
            f"[TeleopHuNavSim] Using fallback CHOIS placement from {waypoint_path}: "
            f"position={origin}, yaw_deg={yaw_deg:.3f}"
        )
        return {"position": origin, "yaw_deg": yaw_deg}

    def _resolve_custom_animated_usd_entries(self, animated_usd_path):
        package_share = Path(find_package_share_directory())
        package_copy = (
            package_share
            / "assets"
            / "sample_motion_001_with_cache"
            / "sample_motion_001_with_cache.usd"
        )
        entries = []
        used_usd_paths = set()

        explicit_candidates = [
            animated_usd_path,
            os.environ.get("HUNAV_CUSTOM_ANIMATED_USD"),
            DEFAULT_CHOIS_ANIMATED_USD_LOCAL,
            DEFAULT_CHOIS_ANIMATED_USD,
            str(package_copy),
        ]
        for candidate in explicit_candidates:
            if candidate and os.path.exists(candidate) and candidate not in used_usd_paths:
                asset_key = self._asset_key_from_scanned_usd_path(candidate)
                prim_name = self._prim_safe_name(asset_key)
                entries.append({
                    "usd_path": candidate,
                    "prim_path": f"/World/CHOISActors/{prim_name}",
                    "asset_key": asset_key,
                    "placement": self._load_custom_animated_placement(asset_key, candidate),
                })
                used_usd_paths.add(candidate)

        if self.chois_animation_usd_root:
            usd_root = Path(self.chois_animation_usd_root) / "usd"
            if usd_root.exists():
                usd_candidates = sorted(str(path) for path in usd_root.rglob("*.usd"))
                preferred_usd_candidates = [
                    path for path in usd_candidates if path.endswith("_with_cache.usd")
                ]
                ordered_usd_candidates = preferred_usd_candidates + [
                    path for path in usd_candidates if path not in set(preferred_usd_candidates)
                ]
                used_asset_keys = {entry["asset_key"] for entry in entries}

                for usd_candidate in ordered_usd_candidates:
                    if usd_candidate in used_usd_paths:
                        continue

                    asset_key = self._asset_key_from_scanned_usd_path(usd_candidate)
                    if asset_key in used_asset_keys:
                        continue

                    prim_name = self._prim_safe_name(asset_key)
                    entries.append({
                        "usd_path": usd_candidate,
                        "prim_path": f"/World/CHOISActors/{prim_name}",
                        "asset_key": asset_key,
                        "placement": self._load_custom_animated_placement(
                            asset_key, usd_candidate
                        ),
                    })
                    used_usd_paths.add(usd_candidate)
                    used_asset_keys.add(asset_key)

        if entries:
            print(
                f"[TeleopHuNavSim] Loading {len(entries)} CHOIS animated USD asset(s)."
            )
            return entries

        print(
            "[TeleopHuNavSim] No custom animated USD found. "
            "Skipping CHOIS actor load."
        )
        return []

    def _resolve_clicked_points_output_path(self):
        candidates = [
            os.environ.get("HUNAV_CLICKED_POINTS_NPY"),
            DEFAULT_CLICKED_POINTS_NPY,
            f"{root}/isaac_sim/hunav_isaac_ws/src/animation_usd/rviz_clicked_points.npy",
        ]
        for candidate in candidates:
            if candidate:
                parent = Path(candidate).parent
                if parent.exists():
                    return candidate
        return str(Path.cwd() / "rviz_clicked_points.npy")

    def _resolve_goal_poses_output_path(self):
        candidates = [
            os.environ.get("HUNAV_GOAL_POSES_NPY"),
            DEFAULT_GOAL_POSES_NPY,
            f"{root}/isaac_sim/hunav_isaac_ws/src/animation_usd/rviz_goal_poses.npy",
        ]
        for candidate in candidates:
            if candidate:
                parent = Path(candidate).parent
                if parent.exists():
                    return candidate
        return str(Path.cwd() / "rviz_goal_poses.npy")

    def _resolve_chois_debug_output_root(self):
        candidates = [
            os.environ.get("HUNAV_CHOIS_DEBUG_ROOT"),
            str(Path(self.chois_animation_usd_root) / "debug")
            if self.chois_animation_usd_root
            else None,
            DEFAULT_CHOIS_DEBUG_ROOT,
        ]
        for candidate in candidates:
            if candidate:
                Path(candidate).mkdir(parents=True, exist_ok=True)
                return candidate
        fallback = Path.cwd() / "animation_debug"
        fallback.mkdir(parents=True, exist_ok=True)
        return str(fallback)

    def _resolve_chois_debug_visualization(self):
        value = os.environ.get("HUNAV_CHOIS_DEBUG_VISUALIZATION")
        if value is None:
            return DEFAULT_CHOIS_DEBUG_VISUALIZATION
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _resolve_gt_camera_prim_path(self):
        explicit = os.environ.get("HUNAV_GT_CAMERA_PRIM_PATH")
        if explicit:
            prim = self.world.stage.GetPrimAtPath(explicit)
            if prim and prim.IsValid():
                return explicit
            self.get_logger().warning(
                f"HUNAV_GT_CAMERA_PRIM_PATH={explicit} does not exist in stage"
            )
        return self._find_default_camera_prim_path()

    def _find_default_camera_prim_path(self):
        hint = os.environ.get("HUNAV_GT_CAMERA_PRIM_HINT", DEFAULT_GT_CAMERA_PRIM_HINT).lower()
        best_path = None
        best_score = None
        for prim in Usd.PrimRange(self.world.stage.GetPseudoRoot()):
            if not prim.IsA(UsdGeom.Camera):
                continue
            path = str(prim.GetPath())
            path_lower = path.lower()
            score = 0
            if hint and hint in path_lower:
                score += 100
            if "camera" in path_lower:
                score += 20
            if "nova_carter" in path_lower:
                score += 10
            if "ros2" in path_lower:
                score -= 2
            if best_score is None or score > best_score:
                best_score = score
                best_path = path
        return best_path

    def _resolve_named_gt_camera_prim_path(self, path_fragment: str):
        fragment = (path_fragment or "").lower()
        if not fragment:
            return None
        for prim in Usd.PrimRange(self.world.stage.GetPseudoRoot()):
            if not prim.IsA(UsdGeom.Camera):
                continue
            path = str(prim.GetPath())
            if fragment in path.lower():
                return path
        return None

    def _is_gt_candidate(self, prim):
        if prim is None or not prim.IsValid():
            return False
        path = str(prim.GetPath())
        if not path.startswith("/World/"):
            return False
        name = prim.GetName()
        lowered = name.lower()
        ignored_names = {
            "GroundPlane",
            "Camera",
            "CHOISDebug",
            "Nova_Carter",
            "ROS2",
            "Animations",
            "Biped_Setup",
            "physicsScene",
            "defaultPrim",
            "Looks",
            "_materials",
            "WalkLoop",
            "IdleLoop",
        }
        ignored_prefixes = (
            "looks",
            "_materials",
            "physics",
        )
        ignored_path_fragments = (
            "/looks",
            "/_materials",
            "/materials",
            "/camera",
            "/ros2",
            "/nova_carter",
            "/physicsscene",
            "/biped_setup",
            "/animations",
        )
        if name in ignored_names:
            return False
        if lowered in {item.lower() for item in ignored_names}:
            return False
        if any(lowered.startswith(prefix) for prefix in ignored_prefixes):
            return False
        path_lower = path.lower()
        if any(fragment in path_lower for fragment in ignored_path_fragments):
            return False
        if lowered in {"world", "choisactors", "characters", "warehouse_empty_small_realtime"}:
            return False
        if prim.IsA(UsdGeom.Camera):
            return False
        if not prim.IsA(UsdGeom.Xform):
            return False
        if path.startswith("/World/Characters/") and lowered.startswith("agent"):
            return True
        if "/CHOISActors/" in path:
            parent_path = str(prim.GetParent().GetPath()) if prim.GetParent() else ""
            if parent_path == "/World/CHOISActors":
                return True
            relative_path = path.split("/CHOISActors/", 1)[1]
            path_parts = [part for part in relative_path.split("/") if part]
            return (
                len(path_parts) == 3
                and path_parts[1] == "Armature_1"
                and path_parts[2] == "body_1"
            ) or (
                len(path_parts) == 2
                and path_parts[1] == "Animated_Object"
            )
        if path.startswith("/World/Characters/") and not lowered.startswith("agent"):
            return False
        return True

    def _iter_gt_candidate_prims(self):
        world_prim = self.world.stage.GetPrimAtPath("/World")
        if not world_prim or not world_prim.IsValid():
            return []
        candidates = []
        seen_paths = set()

        def add_candidate(prim):
            if prim is None or not prim.IsValid():
                return
            path = str(prim.GetPath())
            if path in seen_paths:
                return
            seen_paths.add(path)
            candidates.append(prim)

        for prim in Usd.PrimRange(world_prim):
            if str(prim.GetPath()) == "/World":
                continue
            if self._is_gt_candidate(prim):
                add_candidate(prim)
        return candidates

    def _semantic_name_from_prim(self, prim):
        path = str(prim.GetPath())
        name = prim.GetName()
        lower = name.lower()
        if "/CHOISActors/" in path:
            relative_path = path.split("/CHOISActors/", 1)[1]
            path_parts = [part for part in relative_path.split("/") if part]
            actor_name = re.sub(r"_[0-9]+$", "", path_parts[0].lower()) if path_parts else lower
            if len(path_parts) == 1:
                return actor_name
            if len(path_parts) >= 3 and path_parts[1] == "Armature_1" and path_parts[2] == "body_1":
                return f"{actor_name}_body"
            if len(path_parts) >= 2 and path_parts[1] == "Animated_Object":
                return f"{actor_name}_object"
            return actor_name
        if path.startswith("/World/Characters/") and lower.startswith("agent"):
            return "person"
        semantic_aliases = [
            ("forklift", "forklift"),
            ("ceilinglight", "light"),
            ("wheelchair", "wheelchair"),
            ("hospitalbed", "hospital_bed"),
            ("bed", "hospital_bed"),
            ("stretcher", "stretcher"),
            ("doorwall", "wall"),
            ("backwall", "wall"),
            ("sidewall", "wall"),
            ("wall", "wall"),
            ("floor", "floor"),
            ("ceiling", "ceiling"),
            ("door", "door"),
            ("window", "window"),
            ("glass", "glass"),
            ("sink", "sink"),
            ("washbasin", "sink"),
            ("toilet", "toilet"),
            ("urinal", "urinal"),
            ("cabinet", "cabinet"),
            ("cupboard", "cabinet"),
            ("filecabinet", "cabinet"),
            ("papercase", "cabinet"),
            ("dispenser", "dispenser"),
            ("drinksmachine", "vending_machine"),
            ("machine", "machine"),
            ("bottle", "bottle"),
            ("screen", "screen"),
            ("monitor", "screen"),
            ("tvdisplay", "screen"),
            ("computer", "computer"),
            ("keyboard", "keyboard"),
            ("mouse", "mouse"),
            ("printer", "printer"),
            ("smartphone", "phone"),
            ("phone", "phone"),
            ("sign", "sign"),
            ("pipe", "pipe"),
            ("extinguisher", "extinguisher"),
            ("smokedetector", "smoke_detector"),
            ("cctv", "camera"),
            ("picture", "picture"),
            ("frame", "picture"),
            ("clock", "clock"),
            ("plant", "plant"),
            ("vase", "vase"),
            ("sofa", "sofa"),
            ("bench", "bench"),
            ("desk", "desk"),
            ("desktable", "desk"),
            ("reception", "reception"),
            ("receptiontable", "reception"),
            ("elevator", "elevator"),
            ("cart", "cart"),
            ("pallet", "pallet"),
            ("klt", "klt_bin"),
            ("bin", "klt_bin"),
            ("box", "box"),
            ("barel", "barel"),
            ("barrel", "barel"),
            ("pillar", "pillar"),
            ("lamp", "lamp"),
            ("wire", "wire"),
            ("rack", "rack"),
            ("shelf", "shelf"),
            ("booksset", "book"),
            ("book", "book"),
            ("ringbinder", "binder"),
            ("binder", "binder"),
            ("pencilbox", "stationery_box"),
            ("pencilcase", "stationery_box"),
            ("pencil", "pencil"),
            ("marker", "marker"),
            ("feltpen", "marker"),
            ("a4", "paper"),
            ("paper", "paper"),
            ("building", "building"),
            ("personenleitsystem", "barrier"),
            ("beam", "beam"),
            ("bracketbeam", "beam"),
            ("table", "table"),
            ("chair", "chair"),
            ("trashcan", "trashcan"),
            ("tripod", "tripod"),
            ("person", "person"),
            ("body", "person"),
            ("human", "person"),
        ]
        for token, label in semantic_aliases:
            if token in lower:
                return label
        return re.sub(r"_[0-9]+$", "", lower)

    def _prim_has_renderable_descendant(self, prim):
        if prim is None or not prim.IsValid():
            return False
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Mesh):
                return True
        return False

    def _iter_scene_semantic_prims(self):
        world_prim = self.world.stage.GetPrimAtPath("/World")
        if not world_prim or not world_prim.IsValid():
            return []

        ignored_path_fragments = (
            "/Looks",
            "/looks",
            "/_materials",
            "/Materials",
            "/materials",
            "/ROS2",
            "/Animations",
            "/Biped_Setup",
            "/CHOISActors",
            "/Characters",
            "/Nova_Carter",
        )
        candidates = []
        seen_paths = set()
        for prim in Usd.PrimRange(world_prim):
            if prim is None or not prim.IsValid():
                continue
            path = str(prim.GetPath())
            if path == "/World":
                continue
            if path in seen_paths:
                continue
            if any(fragment in path for fragment in ignored_path_fragments):
                continue
            if prim.IsA(UsdGeom.Camera):
                continue
            if not prim.IsA(UsdGeom.Xform):
                continue
            has_direct_renderable_child = False
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh) or child.IsA(UsdGeom.Plane):
                    has_direct_renderable_child = True
                    break
            has_authored_reference = bool(prim.GetMetadata("references"))
            if not has_direct_renderable_child and not has_authored_reference:
                continue
            seen_paths.add(path)
            candidates.append(prim)
        return candidates

    def _apply_scene_semantics(self):
        applied_count = 0
        for prim in self._iter_scene_semantic_prims():
            semantic_label = self._semantic_name_from_prim(prim)
            if not semantic_label:
                continue
            self._apply_semantics_if_available(prim, semantic_label)
            applied_count += 1
        print(f"[TeleopHuNavSim] Applied scene semantics to {applied_count} prims.")

    def _ensure_hospital_lighting(self):
        if str(self.map_name).lower() != "hospital":
            return

        stage = self.world.stage
        hospital_prim = stage.GetPrimAtPath("/World/Hospital")
        if not hospital_prim or not hospital_prim.IsValid():
            hospital_prim = stage.GetPrimAtPath("/World")
        if not hospital_prim or not hospital_prim.IsValid():
            return

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        world_bound = bbox_cache.ComputeWorldBound(hospital_prim)
        aligned_box = world_bound.ComputeAlignedBox()
        if aligned_box.IsEmpty():
            return

        min_pt = aligned_box.GetMin()
        max_pt = aligned_box.GetMax()
        center_x = float((min_pt[0] + max_pt[0]) * 0.5)
        center_y = float((min_pt[1] + max_pt[1]) * 0.5)
        span_x = max(float(max_pt[0] - min_pt[0]), 1.0)
        span_y = max(float(max_pt[1] - min_pt[1]), 1.0)
        ceiling_z = float(max_pt[2]) - 0.25

        lights_root_path = "/World/HospitalExtraLights"
        stage.DefinePrim(lights_root_path, "Xform")
        x_offsets = (-0.25 * span_x, 0.25 * span_x)
        y_offsets = (-0.25 * span_y, 0.25 * span_y)
        for ix, offset_x in enumerate(x_offsets):
            for iy, offset_y in enumerate(y_offsets):
                light = UsdLux.SphereLight.Define(
                    stage, f"{lights_root_path}/CeilingLight_{ix}_{iy}"
                )
                light.CreateIntensityAttr(60000.0)
                light.CreateRadiusAttr(0.45)
                light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
                light.CreateExposureAttr(2.0)
                xform = UsdGeom.Xformable(light.GetPrim())
                translate_op = None
                for op in xform.GetOrderedXformOps():
                    if op.GetOpName() == "xformOp:translate":
                        translate_op = op
                        break
                if translate_op is None:
                    translate_op = xform.AddTranslateOp()
                translate_op.Set(
                    Gf.Vec3d(center_x + offset_x, center_y + offset_y, ceiling_z)
                )
        print("[TeleopHuNavSim] Added fallback hospital ceiling lights.")

    def _apply_semantics_if_available(self, prim, semantic_label):
        if prim is None or not prim.IsValid() or not semantic_label:
            return
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
        if add_update_semantics is None:
            return
        try:
            add_update_semantics(prim, semantic_label)
        except TypeError:
            try:
                add_update_semantics(prim, semantic_label, "class")
            except Exception as exc:
                print(
                    f"[TeleopHuNavSim] Warning: failed to apply semantics "
                    f"'{semantic_label}' to {prim.GetPath()}: {exc}"
                )
        except Exception as exc:
            print(
                f"[TeleopHuNavSim] Warning: failed to apply semantics "
                f"'{semantic_label}' to {prim.GetPath()}: {exc}"
            )

    def _apply_chois_actor_semantics(self, actor_prim):
        if actor_prim is None or not actor_prim.IsValid():
            return
        actor_label = self._semantic_name_from_prim(actor_prim)
        self._apply_semantics_if_available(actor_prim, actor_label)

        body_prim = self.world.stage.GetPrimAtPath(
            f"{actor_prim.GetPath()}/Armature_1/body_1"
        )
        if body_prim and body_prim.IsValid():
            self._apply_semantics_if_available(
                body_prim, self._semantic_name_from_prim(body_prim)
            )

        object_prim = self.world.stage.GetPrimAtPath(
            f"{actor_prim.GetPath()}/Animated_Object"
        )
        if object_prim and object_prim.IsValid():
            self._apply_semantics_if_available(
                object_prim, self._semantic_name_from_prim(object_prim)
            )

    def _compute_world_bbox(self, prim):
        bbox_cache = UsdGeom.BBoxCache(
            self._get_current_animation_time_code(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        world_bound = bbox_cache.ComputeWorldBound(prim)
        aligned = world_bound.ComputeAlignedBox()
        if aligned.IsEmpty():
            return None
        return aligned

    def _bbox_corners(self, aligned_box):
        min_pt = aligned_box.GetMin()
        max_pt = aligned_box.GetMax()
        xs = [float(min_pt[0]), float(max_pt[0])]
        ys = [float(min_pt[1]), float(max_pt[1])]
        zs = [float(min_pt[2]), float(max_pt[2])]
        return [
            Gf.Vec3d(x, y, z)
            for x in xs
            for y in ys
            for z in zs
        ]

    def _project_world_points(self, camera_prim, points_world):
        camera = UsdGeom.Camera(camera_prim)
        if not camera:
            return []
        xform_cache = UsdGeom.XformCache(self._get_current_animation_time_code())
        cam_world = xform_cache.GetLocalToWorldTransform(camera_prim)
        world_to_cam = cam_world.GetInverse()
        focal = float(camera.GetFocalLengthAttr().Get() or 24.0)
        h_ap = float(camera.GetHorizontalApertureAttr().Get() or 20.955)
        v_ap = float(camera.GetVerticalApertureAttr().Get() or 15.2908)
        fx = self.gt_image_width * focal / max(h_ap, 1e-6)
        fy = self.gt_image_height * focal / max(v_ap, 1e-6)
        cx = self.gt_image_width * 0.5
        cy = self.gt_image_height * 0.5

        pixels = []
        for point in points_world:
            cam_pt = world_to_cam.Transform(point)
            z_forward = -float(cam_pt[2])
            if z_forward <= 1e-4:
                continue
            x_px = fx * (float(cam_pt[0]) / z_forward) + cx
            y_px = fy * (-float(cam_pt[1]) / z_forward) + cy
            pixels.append((x_px, y_px, z_forward))
        return pixels

    def _project_bbox_to_image(self, camera_prim, aligned_box):
        pixels = self._project_world_points(camera_prim, self._bbox_corners(aligned_box))
        if not pixels:
            return None
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        zs = [p[2] for p in pixels]
        bbox = {
            "xmin": float(min(xs)),
            "ymin": float(min(ys)),
            "xmax": float(max(xs)),
            "ymax": float(max(ys)),
            "zmin": float(min(zs)),
            "zmax": float(max(zs)),
        }
        bbox["visible"] = (
            bbox["xmax"] >= 0.0
            and bbox["xmin"] < float(self.gt_image_width)
            and bbox["ymax"] >= 0.0
            and bbox["ymin"] < float(self.gt_image_height)
        )
        return bbox

    def _bbox_overlap_ratio(self, bbox_a, bbox_b):
        xmin = max(float(bbox_a["xmin"]), float(bbox_b["xmin"]))
        ymin = max(float(bbox_a["ymin"]), float(bbox_b["ymin"]))
        xmax = min(float(bbox_a["xmax"]), float(bbox_b["xmax"]))
        ymax = min(float(bbox_a["ymax"]), float(bbox_b["ymax"]))
        if xmax <= xmin or ymax <= ymin:
            return 0.0
        inter_area = (xmax - xmin) * (ymax - ymin)
        area_a = max(
            1e-6,
            (float(bbox_a["xmax"]) - float(bbox_a["xmin"]))
            * (float(bbox_a["ymax"]) - float(bbox_a["ymin"])),
        )
        return float(inter_area / area_a)

    def _filter_occluded_gt_objects(self, visible_objects):
        if len(visible_objects) < 2:
            return visible_objects
        kept = []
        for candidate in sorted(visible_objects, key=lambda item: float(item["depth_range"][0])):
            occluded = False
            for other in kept:
                overlap = self._bbox_overlap_ratio(
                    candidate["projected_bbox"],
                    other["projected_bbox"],
                )
                if overlap < self.gt_occlusion_overlap_threshold:
                    continue
                if float(candidate["depth_range"][0]) > float(other["depth_range"][1]) - self.gt_occlusion_depth_margin:
                    occluded = True
                    break
            if not occluded:
                kept.append(candidate)
        kept.sort(
            key=lambda item: (
                -(item["projected_bbox"]["xmax"] - item["projected_bbox"]["xmin"])
                * (item["projected_bbox"]["ymax"] - item["projected_bbox"]["ymin"]),
                item["prim_path"],
            )
        )
        return kept

    def _collect_gt_visible_objects(self, camera_prim_path=None):
        resolved_camera_prim_path = camera_prim_path or self.gt_camera_prim_path
        if not resolved_camera_prim_path:
            resolved_camera_prim_path = self._resolve_gt_camera_prim_path()
            if not resolved_camera_prim_path:
                return [], ""
            if camera_prim_path is None:
                self.gt_camera_prim_path = resolved_camera_prim_path
        camera_prim = self.world.stage.GetPrimAtPath(resolved_camera_prim_path)
        if not camera_prim or not camera_prim.IsValid():
            return [], resolved_camera_prim_path

        visible_objects = []
        for prim in self._iter_gt_candidate_prims():
            aligned_box = self._compute_world_bbox(prim)
            if aligned_box is None:
                continue
            projected = self._project_bbox_to_image(camera_prim, aligned_box)
            if not projected or not projected["visible"]:
                continue
            center = aligned_box.GetMidpoint()
            extent = aligned_box.GetSize()
            visible_objects.append(
                {
                    "stable_id": str(prim.GetPath()),
                    "prim_path": str(prim.GetPath()),
                    "name": prim.GetName(),
                    "semantic_label": self._semantic_name_from_prim(prim),
                    "bbox_center_world": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                    ],
                    "bbox_extent_world": [
                        float(extent[0]),
                        float(extent[1]),
                        float(extent[2]),
                    ],
                    "projected_bbox": {
                        "xmin": projected["xmin"],
                        "ymin": projected["ymin"],
                        "xmax": projected["xmax"],
                        "ymax": projected["ymax"],
                    },
                    "depth_range": [projected["zmin"], projected["zmax"]],
                }
            )
        visible_objects.sort(
            key=lambda item: (
                -(item["projected_bbox"]["xmax"] - item["projected_bbox"]["xmin"])
                * (item["projected_bbox"]["ymax"] - item["projected_bbox"]["ymin"]),
                item["prim_path"],
            )
        )
        return self._filter_occluded_gt_objects(visible_objects), resolved_camera_prim_path

    def _publish_gt_visible_objects_for_camera(self, publisher, camera_prim_path):
        visible_objects, resolved_camera_prim_path = self._collect_gt_visible_objects(camera_prim_path)
        payload = {
            "stamp_sec": float(self.custom_animation_time_seconds or 0.0),
            "camera_prim_path": resolved_camera_prim_path or "",
            "image_size": [int(self.gt_image_width), int(self.gt_image_height)],
            "count": len(visible_objects),
            "objects": visible_objects,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        publisher.publish(msg)

    def _publish_gt_visible_objects(self):
        self._publish_gt_visible_objects_for_camera(
            self.gt_visible_objects_pub,
            self.gt_camera_prim_path,
        )
        if self.gt_front_camera_prim_path:
            self._publish_gt_visible_objects_for_camera(
                self.gt_visible_objects_front_pub,
                self.gt_front_camera_prim_path,
            )
        if self.gt_rear_camera_prim_path:
            self._publish_gt_visible_objects_for_camera(
                self.gt_visible_objects_rear_pub,
                self.gt_rear_camera_prim_path,
            )

    def _make_pose_from_xyz(self, xyz):
        pose = Pose()
        pose.position.x = float(xyz[0])
        pose.position.y = float(xyz[1])
        pose.position.z = float(xyz[2])
        pose.orientation.w = 1.0
        return pose

    def _collect_chois_entity_state(self):
        if not self.custom_animated_assets:
            return []

        entities = []
        time_code = self._get_current_animation_time_code()
        bbox_cache = UsdGeom.BBoxCache(
            time_code,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )

        def add_entity(asset_key, entity_type, prim):
            if prim is None or not prim.IsValid():
                return
            aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if aligned.IsEmpty():
                return
            center = aligned.GetMidpoint()
            extent = aligned.GetSize()
            min_pt = aligned.GetMin()
            entities.append(
                {
                    "asset_key": asset_key,
                    "type": entity_type,
                    "prim_path": str(prim.GetPath()),
                    "position": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                    ],
                    "ground_position": [
                        float(center[0]),
                        float(center[1]),
                        float(min_pt[2]),
                    ],
                    "bbox_extent": [
                        float(extent[0]),
                        float(extent[1]),
                        float(extent[2]),
                    ],
                }
            )

        for prim_path, asset_meta in sorted(self.custom_animated_assets.items()):
            actor_prim = asset_meta.get("actor_prim")
            if actor_prim is None or not actor_prim.IsValid():
                continue
            asset_key = asset_meta.get("asset_key") or Path(prim_path).name
            body_prim = self.world.stage.GetPrimAtPath(
                f"{actor_prim.GetPath()}/Armature_1/body_1"
            )
            object_prim = self.world.stage.GetPrimAtPath(
                f"{actor_prim.GetPath()}/Animated_Object"
            )
            add_entity(asset_key, "human", body_prim if body_prim and body_prim.IsValid() else actor_prim)
            add_entity(asset_key, "object", object_prim)

        return entities

    def _build_chois_marker(self, entity, marker_id, stamp):
        marker = Marker()
        marker.header.frame_id = self.chois_frame_id
        marker.header.stamp = stamp
        marker.ns = f"chois_{entity['type']}"
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.type = Marker.SPHERE if entity["type"] == "human" else Marker.CUBE
        marker.pose = self._make_pose_from_xyz(entity["ground_position"])
        marker.pose.position.z += 0.45 if entity["type"] == "human" else 0.20
        marker.scale.x = 0.45 if entity["type"] == "human" else 0.35
        marker.scale.y = 0.45 if entity["type"] == "human" else 0.35
        marker.scale.z = 0.90 if entity["type"] == "human" else 0.35
        if entity["type"] == "human":
            marker.color.r = 1.0
            marker.color.g = 0.55
            marker.color.b = 0.05
        else:
            marker.color.r = 0.05
            marker.color.g = 0.95
            marker.color.b = 0.35
        marker.color.a = 0.9
        return marker

    def _build_chois_label_marker(self, entity, marker_id, stamp):
        marker = Marker()
        marker.header.frame_id = self.chois_frame_id
        marker.header.stamp = stamp
        marker.ns = "chois_labels"
        marker.id = marker_id
        marker.action = Marker.ADD
        marker.type = Marker.TEXT_VIEW_FACING
        marker.pose = self._make_pose_from_xyz(entity["ground_position"])
        marker.pose.position.z += 1.15 if entity["type"] == "human" else 0.65
        marker.scale.z = 0.35
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.95
        marker.text = f"{entity['asset_key']} {entity['type']}"
        return marker

    def _publish_chois_state(self):
        entities = self._collect_chois_entity_state()
        stamp = self.get_clock().now().to_msg()

        human_poses = PoseArray()
        human_poses.header.frame_id = self.chois_frame_id
        human_poses.header.stamp = stamp
        object_poses = PoseArray()
        object_poses.header.frame_id = self.chois_frame_id
        object_poses.header.stamp = stamp

        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)

        for index, entity in enumerate(entities):
            pose = self._make_pose_from_xyz(entity["ground_position"])
            if entity["type"] == "human":
                human_poses.poses.append(pose)
            elif entity["type"] == "object":
                object_poses.poses.append(pose)
            markers.markers.append(self._build_chois_marker(entity, index, stamp))
            markers.markers.append(self._build_chois_label_marker(entity, index + 1000, stamp))

        payload = {
            "stamp_sec": float(self.custom_animation_time_seconds or 0.0),
            "frame_id": self.chois_frame_id,
            "count": len(entities),
            "entities": entities,
        }
        state_msg = String()
        state_msg.data = json.dumps(payload, ensure_ascii=True)

        self.chois_state_pub.publish(state_msg)
        self.chois_human_poses_pub.publish(human_poses)
        self.chois_object_poses_pub.publish(object_poses)
        self.chois_markers_pub.publish(markers)

    def _build_debug_curve_points(self, points_xy, z_value):
        return [
            Gf.Vec3f(float(point[0]), float(point[1]), float(z_value))
            for point in np.asarray(points_xy)[:, :2]
        ]

    def _create_debug_curve(self, prim_path: str, points, color_rgb, width=0.12):
        if not points:
            return None

        stage = self.world.stage
        debug_points = UsdGeom.Points.Define(stage, prim_path)
        debug_points.GetPointsAttr().Set(points)
        debug_points.GetWidthsAttr().Set([float(width)] * len(points))
        debug_points.GetDisplayColorAttr().Set([Gf.Vec3f(*color_rgb)])
        return debug_points

    def _update_debug_curve(self, curve, points):
        if curve is None or not points:
            return
        curve.GetPointsAttr().Set(points)
        curve.GetWidthsAttr().Set([float(curve.GetWidthsAttr().Get()[0])] * len(points))

    def _get_current_animation_time_code(self):
        if self.custom_animated_time_range is None:
            return Usd.TimeCode.Default()

        fps = float(self.custom_animated_time_range["fps"])
        if fps <= 0.0:
            return Usd.TimeCode.Default()

        if self.custom_animation_time_seconds is None:
            return Usd.TimeCode(self.custom_animated_time_range["start_time_code"])

        return Usd.TimeCode(self.custom_animation_time_seconds * fps)

    def _compute_actor_debug_world_position(self, actor_prim):
        time_code = self._get_current_animation_time_code()
        bbox_cache = UsdGeom.BBoxCache(
            time_code,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        world_bound = bbox_cache.ComputeWorldBound(actor_prim)
        aligned_range = world_bound.ComputeAlignedBox()
        center = aligned_range.GetMidpoint()
        min_z = aligned_range.GetMin()[2]
        return Gf.Vec3f(float(center[0]), float(center[1]), float(min_z))

    def _initialize_custom_animated_debug(self, actor_prim, animated_entry):
        asset_key = animated_entry["asset_key"]
        waypoint_path = self._resolve_chois_waypoints_path(asset_key)
        expected_waypoints = None
        expected_curve = None
        if waypoint_path and os.path.exists(waypoint_path):
            expected_waypoints = np.asarray(np.load(waypoint_path), dtype=np.float32)
            expected_curve = self._create_debug_curve(
                f"/World/CHOISDebug/{self._prim_safe_name(asset_key)}_ExpectedPath",
                self._build_debug_curve_points(expected_waypoints, DEFAULT_CHOIS_ACTOR_Z),
                (1.0, 0.1, 0.1),
                width=0.18,
            )

        initial_actual_point = self._compute_actor_debug_world_position(actor_prim)
        actual_curve = self._create_debug_curve(
            f"/World/CHOISDebug/{self._prim_safe_name(asset_key)}_ActualPath",
            [initial_actual_point],
            (0.1, 1.0, 0.1),
            width=0.22,
        )
        self.custom_animated_debug_data[animated_entry["prim_path"]] = {
            "asset_key": asset_key,
            "actor_prim": actor_prim,
            "waypoint_path": waypoint_path,
            "expected_waypoints": expected_waypoints,
            "expected_curve": expected_curve,
            "actual_curve": actual_curve,
            "actual_points": [initial_actual_point],
        }

    def _sample_custom_animated_debug_paths(self):
        if not self.enable_chois_debug_visualization:
            return
        for debug_entry in self.custom_animated_debug_data.values():
            world_position = self._compute_actor_debug_world_position(
                debug_entry["actor_prim"]
            )
            actual_points = debug_entry["actual_points"]
            if actual_points:
                prev_point = actual_points[-1]
                delta = np.linalg.norm(
                    np.asarray(
                        [
                            float(world_position[0]) - float(prev_point[0]),
                            float(world_position[1]) - float(prev_point[1]),
                            float(world_position[2]) - float(prev_point[2]),
                        ],
                        dtype=np.float32,
                    )
                )
                if delta < 1e-4:
                    continue
            actual_points.append(world_position)
            self._update_debug_curve(debug_entry["actual_curve"], actual_points)
            self._save_single_custom_animated_debug_path(debug_entry)

    def _save_single_custom_animated_debug_path(self, debug_entry):
        asset_key = debug_entry["asset_key"]
        safe_name = self._prim_safe_name(asset_key)
        actual_points = np.asarray(
            [
                [float(point[0]), float(point[1]), float(point[2])]
                for point in debug_entry["actual_points"]
            ],
            dtype=np.float32,
        )
        actual_path = Path(self.chois_debug_output_root) / f"{safe_name}_actual_path.npy"
        if actual_points.size:
            np.save(actual_path, actual_points)

        expected_waypoints = debug_entry["expected_waypoints"]
        expected_path = Path(self.chois_debug_output_root) / f"{safe_name}_expected_path.npy"
        if expected_waypoints is not None:
            np.save(expected_path, np.asarray(expected_waypoints[:, :2], dtype=np.float32))

        return actual_path, expected_path

    def _save_custom_animated_debug_paths(self):
        if not self.enable_chois_debug_visualization:
            return
        for debug_entry in self.custom_animated_debug_data.values():
            actual_path, expected_path = self._save_single_custom_animated_debug_path(
                debug_entry
            )
            print(
                f"[TeleopHuNavSim] Saved debug trajectories for {debug_entry['asset_key']} "
                f"to {actual_path} and {expected_path}"
            )

    def _clicked_point_callback(self, msg: PointStamped):
        point = [
            float(msg.point.x),
            float(msg.point.y),
            float(msg.point.z),
        ]
        self.clicked_points.append(point)
        points_array = np.asarray(self.clicked_points, dtype=np.float32)
        np.save(self.clicked_points_output_path, points_array)
        self.get_logger().info(
            f"Saved clicked point #{len(self.clicked_points)} to "
            f"{self.clicked_points_output_path}: {point}"
        )

    def _goal_pose_callback(self, msg: PoseStamped):
        q = msg.pose.orientation
        yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        goal_pose = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(yaw),
        ]
        self.goal_poses.append(goal_pose)
        goal_poses_array = np.asarray(self.goal_poses, dtype=np.float32)
        np.save(self.goal_poses_output_path, goal_poses_array)
        self.get_logger().info(
            f"Saved goal pose #{len(self.goal_poses)} to "
            f"{self.goal_poses_output_path}: {goal_pose}"
        )

    def _spawn_custom_animated_usd(self, usd_path: str, prim_path: str, placement):
        actor_prim = self.builder.add_usd_reference(usd_path, prim_path)
        xform = UsdGeom.Xformable(actor_prim)
        transform_op = None
        translate_op = None
        orient_op = None
        for op in xform.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:transform":
                transform_op = op
            elif op.GetOpName() == "xformOp:translate":
                translate_op = op
            elif op.GetOpName() == "xformOp:orient":
                orient_op = op
        if transform_op is None:
            transform_op = xform.AddTransformOp()

        position = placement["position"]
        yaw_deg = placement["yaw_deg"]
        projection_label = placement.get("projection", "+x,+y")
        projection_matrix_2d = self._projection_matrix_from_label(projection_label)
        yaw_rad = float(np.radians(yaw_deg))
        rotation_2d = np.asarray(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad)],
                [np.sin(yaw_rad), np.cos(yaw_rad)],
            ],
            dtype=np.float32,
        )
        linear_2d = rotation_2d @ projection_matrix_2d

        transform = Gf.Matrix4d(1.0)
        transform.SetRow(
            0, Gf.Vec4d(float(linear_2d[0, 0]), float(linear_2d[0, 1]), 0.0, 0.0)
        )
        transform.SetRow(
            1, Gf.Vec4d(float(linear_2d[1, 0]), float(linear_2d[1, 1]), 0.0, 0.0)
        )
        transform.SetRow(2, Gf.Vec4d(0.0, 0.0, 1.0, 0.0))
        transform.SetRow(
            3,
            Gf.Vec4d(
                float(position[0]), float(position[1]), float(position[2]), 1.0
            ),
        )
        transform_op.Set(transform)

        if translate_op is not None:
            translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        if orient_op is not None:
            orient_op.Set(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        print(
            f"[TeleopHuNavSim] Spawned {usd_path} at {position} "
            f"with projection={projection_label}, yaw={yaw_deg:.3f} deg"
        )

        asset_stage = Usd.Stage.Open(usd_path)
        if asset_stage:
            start_time_code = float(asset_stage.GetStartTimeCode())
            end_time_code = float(asset_stage.GetEndTimeCode())
            fps = float(asset_stage.GetTimeCodesPerSecond() or 24.0)
            start_time_seconds = start_time_code / fps
            end_time_seconds = end_time_code / fps
            stage = self.world.stage
            stage.SetTimeCodesPerSecond(fps)
            stage.SetStartTimeCode(start_time_code)
            stage.SetEndTimeCode(end_time_code)
            if self.custom_animated_time_range is None:
                self.custom_animated_time_range = {
                    "start_time_code": start_time_code,
                    "end_time_code": end_time_code,
                    "fps": fps,
                    "start_time_seconds": start_time_seconds,
                    "end_time_seconds": end_time_seconds,
                    "duration_seconds": max(end_time_seconds - start_time_seconds, 1.0 / fps),
                }
            else:
                self.custom_animated_time_range["start_time_code"] = min(
                    self.custom_animated_time_range["start_time_code"], start_time_code
                )
                self.custom_animated_time_range["end_time_code"] = max(
                    self.custom_animated_time_range["end_time_code"], end_time_code
                )
                self.custom_animated_time_range["start_time_seconds"] = min(
                    self.custom_animated_time_range["start_time_seconds"], start_time_seconds
                )
                self.custom_animated_time_range["end_time_seconds"] = max(
                    self.custom_animated_time_range["end_time_seconds"], end_time_seconds
                )
                self.custom_animated_time_range["duration_seconds"] = max(
                    self.custom_animated_time_range["end_time_seconds"]
                    - self.custom_animated_time_range["start_time_seconds"],
                    1.0 / self.custom_animated_time_range["fps"],
                )
            self.custom_animation_time_seconds = start_time_seconds
            self.custom_animated_assets[prim_path] = {
                "prim_path": prim_path,
                "actor_prim": actor_prim,
                "usd_path": usd_path,
                "start_time_code": start_time_code,
                "end_time_code": end_time_code,
                "time_span_code": max(end_time_code - start_time_code, 1.0),
                "fps": fps,
                "start_time_seconds": start_time_seconds,
                "end_time_seconds": end_time_seconds,
                "duration_seconds": max(end_time_seconds - start_time_seconds, 1.0 / fps),
                "wrap_count": 0,
            }
            print(
                f"[TeleopHuNavSim] Loaded animated USD time range "
                f"{start_time_code} -> {end_time_code} timeCodes "
                f"({start_time_seconds:.3f}s -> {end_time_seconds:.3f}s) at {fps} fps"
            )
        else:
            print(
                f"[TeleopHuNavSim] Warning: could not open animated USD "
                f"for time range metadata: {usd_path}"
            )
        self._apply_chois_actor_semantics(actor_prim)
        return actor_prim

    def _start_custom_animation_loop(self):
        timeline = omni.timeline.get_timeline_interface()
        if timeline is None:
            print("[TeleopHuNavSim] Timeline interface unavailable.")
            return
        if self.custom_animated_time_range is not None:
            start_time = self.custom_animated_time_range["start_time_seconds"]
            end_time = self.custom_animated_time_range["end_time_seconds"]
            if hasattr(timeline, "set_start_time"):
                timeline.set_start_time(start_time)
            if hasattr(timeline, "set_end_time"):
                timeline.set_end_time(end_time)
            if hasattr(timeline, "set_current_time"):
                timeline.set_current_time(start_time)
            if hasattr(timeline, "set_auto_update"):
                timeline.set_auto_update(True)
        if hasattr(timeline, "set_looping"):
            timeline.set_looping(True)
        if hasattr(timeline, "play"):
            timeline.play()
        print("[TeleopHuNavSim] Timeline playback started in loop mode.")

    def _advance_custom_animation_loop(self, dt: float):
        """
        Manually wrap timeline time so the referenced USD animation loops even when
        Isaac ignores the timeline's native looping flag for this asset type.
        """
        if self.custom_animated_time_range is None:
            return

        timeline = omni.timeline.get_timeline_interface()
        if timeline is None or not hasattr(timeline, "set_current_time"):
            return

        start_time = self.custom_animated_time_range["start_time_seconds"]
        duration = self.custom_animated_time_range["duration_seconds"]
        if self.custom_animation_time_seconds is None:
            self.custom_animation_time_seconds = start_time
        else:
            elapsed = (self.custom_animation_time_seconds - start_time) + dt
            self.custom_animation_time_seconds = start_time + (elapsed % duration)

        timeline.set_current_time(self.custom_animation_time_seconds)
        self._update_independent_custom_animation_loops()

    def _set_actor_reference_time_offset(self, actor_prim, usd_path: str, offset_time_codes: float):
        if actor_prim is None:
            return
        references = actor_prim.GetReferences()
        if references is None:
            return
        references.ClearReferences()
        references.AddReference(
            usd_path,
            layerOffset=Sdf.LayerOffset(offset=float(offset_time_codes), scale=1.0),
        )
        self._apply_chois_actor_semantics(actor_prim)

    def _update_independent_custom_animation_loops(self):
        if not self.custom_animated_assets:
            return
        if self.custom_animation_time_seconds is None:
            return

        for asset_meta in self.custom_animated_assets.values():
            duration_seconds = float(asset_meta["duration_seconds"])
            if duration_seconds <= 0.0:
                continue

            elapsed_seconds = self.custom_animation_time_seconds - float(
                asset_meta["start_time_seconds"]
            )
            if elapsed_seconds < 0.0:
                wrap_count = 0
            else:
                wrap_count = int(np.floor(elapsed_seconds / duration_seconds))

            if wrap_count == int(asset_meta["wrap_count"]):
                continue

            asset_meta["wrap_count"] = wrap_count
            offset_time_codes = -wrap_count * float(asset_meta["time_span_code"])
            try:
                self._set_actor_reference_time_offset(
                    asset_meta["actor_prim"],
                    asset_meta["usd_path"],
                    offset_time_codes,
                )
                print(
                    f"[TeleopHuNavSim] Looping {asset_meta['prim_path']} independently "
                    f"with reference offset {offset_time_codes:.3f} timeCodes"
                )
            except Exception as exc:
                print(
                    f"[TeleopHuNavSim] Failed to update loop offset for "
                    f"{asset_meta['prim_path']}: {exc}"
                )

    def _signal_handler(self, signum, frame):
        print("\n\nCaught shutdown signal, closing app and stopping hunav nodes...\n\n")
        self._save_custom_animated_debug_paths()
        self.hunav.close_hunav_nodes()
        simulation_app.close()

    def _cmd_vel_callback(self, msg):
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def _on_physics_step(self, dt: float):
        """
        Kept for compatibility, but HuNav updates now run after `world.step()`
        so scripted USD animation has already updated the prim transforms.
        """
        return

    def create_ros_clock_action_graph(self, graph_path="/World/ROS2"):
        try:
            keys = og.Controller.Keys
            graph_params = {
                # Create the necessary nodes.
                keys.CREATE_NODES: [
                    # Node for generating a tick on playback.
                    ("on_playback_tick", "omni.graph.action.OnPlaybackTick"),
                    # Node for reading the simulation time.
                    (
                        "isaac_read_simulation_time",
                        "isaacsim.core.nodes.IsaacReadSimulationTime",
                    ),
                    # Node to create a ROS2 context.
                    ("ros2_context", "isaacsim.ros2.bridge.ROS2Context"),
                    # Node to publish the clock over ROS2.
                    ("ros2_publish_clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                # Connect outputs to inputs:
                keys.CONNECT: [
                    # Connect context output to the publish clock's context input.
                    (
                        "ros2_context.outputs:context",
                        "ros2_publish_clock.inputs:context",
                    ),
                    # Connect tick output to publish clock's execIn.
                    (
                        "on_playback_tick.outputs:tick",
                        "ros2_publish_clock.inputs:execIn",
                    ),
                    # Connect simulation time output to publish clock's timeStamp.
                    (
                        "isaac_read_simulation_time.outputs:simulationTime",
                        "ros2_publish_clock.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    # For the simulation time node.
                    ("isaac_read_simulation_time.inputs:resetOnStop", True),
                    ("isaac_read_simulation_time.inputs:swhFrameNumber", 0),
                    # For the ROS2PublishClock node.
                    ("ros2_publish_clock.inputs:nodeNamespace", ""),
                    ("ros2_publish_clock.inputs:qosProfile", ""),
                    ("ros2_publish_clock.inputs:queueSize", 10),
                    ("ros2_publish_clock.inputs:timeStamp", 0.0),
                    ("ros2_publish_clock.inputs:topicName", "clock"),
                    # For the ROS2Context node.
                    ("ros2_context.inputs:useDomainIDEnvVar", True),
                    ("ros2_context.inputs:domain_id", 0),
                ],
            }
            og.Controller.edit(
                {"graph_path": graph_path, "evaluator_name": "execution"},
                graph_params,
            )
            print(f"Successfully created ROS_Clock action graph at {graph_path}")
        except Exception as e:
            print(f"Error creating ROS_Clock action graph: {e}")

    def run(self):
        self.world.reset()
        self._start_custom_animation_loop()
        self.hunav.send_agents_msg()
        try:
            while simulation_app.is_running():
                self._advance_custom_animation_loop(self.world.get_physics_dt())
                self.world.step(render=True)
                self.gt_publish_step_counter += 1
                if self.gt_publish_step_counter % self.gt_publish_every_n_steps == 0:
                    self._publish_gt_visible_objects()
                self.chois_publish_step_counter += 1
                if self.chois_publish_step_counter % self.chois_publish_every_n_steps == 0:
                    self._publish_chois_state()
                if self.enable_chois_debug_visualization:
                    self._sample_custom_animated_debug_paths()
                self.hunav.send_agents_msg()
                wheel_action = self.diff_controller.forward([self.cmd_lin, self.cmd_ang])
                self.robot.apply_wheel_actions(wheel_action)
        finally:
            if self.enable_chois_debug_visualization:
                self._save_custom_animated_debug_paths()
