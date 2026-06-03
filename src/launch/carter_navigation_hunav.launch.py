#!/usr/bin/env python3
"""HuNav-controlled Carter Nav2 launch.

This keeps the selected static map on /nav2_map so Isaac Sim's own /map
publisher cannot override the map shown by Nav2/RViz.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def _launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration("world").perform(context)
    map_file = LaunchConfiguration("map").perform(context)
    params_file = LaunchConfiguration("params_file").perform(context)
    map_topic = LaunchConfiguration("map_topic").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context)
    use_sim_time = LaunchConfiguration("use_sim_time")

    wrapper_share = get_package_share_directory("hunav_isaac_wrapper")
    carter_share = get_package_share_directory("carter_navigation")
    nav2_bringup_launch_dir = os.path.join(get_package_share_directory("nav2_bringup"), "launch")

    if not map_file:
        env_map = os.environ.get("HUNAV_MAP", "")
        map_file = env_map or os.path.join(wrapper_share, "maps", f"{world}.yaml")

    if not params_file:
        env_params = os.environ.get("HUNAV_NAV2_PARAMS", "")
        params_file = env_params or os.path.join(
            wrapper_share,
            "config",
            "navigation_params",
            "carter_navigation_params.yaml",
        )

    if not rviz_config:
        env_rviz_config = os.environ.get("HUNAV_RVIZ_CONFIG", "")
        chois_rviz_config = "/workspace/hunav_isaac_ws/config/carter_navigation_hunav_chois.rviz"
        if env_rviz_config:
            rviz_config = env_rviz_config
        elif os.path.exists(chois_rviz_config):
            rviz_config = chois_rviz_config
        else:
            rviz_config = os.path.join(carter_share, "rviz2", "carter_navigation.rviz")

    return [
        LogInfo(msg=[f"HuNav Carter Navigation world: {world}"]),
        LogInfo(msg=[f"HuNav Carter Navigation map: {map_file}"]),
        LogInfo(msg=[f"HuNav Carter Navigation map topic: {map_topic}"]),
        LogInfo(msg=[f"HuNav Carter Navigation RViz config: {rviz_config}"]),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_config],
            output="screen",
            remappings=[
                ("/map", map_topic),
                ("map", map_topic),
            ],
        ),
        GroupAction(
            [
                SetRemap(src="/map", dst=map_topic),
                SetRemap(src="map", dst=map_topic),
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource([nav2_bringup_launch_dir, "/bringup_launch.py"]),
                    launch_arguments={
                        "map": map_file,
                        "use_sim_time": use_sim_time,
                        "params_file": params_file,
                    }.items(),
                ),
            ]
        ),
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="pointcloud_to_laserscan",
            remappings=[
                ("cloud_in", ["/front_3d_lidar/lidar_points"]),
                ("scan", ["/scan"]),
            ],
            parameters=[
                {
                    "target_frame": "front_3d_lidar",
                    "transform_tolerance": 0.01,
                    "min_height": -0.4,
                    "max_height": 1.5,
                    "angle_min": -1.5708,
                    "angle_max": 1.5708,
                    "angle_increment": 0.0087,
                    "scan_time": 0.3333,
                    "range_min": 0.05,
                    "range_max": 100.0,
                    "use_inf": True,
                    "inf_epsilon": 1.0,
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world",
                default_value=EnvironmentVariable("HUNAV_WORLD", default_value="warehouse"),
                description="HuNav world name used to choose maps/<world>.yaml when map is not set",
            ),
            DeclareLaunchArgument(
                "map",
                default_value="",
                description="Full path to map YAML. Defaults to $HUNAV_MAP or hunav_isaac_wrapper/maps/<world>.yaml",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description="Full path to Nav2 params YAML. Defaults to $HUNAV_NAV2_PARAMS or the wrapper config.",
            ),
            DeclareLaunchArgument(
                "map_topic",
                default_value="/nav2_map",
                description="Topic used by Nav2 and RViz for the selected static map.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=EnvironmentVariable("HUNAV_RVIZ_CONFIG", default_value=""),
                description=(
                    "Full path to RViz config. Defaults to $HUNAV_RVIZ_CONFIG, "
                    "/workspace/hunav_isaac_ws/config/carter_navigation_hunav_chois.rviz, "
                    "or carter_navigation.rviz."
                ),
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation clock.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
