#!/usr/bin/env python3
"""
Launch file for HuNav Isaac Wrapper simulation.
Launches the main script using the ROS2 launcher, which handles Isaac Sim python detection.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description that uses the ROS2 launcher."""
    
    # Declare launch arguments
    scenario_arg = DeclareLaunchArgument(
        'scenario',
        default_value='',
        description='Scenario configuration file to use (optional - will use interactive mode if not specified)'
    )
    
    batch_arg = DeclareLaunchArgument(
        'batch',
        default_value='false',
        description='Run in batch mode (non-interactive)'
    )
    
    extra_args_arg = DeclareLaunchArgument(
        'extra_args',
        default_value='',
        description='Additional arguments to pass to the main script'
    )
    
    # Launch configurations
    scenario = LaunchConfiguration('scenario')
    batch = LaunchConfiguration('batch')
    extra_args = LaunchConfiguration('extra_args')
    
    # Use the ROS2 launcher which handles Isaac Sim python detection
    launcher_with_scenario = ExecuteProcess(
        cmd=['ros2', 'run', 'hunav_isaac_wrapper', 'hunav_isaac_launcher',
             '--config', scenario, '--batch', extra_args],
        condition=IfCondition(scenario),
        output='screen',
        name='hunav_isaac_launcher_scenario'
    )
    
    # Interactive mode (default when no scenario specified)
    launcher_interactive = ExecuteProcess(
        cmd=['ros2', 'run', 'hunav_isaac_wrapper', 'hunav_isaac_launcher', extra_args],
        condition=UnlessCondition(scenario),
        output='screen',
        name='hunav_isaac_launcher_interactive'
    )

    segmentation_colorizer = Node(
        package='seg_vis',
        executable='segmentation_colorizer',
        name='segmentation_colorizer',
        output='screen',
    )

    gt_rviz_markers = Node(
        package='gt_vis',
        executable='gt_rviz_markers',
        name='gt_rviz_markers',
        output='screen',
        parameters=[{
            'world': '',
            'marker_topic': '/gt/markers',
            'chois_action_topic': '/gt/chois_actions',
            'hunav_topic': 'human_states',
            'chois_topic': '/chois/state',
        }],
    )
    
    return LaunchDescription([
        # Launch arguments
        scenario_arg,
        batch_arg,
        extra_args_arg,
        
        # Log launch info
        LogInfo(msg=['Launching HuNav Isaac Wrapper...']),
        LogInfo(msg=['Use scenario parameter to specify a scenario file']),
        LogInfo(msg=['Otherwise interactive mode will start']),
        LogInfo(msg=['Starting segmentation colorizer bridge...']),
        LogInfo(msg=['Starting GT RViz marker bridge...']),
        
        # Bridge and launcher processes
        segmentation_colorizer,
        gt_rviz_markers,
        launcher_with_scenario,
        launcher_interactive,
    ])
