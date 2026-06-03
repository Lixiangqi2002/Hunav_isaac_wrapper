#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import rclpy
import yaml
from geometry_msgs.msg import Point
from hunav_msgs.msg import Agents
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


Color = tuple[float, float, float, float]

STATIC_COLOR: Color = (1.0, 0.62, 0.05, 0.95)
HUNAV_COLOR: Color = (0.05, 0.55, 1.0, 0.9)
CHOIS_HUMAN_COLOR: Color = (1.0, 0.45, 0.05, 0.9)
CHOIS_OBJECT_COLOR: Color = (0.0, 0.9, 0.35, 0.9)
CHOIS_PAIR_COLOR: Color = (1.0, 1.0, 1.0, 0.8)
TEXT_COLOR: Color = (1.0, 1.0, 1.0, 0.95)

VALID_WORLDS = {"warehouse", "hospital", "office"}
CHOIS_YAW_MIN_DISPLACEMENT_M = 0.01
CHOIS_YAW_MAX_JUMP_M = 2.0


def as_float_list(value: Any, length: int = 3) -> list[float] | None:
    if isinstance(value, dict) and all(key in value for key in ("x", "y", "z")):
        return [float(value["x"]), float(value["y"]), float(value["z"])][:length]
    if isinstance(value, (list, tuple)) and len(value) >= length:
        return [float(value[index]) for index in range(length)]
    return None


def entity_position(entity: dict[str, Any]) -> list[float] | None:
    # Prefer live world bbox center. ground_position is only used for z anchoring.
    for key in ("position", "bbox_center", "translation", "xyz", "ground_position"):
        position = as_float_list(entity.get(key))
        if position is not None:
            return position
    pose = entity.get("pose")
    if isinstance(pose, dict):
        for key in ("position", "translation"):
            position = as_float_list(pose.get(key))
            if position is not None:
                return position
    return as_float_list(pose)


def infer_world(value: str | None) -> str:
    candidate = (value or os.environ.get("HUNAV_GT_WORLD") or os.environ.get("HUNAV_WORLD") or "").strip().lower()
    if candidate in VALID_WORLDS:
        return candidate
    current = Path("/workspace/hunav_isaac_ws/src/Hunav_isaac_wrapper/.current_hunav_world")
    if current.exists():
        for line in current.read_text(encoding="utf-8").splitlines():
            if line.startswith("export HUNAV_WORLD="):
                candidate = line.split("=", 1)[1].strip().strip('"').lower()
                if candidate in VALID_WORLDS:
                    return candidate
    return "warehouse"


def annotation_roots() -> list[Path]:
    roots = []
    if os.environ.get("GT_ANNOTATION_ROOT"):
        roots.append(Path(os.environ["GT_ANNOTATION_ROOT"]))
    roots.extend(
        [
            Path("/workspace/gt_annotation/annotations"),
            Path("/data/code/hunavsim_docker/isaac_sim/gt_annotation/annotations"),
        ]
    )
    return roots


def static_candidates(world: str) -> list[Path]:
    paths = []
    for root in annotation_roots():
        paths.append(root / "05_duplicate_review" / world / f"{world}_no_duplicate.json")
        paths.append(root / "03_static_extraction" / world / "static_objects.json")
    return paths


def chois_action_candidates(world: str) -> list[Path]:
    return [root / "01_chois_actions" / f"{world}.yaml" for root in annotation_roots()]


def load_chois_actions(world: str, explicit_path: str | None = None) -> tuple[Path | None, dict[str, dict[str, Any]]]:
    candidates = [Path(explicit_path)] if explicit_path else chois_action_candidates(world)
    for path in candidates:
        path = path.expanduser()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        raw_assets = data.get("assets") or {}
        if isinstance(raw_assets, dict):
            return path, {str(key): dict(value or {}) for key, value in raw_assets.items()}
        return path, {
            str(item.get("asset_key")): dict(item)
            for item in raw_assets
            if isinstance(item, dict) and item.get("asset_key")
        }
    return None, {}


def load_static_objects(world: str, explicit_path: str | None, foreground_only: bool) -> tuple[Path | None, list[dict[str, Any]]]:
    candidates = [Path(explicit_path)] if explicit_path else static_candidates(world)
    for path in candidates:
        path = path.expanduser()
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        objects = payload.get("objects", [])
        if foreground_only:
            objects = [
                obj
                for obj in objects
                if obj.get("is_foreground") is True or obj.get("review_status") == "foreground"
            ]
        objects = [
            obj
            for obj in objects
            if as_float_list(obj.get("bbox_center")) is not None
            and as_float_list(obj.get("bbox_extent")) is not None
        ]
        return path, objects
    return None, []


def quaternion_from_yaw(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def frame_state_from_stamp(stamp_sec: float, asset: dict[str, Any]) -> dict[str, int | float]:
    fps = float(asset.get("fps") or 24.0)
    start_frame = int(asset.get("start_frame") or 1)
    end_frame = int(asset.get("end_frame") or start_frame)
    duration_frames = int(asset.get("duration_frames") or max(end_frame - start_frame + 1, 1))
    absolute_frame = int(round(float(stamp_sec) * fps))
    elapsed_frames = absolute_frame - start_frame
    local_index = elapsed_frames % max(duration_frames, 1)
    cycle_index = elapsed_frames // max(duration_frames, 1) if elapsed_frames >= 0 else 0
    source_frame = start_frame + local_index
    return {
        "fps": fps,
        "source_frame": int(source_frame),
        "cycle_index": int(cycle_index),
        "local_frame_index": int(local_index),
        "source_time_sec": float(source_frame / fps),
    }


def active_segment(asset: dict[str, Any], source_frame: int) -> dict[str, Any] | None:
    for segment in asset.get("segments") or []:
        start = int(segment.get("start_frame") or 0)
        end = int(segment.get("end_frame") or 0)
        if start <= int(source_frame) <= end:
            return {
                "action": str(segment.get("action") or "unlabeled"),
                "start_frame": start,
                "end_frame": end,
            }
    return None


class GtRvizMarkers(Node):
    def __init__(self) -> None:
        super().__init__("gt_rviz_markers")
        self.declare_parameter("world", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("marker_topic", "/gt/markers")
        self.declare_parameter("chois_action_topic", "/gt/chois_actions")
        self.declare_parameter("hunav_topic", "human_states")
        self.declare_parameter("chois_topic", "/chois/state")
        self.declare_parameter("static_path", "")
        self.declare_parameter("chois_action_path", "")
        self.declare_parameter("foreground_only", True)
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("show_static_labels", False)
        self.declare_parameter("dynamic_timeout_sec", 2.5)

        self.world = infer_world(str(self.get_parameter("world").value or ""))
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.show_static_labels = bool(self.get_parameter("show_static_labels").value)
        self.dynamic_timeout_sec = float(self.get_parameter("dynamic_timeout_sec").value)
        foreground_only = bool(self.get_parameter("foreground_only").value)
        static_path_param = str(self.get_parameter("static_path").value or "")
        chois_action_path_param = str(self.get_parameter("chois_action_path").value or "")
        marker_topic = str(self.get_parameter("marker_topic").value)
        chois_action_topic = str(self.get_parameter("chois_action_topic").value)
        hunav_topic = str(self.get_parameter("hunav_topic").value)
        chois_topic = str(self.get_parameter("chois_topic").value)
        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)

        static_path, self.static_objects = load_static_objects(
            self.world,
            static_path_param or None,
            foreground_only,
        )
        if static_path is None:
            self.get_logger().warn(f"No GT static JSON found for world={self.world}")
        else:
            self.get_logger().info(f"Loaded {len(self.static_objects)} static objects from {static_path}")
        chois_action_path, self.chois_actions = load_chois_actions(
            self.world,
            chois_action_path_param or None,
        )
        if chois_action_path is None:
            self.get_logger().warn(f"No CHOIS action YAML found for world={self.world}")
        else:
            self.get_logger().info(f"Loaded {len(self.chois_actions)} CHOIS action annotations from {chois_action_path}")

        self.latest_hunav_agents: list[Any] = []
        self.latest_hunav_time: float | None = None
        self.latest_chois_payload: dict[str, Any] | None = None
        self.latest_chois_time: float | None = None
        self.latest_chois_actions_by_asset: dict[str, dict[str, Any]] = {}
        self.chois_previous_positions: dict[str, list[float]] = {}
        self.chois_yaws: dict[str, float] = {}

        self.publisher = self.create_publisher(MarkerArray, marker_topic, 10)
        self.chois_action_pub = self.create_publisher(String, chois_action_topic, 10)
        self.create_subscription(Agents, hunav_topic, self.on_hunav_agents, 10)
        self.create_subscription(String, chois_topic, self.on_chois_state, 10)
        self.create_timer(1.0 / max(0.1, publish_rate_hz), self.publish_markers)

        self.get_logger().info(f"Subscribing to HuNav agents on {hunav_topic}")
        self.get_logger().info(f"Subscribing to CHOIS state on {chois_topic}")
        self.get_logger().info(f"Publishing GT markers on {marker_topic}")
        self.get_logger().info(f"Publishing CHOIS GT actions on {chois_action_topic}")

    def now_sec(self) -> float:
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def on_hunav_agents(self, msg: Agents) -> None:
        self.latest_hunav_agents = list(msg.agents)
        self.latest_hunav_time = self.now_sec()

    def on_chois_state(self, msg: String) -> None:
        try:
            self.latest_chois_payload = json.loads(msg.data)
            self.latest_chois_time = self.now_sec()
            self.update_chois_motion_yaws(self.latest_chois_payload)
            self.publish_chois_actions()
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse CHOIS state JSON: {exc}")

    def chois_entity_key(self, entity: dict[str, Any], fallback_index: int = 0) -> str:
        asset_key = str(entity.get("asset_key") or f"asset_{fallback_index}")
        entity_type = str(entity.get("type") or "object")
        return f"{asset_key}:{entity_type}"

    def update_chois_motion_yaws(self, payload: dict[str, Any]) -> None:
        for index, entity in enumerate(payload.get("entities", [])):
            if not isinstance(entity, dict):
                continue
            position = entity_position(entity)
            if position is None:
                continue
            key = self.chois_entity_key(entity, index)
            previous = self.chois_previous_positions.get(key)
            current = [float(position[0]), float(position[1]), float(position[2])]
            if previous is None:
                self.chois_previous_positions[key] = current
                explicit_yaw = entity.get("yaw")
                if explicit_yaw not in (None, ""):
                    self.chois_yaws.setdefault(key, float(explicit_yaw))
                continue

            dx = current[0] - previous[0]
            dy = current[1] - previous[1]
            distance = math.hypot(dx, dy)
            if distance > CHOIS_YAW_MAX_JUMP_M:
                self.chois_previous_positions[key] = current
                continue
            if distance >= CHOIS_YAW_MIN_DISPLACEMENT_M:
                self.chois_yaws[key] = math.atan2(dy, dx)
                self.chois_previous_positions[key] = current

    def chois_yaw_for_entity(self, entity: dict[str, Any], fallback_index: int = 0) -> float:
        key = self.chois_entity_key(entity, fallback_index)
        if key in self.chois_yaws:
            return self.chois_yaws[key]
        explicit_yaw = entity.get("yaw")
        return float(explicit_yaw or 0.0)

    def build_chois_action_payload(self) -> dict[str, Any]:
        payload = self.latest_chois_payload or {}
        stamp_sec = float(payload.get("stamp_sec") or 0.0)
        entities_by_asset: dict[str, dict[str, dict[str, Any]]] = {}
        for entity in payload.get("entities", []):
            if not isinstance(entity, dict):
                continue
            asset_key = str(entity.get("asset_key") or "")
            entity_type = str(entity.get("type") or "")
            if not asset_key or not entity_type:
                continue
            entities_by_asset.setdefault(asset_key, {})[entity_type] = entity

        actions = []
        actions_by_asset = {}
        for asset_key, grouped in sorted(entities_by_asset.items()):
            asset = self.chois_actions.get(asset_key)
            if not asset:
                continue
            frame_state = frame_state_from_stamp(stamp_sec, asset)
            source_frame = int(frame_state["source_frame"])
            segment = active_segment(asset, source_frame)
            action = None if segment is None else segment["action"]
            human = grouped.get("human")
            obj = grouped.get("object")
            human_position = entity_position(human) if human else None
            object_position = entity_position(obj) if obj else None
            start_frame = source_frame if segment is None else int(segment["start_frame"])
            end_frame = source_frame if segment is None else int(segment["end_frame"])
            fps = float(frame_state["fps"])
            row = {
                "type": "hoi_action",
                "asset_key": asset_key,
                "action": action,
                "source_frame": source_frame,
                "cycle_index": int(frame_state["cycle_index"]),
                "local_frame_index": int(frame_state["local_frame_index"]),
                "source_time_sec": float(frame_state["source_time_sec"]),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_sec": float(start_frame / fps),
                "end_sec": float(end_frame / fps),
                "fps": fps,
                "subject_id": f"chois_human_{asset_key}" if human else None,
                "object_id": f"chois_object_{asset_key}" if obj else None,
                "human_position": human_position,
                "object_position": object_position,
                "prompt": str(asset.get("prompt") or ""),
                "object_label": str(asset.get("object_label") or ""),
            }
            actions.append(row)
            actions_by_asset[asset_key] = row
        self.latest_chois_actions_by_asset = actions_by_asset
        return {
            "world": self.world,
            "stamp_sec": stamp_sec,
            "frame_id": payload.get("frame_id", self.frame_id),
            "count": len(actions),
            "actions": actions,
        }

    def publish_chois_actions(self) -> None:
        if self.latest_chois_payload is None:
            return
        msg = String()
        msg.data = json.dumps(self.build_chois_action_payload(), ensure_ascii=True)
        self.chois_action_pub.publish(msg)

    def marker(self, marker_id: int, namespace: str, marker_type: int) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def set_color(marker: Marker, color: Color) -> None:
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

    @staticmethod
    def set_position(marker: Marker, xyz: list[float]) -> None:
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2])

    def bbox_marker(self, marker_id: int, namespace: str, center: list[float], extent: list[float], color: Color) -> Marker:
        marker = self.marker(marker_id, namespace, Marker.LINE_LIST)
        marker.scale.x = 0.04
        self.set_color(marker, color)
        cx, cy, cz = center
        ex, ey, ez = [max(float(value), 0.02) * 0.5 for value in extent]
        corners = [
            (cx - ex, cy - ey, cz - ez),
            (cx + ex, cy - ey, cz - ez),
            (cx + ex, cy + ey, cz - ez),
            (cx - ex, cy + ey, cz - ez),
            (cx - ex, cy - ey, cz + ez),
            (cx + ex, cy - ey, cz + ez),
            (cx + ex, cy + ey, cz + ez),
            (cx - ex, cy + ey, cz + ez),
        ]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for start, end in edges:
            for index in (start, end):
                point = Point()
                point.x, point.y, point.z = [float(value) for value in corners[index]]
                marker.points.append(point)
        return marker

    def text_marker(self, marker_id: int, namespace: str, xyz: list[float], text: str, size: float = 0.3) -> Marker:
        marker = self.marker(marker_id, namespace, Marker.TEXT_VIEW_FACING)
        self.set_position(marker, xyz)
        marker.scale.z = size
        marker.text = text
        self.set_color(marker, TEXT_COLOR)
        return marker

    def human_shape_markers(
        self,
        marker_id_base: int,
        namespace: str,
        xyz: list[float],
        yaw: float,
        color: Color,
    ) -> list[Marker]:
        x, y = float(xyz[0]), float(xyz[1])
        ground_z = max(float(xyz[2]) - 0.78, 0.02)
        foot_z = ground_z + 0.05
        hip_z = ground_z + 0.55
        chest_z = ground_z + 1.08
        shoulder_z = ground_z + 1.18
        head_z = ground_z + 1.48

        forward = (math.cos(yaw), math.sin(yaw))
        side = (-math.sin(yaw), math.cos(yaw))

        def offset(base_z: float, forward_scale: float = 0.0, side_scale: float = 0.0) -> tuple[float, float, float]:
            return (
                x + forward[0] * forward_scale + side[0] * side_scale,
                y + forward[1] * forward_scale + side[1] * side_scale,
                base_z,
            )

        head = self.marker(marker_id_base, f"{namespace}_head", Marker.SPHERE)
        self.set_position(head, [x, y, head_z])
        head.scale.x = 0.28
        head.scale.y = 0.28
        head.scale.z = 0.28
        self.set_color(head, color)

        skeleton = self.marker(marker_id_base + 1, f"{namespace}_skeleton", Marker.LINE_LIST)
        skeleton.scale.x = 0.07
        self.set_color(skeleton, color)
        segments = [
            (offset(hip_z), offset(chest_z)),
            (offset(chest_z), offset(shoulder_z, side_scale=-0.32)),
            (offset(chest_z), offset(shoulder_z, side_scale=0.32)),
            (offset(shoulder_z, side_scale=-0.32), offset(ground_z + 0.78, forward_scale=0.10, side_scale=-0.48)),
            (offset(shoulder_z, side_scale=0.32), offset(ground_z + 0.78, forward_scale=0.10, side_scale=0.48)),
            (offset(hip_z), offset(foot_z, forward_scale=0.18, side_scale=-0.22)),
            (offset(hip_z), offset(foot_z, forward_scale=0.18, side_scale=0.22)),
        ]
        for start, end in segments:
            for value in (start, end):
                point = Point()
                point.x = float(value[0])
                point.y = float(value[1])
                point.z = float(value[2])
                skeleton.points.append(point)

        heading = self.marker(marker_id_base + 2, f"{namespace}_heading", Marker.ARROW)
        self.set_position(heading, [x, y, ground_z + 0.12])
        qx, qy, qz, qw = quaternion_from_yaw(yaw)
        heading.pose.orientation.x = qx
        heading.pose.orientation.y = qy
        heading.pose.orientation.z = qz
        heading.pose.orientation.w = qw
        heading.scale.x = 0.55
        heading.scale.y = 0.10
        heading.scale.z = 0.10
        self.set_color(heading, color)
        return [head, skeleton, heading]

    def static_markers(self) -> list[Marker]:
        markers = []
        for index, obj in enumerate(self.static_objects):
            center = as_float_list(obj.get("bbox_center"))
            extent = as_float_list(obj.get("bbox_extent"))
            if center is None or extent is None:
                continue
            markers.append(self.bbox_marker(index, "gt_static_foreground_bbox", center, extent, STATIC_COLOR))
            if self.show_static_labels:
                label_z = center[2] + max(extent[2] * 0.55, 0.35)
                label = str(obj.get("semantic_label") or obj.get("name") or index)
                markers.append(self.text_marker(100000 + index, "gt_static_foreground_labels", [center[0], center[1], label_z], label, 0.22))
        return markers

    def hunav_markers(self) -> list[Marker]:
        if self.latest_hunav_time is None or self.now_sec() - self.latest_hunav_time > self.dynamic_timeout_sec:
            return []
        markers = []
        for index, agent in enumerate(self.latest_hunav_agents):
            position = agent.position.position
            agent_id = getattr(agent, "id", index)
            name = getattr(agent, "name", "") or f"agent_{agent_id}"
            yaw = float(getattr(agent, "yaw", 0.0) or 0.0)

            markers.extend(
                self.human_shape_markers(
                    200000 + index * 10,
                    "gt_hunav_person",
                    [position.x, position.y, 0.82],
                    yaw,
                    HUNAV_COLOR,
                )
            )

            markers.append(self.text_marker(220000 + index, "gt_hunav_labels", [position.x, position.y, 1.75], f"HuNav {name}", 0.3))
        return markers

    def chois_markers(self) -> list[Marker]:
        if self.latest_chois_payload is None or self.latest_chois_time is None:
            return []
        if self.now_sec() - self.latest_chois_time > self.dynamic_timeout_sec:
            return []
        markers = []
        by_asset: dict[str, dict[str, tuple[dict[str, Any], list[float]]]] = {}
        for index, entity in enumerate(self.latest_chois_payload.get("entities", [])):
            if not isinstance(entity, dict):
                continue
            position = entity_position(entity)
            if position is None:
                continue
            asset_key = str(entity.get("asset_key") or f"asset_{index}")
            entity_type = str(entity.get("type") or "object")
            marker_xyz = list(position)
            if entity_type == "human":
                marker_xyz[2] = max(marker_xyz[2], 0.75)
                yaw = self.chois_yaw_for_entity(entity, index)
                markers.extend(
                    self.human_shape_markers(
                        300000 + index * 10,
                        "gt_chois_human",
                        marker_xyz,
                        yaw,
                        CHOIS_HUMAN_COLOR,
                    )
                )
            else:
                marker = self.marker(300000 + index * 10, "gt_chois_object", Marker.CUBE)
                extent = as_float_list(entity.get("bbox_extent")) or [0.35, 0.35, 0.35]
                marker.scale.x = max(extent[0], 0.12)
                marker.scale.y = max(extent[1], 0.12)
                marker.scale.z = max(extent[2], 0.12)
                self.set_position(marker, marker_xyz)
                self.set_color(marker, CHOIS_OBJECT_COLOR)
                markers.append(marker)
            action_row = self.latest_chois_actions_by_asset.get(asset_key) or {}
            action = action_row.get("action")
            action_suffix = f" {action}" if action else ""
            markers.append(self.text_marker(310000 + index, "gt_chois_labels", [marker_xyz[0], marker_xyz[1], marker_xyz[2] + 0.9], f"CHOIS {asset_key} {entity_type}{action_suffix}", 0.28))
            by_asset.setdefault(asset_key, {})[entity_type] = (entity, marker_xyz)

        pair_index = 0
        for asset_key, pair in sorted(by_asset.items()):
            if "human" not in pair or "object" not in pair:
                continue
            human_pos = pair["human"][1]
            object_pos = pair["object"][1]
            line = self.marker(400000 + pair_index, "gt_chois_pairs", Marker.LINE_LIST)
            line.scale.x = 0.06
            self.set_color(line, CHOIS_PAIR_COLOR)
            for xyz in (human_pos, object_pos):
                point = Point()
                point.x = float(xyz[0])
                point.y = float(xyz[1])
                point.z = float(max(xyz[2], 0.35))
                line.points.append(point)
            markers.append(line)
            midpoint = [(human_pos[0] + object_pos[0]) * 0.5, (human_pos[1] + object_pos[1]) * 0.5, max((human_pos[2] + object_pos[2]) * 0.5, 0.65)]
            action_row = self.latest_chois_actions_by_asset.get(asset_key) or {}
            action = action_row.get("action")
            label = f"pair {asset_key}" + (f" {action}" if action else "")
            markers.append(self.text_marker(410000 + pair_index, "gt_chois_pair_labels", midpoint, label, 0.25))
            pair_index += 1
        return markers

    def publish_markers(self) -> None:
        markers = MarkerArray()
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        markers.markers.append(delete_all)
        markers.markers.extend(self.static_markers())
        markers.markers.extend(self.hunav_markers())
        markers.markers.extend(self.chois_markers())
        self.publisher.publish(markers)


def main() -> None:
    rclpy.init()
    node = GtRvizMarkers()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if exc.__class__.__name__ != "ExternalShutdownException":
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
