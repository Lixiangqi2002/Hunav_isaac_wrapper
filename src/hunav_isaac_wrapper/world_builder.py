#!/usr/bin/env python3
"""
world_builder.py

A simple module that loads a USD stage (map) from disk, replacing the current stage.
"""

import os
import omni
from pxr import Usd, UsdGeom

class WorldBuilder:
    """
    Loads an entire USD stage from disk.
    
    """
    def __init__(self, base_path):
        self.base_path = base_path
        self.usd_context = omni.usd.get_context()

    def _copy_stage_metadata(self, source_stage, target_stage):
        if source_stage is None or target_stage is None:
            return

        try:
            target_stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
        except Exception:
            pass
        try:
            target_stage.SetFramesPerSecond(source_stage.GetFramesPerSecond())
        except Exception:
            pass
        try:
            target_stage.SetStartTimeCode(source_stage.GetStartTimeCode())
        except Exception:
            pass
        try:
            target_stage.SetEndTimeCode(source_stage.GetEndTimeCode())
        except Exception:
            pass

        for metadata_key in (
            "customLayerData",
            "colorConfiguration",
            "colorManagementSystem",
        ):
            try:
                if source_stage.HasAuthoredMetadata(metadata_key):
                    target_stage.SetMetadata(
                        metadata_key, source_stage.GetMetadata(metadata_key)
                    )
            except Exception:
                pass

    def load_map(self, map_name: str):
        """
        Looks for `map_name.usd` inside the 'worlds' folder under base_path and opens it.
        """
        map_path = os.path.join(self.base_path, "worlds", f"{map_name}.usd")
        if not os.path.exists(map_path):
            print(f"[WorldBuilder] Error: map '{map_name}' not found at {map_path}")
            return

        source_stage = Usd.Stage.Open(map_path)
        if source_stage is None:
            print(f"[WorldBuilder] Error: failed to open map '{map_name}' at {map_path}")
            return

        source_default_prim = source_stage.GetDefaultPrim()
        source_default_path = (
            str(source_default_prim.GetPath())
            if source_default_prim and source_default_prim.IsValid()
            else ""
        )

        if source_default_path == "/World":
            self.usd_context.open_stage(map_path)
            print(f"Map '{map_name}' loaded from: {map_path}")
            return

        self.usd_context.new_stage()
        stage = self.usd_context.get_stage()
        if stage is None:
            print(f"[WorldBuilder] Error: failed to create wrapper stage for '{map_name}'")
            return

        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
        self._copy_stage_metadata(source_stage, stage)

        world_prim = stage.DefinePrim("/World", "Xform")
        stage.SetDefaultPrim(world_prim)

        wrapped_prim_path = f"/World/{map_name.capitalize()}"
        wrapped_prim = stage.DefinePrim(wrapped_prim_path, "Xform")
        if source_default_prim and source_default_prim.IsValid():
            wrapped_prim.GetReferences().AddReference(
                map_path, source_default_prim.GetPath()
            )
        else:
            wrapped_prim.GetReferences().AddReference(map_path)
        UsdGeom.Xformable(wrapped_prim)

        print(
            f"Map '{map_name}' loaded from: {map_path} and wrapped at '{wrapped_prim_path}'"
        )

    def add_usd_reference(self, usd_path: str, prim_path: str):
        """
        Add a USD asset as a referenced prim into the current stage.
        """
        stage = self.usd_context.get_stage()
        if stage is None:
            raise RuntimeError("[WorldBuilder] No active stage to add USD reference into.")
        if not os.path.exists(usd_path):
            raise FileNotFoundError(f"[WorldBuilder] USD asset not found: {usd_path}")

        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().ClearReferences()
        prim.GetReferences().AddReference(usd_path)
        UsdGeom.Xformable(prim)
        print(f"[WorldBuilder] Referenced USD '{usd_path}' at '{prim_path}'")
        return prim
