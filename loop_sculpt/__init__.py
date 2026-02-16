bl_info = {
    "name": "Loop Sculpt",
    "author": "Codex",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Edit",
    "description": "Dissolve edge loops quickly with a modal wheel-controlled tool",
    "category": "Mesh",
}

from .loop_sculpt import register, unregister

__all__ = ["register", "unregister"]
