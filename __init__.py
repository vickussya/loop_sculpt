bl_info = {
    "name": "Loop Sculpt",
    "author": "vickussya",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Edit",
    "description": "Dissolve edge loops quickly with a modal wheel-controlled tool",
    "category": "Mesh",
}


def register():
    from . import loop_sculpt
    loop_sculpt.register()


def unregister():
    from . import loop_sculpt
    loop_sculpt.unregister()


__all__ = ["register", "unregister"]
