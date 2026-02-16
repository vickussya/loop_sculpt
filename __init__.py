bl_info = {
    "name": "Loop Sculpt",
    "author": "vickussya",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar",
    "description": "Loop selection and dissolve helper",
    "category": "Mesh",
}

from . import loop_sculpt

register = loop_sculpt.register()
unregister = loop_sculpt.unregister()
