bl_info = {
    "name": "Loop Sculpt",
    "author": "vickussya",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar",
    "description": "Loop selection and dissolve helper",
    "category": "Mesh",
}

def register():
    from . import loop_sculpt
    loop_sculpt.register()

def unregister():
    from . import loop_sculpt
    loop_sculpt.unregister()
