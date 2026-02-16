bl_info = {
    "name": "Loop Sculpt",
    "author": "vickussya",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Edit",
    "description": "Dissolve edge loops quickly with a modal wheel-controlled tool",
    "category": "Mesh",
}

import importlib.util
import pathlib
import sys

_MODULE_NAME = __name__ + "_core"
_MODULE_PATH = pathlib.Path(__file__).with_name("loop_sculpt.py")

_spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MODULE_NAME] = _mod
_spec.loader.exec_module(_mod)

register = _mod.register
unregister = _mod.unregister

__all__ = ["register", "unregister"]
