bl_info = {
    "name": "Loop Sculpt",
    "author": "vickussya",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Edit",
    "description": "Dissolve edge loops quickly with a modal wheel-controlled tool",
    "category": "Mesh",
}

import bpy
import bmesh
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)


def _active_bm(context):
    obj = context.active_object
    if not obj or obj.type != 'MESH' or context.mode != 'EDIT_MESH':
        return None, None
    return obj, bmesh.from_edit_mesh(obj.data)


def _opposite_edge_in_face(face, edge):
    v1, v2 = edge.verts
    for e in face.edges:
        if e is edge:
            continue
        if (v1 in e.verts) or (v2 in e.verts):
            continue
        return e
    return None


def _walk_loop_from_face(edge, face):
    if len(face.verts) != 4:
        return []
    loop_edges = []
    curr_edge = edge
    curr_face = face
    visited = set()
    while True:
        opp = _opposite_edge_in_face(curr_face, curr_edge)
        if not opp or opp in visited:
            break
        loop_edges.append(opp)
        visited.add(opp)
        next_face = None
        for f in opp.link_faces:
            if f is curr_face:
                continue
            if len(f.verts) == 4:
                next_face = f
                break
        if not next_face:
            break
        curr_edge, curr_face = opp, next_face
        if curr_edge is edge:
            break
    return loop_edges


def build_edge_loop(edge):
    # Manual quad loop walk, robust enough for preview.
    loop = {edge}
    for f in edge.link_faces:
        if len(f.verts) != 4:
            continue
        loop.update(_walk_loop_from_face(edge, f))
    return loop


def expand_loops(base_loop, steps):
    loops = [set(base_loop)]
    visited = set(base_loop)
    current = set(base_loop)
    for _i in range(steps):
        neighbor_seeds = set()
        for e in current:
            for f in e.link_faces:
                if len(f.verts) != 4:
                    continue
                opp = _opposite_edge_in_face(f, e)
                if opp and opp not in visited:
                    neighbor_seeds.add(opp)
        if not neighbor_seeds:
            break
        neighbor_loop = set()
        for seed in neighbor_seeds:
            if seed in visited:
                continue
            neighbor_loop.update(build_edge_loop(seed))
        if not neighbor_loop:
            break
        loops.append(neighbor_loop)
        visited.update(neighbor_loop)
        current = neighbor_loop
    return loops


def connected_edge_region(start_edge):
    region = set()
    stack = [start_edge]
    while stack:
        e = stack.pop()
        if e in region:
            continue
        region.add(e)
        for v in e.verts:
            for linked in v.link_edges:
                if linked not in region:
                    stack.append(linked)
    return region


def apply_filters(context, bm, edges, settings, start_edge):
    obj = context.active_object
    if not edges:
        return set()

    filtered = set(edges)

    if settings.limit_region:
        region = connected_edge_region(start_edge)
        filtered = {e for e in filtered if e in region}

    if settings.vg_name:
        group = obj.vertex_groups.get(settings.vg_name)
        if not group:
            return set()
        dlayer = bm.verts.layers.deform.verify()
        thresh = settings.vg_threshold
        def has_weight(v):
            weights = v[dlayer]
            return weights.get(group.index, 0.0) >= thresh
        filtered = {e for e in filtered if has_weight(e.verts[0]) and has_weight(e.verts[1])}

    if settings.material_filter != "NONE":
        try:
            mat_index = int(settings.material_filter)
        except ValueError:
            mat_index = -1
        if mat_index >= 0:
            def has_mat(e):
                for f in e.link_faces:
                    if f.material_index == mat_index:
                        return True
                return False
            filtered = {e for e in filtered if has_mat(e)}

    return filtered


def _deselect_all(bm):
    for e in bm.edges:
        e.select = False
    for v in bm.verts:
        v.select = False
    for f in bm.faces:
        f.select = False


def _restore_selection(bm, sel):
    _deselect_all(bm)
    for e in sel['edges']:
        if e.is_valid:
            e.select = True
    for v in sel['verts']:
        if v.is_valid:
            v.select = True
    for f in sel['faces']:
        if f.is_valid:
            f.select = True


def _status(context, text):
    if context.workspace:
        context.workspace.status_text_set(text)


def _clear_status(context):
    if context.workspace:
        context.workspace.status_text_set(None)


class LoopSculptSettings(PropertyGroup):
    step: IntProperty(
        name="Step",
        description="Dissolve every Nth loop",
        default=2,
        min=1,
        max=100,
    )
    include_start: BoolProperty(
        name="Include Starting Loop",
        description="Allow dissolving the starting loop",
        default=False,
    )
    limit_region: BoolProperty(
        name="Limit to Connected Region",
        default=True,
    )
    vg_name: StringProperty(
        name="Vertex Group",
        description="Only dissolve edges where both vertices are in this group",
        default="HAIR",
    )
    vg_threshold: FloatProperty(
        name="Weight Threshold",
        default=0.1,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )

    def _material_items(self, context):
        items = [("NONE", "None", "No material filter")]
        obj = context.active_object
        if obj and obj.type == 'MESH':
            for idx, slot in enumerate(obj.material_slots):
                name = slot.material.name if slot.material else f"Slot {idx}"
                items.append((str(idx), name, ""))
        return items

    material_filter: EnumProperty(
        name="Material",
        description="Only dissolve edges connected to faces with this material",
        items=_material_items,
    )


class MESH_OT_loop_sculpt(Operator):
    bl_idname = "mesh.loop_sculpt"
    bl_label = "Loop Sculpt"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    extend: IntProperty(default=0, min=0)

    def invoke(self, context, event):
        obj, bm = _active_bm(context)
        if not bm:
            self.report({'WARNING'}, "Active mesh edit mode required")
            return {'CANCELLED'}

        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        start_edge = None
        if bm.select_history and isinstance(bm.select_history[-1], bmesh.types.BMEdge):
            start_edge = bm.select_history[-1]
        if not start_edge:
            for e in bm.edges:
                if e.select:
                    start_edge = e
                    break
        if not start_edge:
            self.report({'WARNING'}, "Select an edge on a loop")
            return {'CANCELLED'}

        base_loop = build_edge_loop(start_edge)
        if not base_loop or len(base_loop) < 2:
            self.report({'WARNING'}, "No valid edge loop found")
            return {'CANCELLED'}

        settings = context.scene.loop_sculpt_settings

        self._start_edge = start_edge
        self._base_loop = base_loop
        self._orig_sel = {
            'edges': {e for e in bm.edges if e.select},
            'verts': {v for v in bm.verts if v.select},
            'faces': {f for f in bm.faces if f.select},
        }
        self._settings_snapshot = {
            'step': settings.step,
            'include_start': settings.include_start,
            'limit_region': settings.limit_region,
            'vg_name': settings.vg_name,
            'vg_threshold': settings.vg_threshold,
            'material_filter': settings.material_filter,
        }

        self.extend = 0
        self._update_preview(context, bm)
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _status_text(self):
        s = self._settings_snapshot
        return f"Loop Sculpt | Extend: {self.extend} | Step: {s['step']}"

    def _candidate_loops(self, bm):
        loops = expand_loops(self._base_loop, self.extend)
        return loops

    def _edges_to_dissolve(self, context, bm):
        loops = self._candidate_loops(bm)
        s = self._settings_snapshot

        candidates = []
        if s['include_start']:
            candidates = loops
        else:
            candidates = loops[1:] if len(loops) > 1 else []

        edges = set()
        if candidates:
            for idx, loop in enumerate(candidates):
                if idx % max(1, s['step']) == 0:
                    edges.update(loop)

        class _SettingsView:
            pass
        sv = _SettingsView()
        sv.step = s['step']
        sv.include_start = s['include_start']
        sv.limit_region = s['limit_region']
        sv.vg_name = s['vg_name']
        sv.vg_threshold = s['vg_threshold']
        sv.material_filter = s['material_filter']

        return apply_filters(context, bm, edges, sv, self._start_edge)

    def _update_preview(self, context, bm):
        edges = self._edges_to_dissolve(context, bm)
        _deselect_all(bm)
        for e in edges:
            if e.is_valid:
                e.select = True
        bmesh.update_edit_mesh(context.active_object.data, loop_triangles=False, destructive=False)

    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS'}:
            self.extend += 1
            obj, bm = _active_bm(context)
            if not bm:
                return {'CANCELLED'}
            self._update_preview(context, bm)
            _status(context, self._status_text())
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELDOWNMOUSE', 'NUMPAD_MINUS'}:
            self.extend = max(0, self.extend - 1)
            obj, bm = _active_bm(context)
            if not bm:
                return {'CANCELLED'}
            self._update_preview(context, bm)
            _status(context, self._status_text())
            return {'RUNNING_MODAL'}

        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            obj, bm = _active_bm(context)
            if not bm:
                return {'CANCELLED'}
            edges = self._edges_to_dissolve(context, bm)
            if not edges:
                self.report({'WARNING'}, "No edges matched filters")
                _restore_selection(bm, self._orig_sel)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                _clear_status(context)
                return {'CANCELLED'}
            bmesh.ops.dissolve_edges(bm, edges=list(edges), use_verts=True)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)
            _clear_status(context)
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            obj, bm = _active_bm(context)
            if bm:
                _restore_selection(bm, self._orig_sel)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            _clear_status(context)
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


class VIEW3D_PT_loop_sculpt(Panel):
    bl_label = "Retopo Cleanup"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Edit"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.loop_sculpt_settings

        layout.operator(MESH_OT_loop_sculpt.bl_idname, text="Loop Sculpt (Ctrl+X)")
        layout.prop(settings, "step")
        layout.prop(settings, "include_start")
        layout.prop(settings, "limit_region")

        col = layout.column(align=True)
        col.label(text="Hair Filters")
        obj = context.active_object
        if obj:
            col.prop_search(settings, "vg_name", obj, "vertex_groups", text="Vertex Group")
        else:
            col.prop(settings, "vg_name")
        col.prop(settings, "vg_threshold")
        col.prop(settings, "material_filter")


addon_keymaps = []


def register_keymap():
    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        km = wm.keyconfigs.addon.keymaps.new(name='Mesh', space_type='EMPTY')
        kmi = km.keymap_items.new(MESH_OT_loop_sculpt.bl_idname, type='X', value='PRESS', ctrl=True)
        addon_keymaps.append((km, kmi))


def unregister_keymap():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()


def register():
    bpy.utils.register_class(LoopSculptSettings)
    bpy.utils.register_class(MESH_OT_loop_sculpt)
    bpy.utils.register_class(VIEW3D_PT_loop_sculpt)
    bpy.types.Scene.loop_sculpt_settings = PointerProperty(type=LoopSculptSettings)
    register_keymap()


def unregister():
    unregister_keymap()
    if hasattr(bpy.types.Scene, "loop_sculpt_settings"):
        del bpy.types.Scene.loop_sculpt_settings
    bpy.utils.unregister_class(VIEW3D_PT_loop_sculpt)
    bpy.utils.unregister_class(MESH_OT_loop_sculpt)
    bpy.utils.unregister_class(LoopSculptSettings)


if __name__ == "__main__":
    register()
