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

def _debug_text():
    text = bpy.data.texts.get("LoopSculpt_Debug")
    if text is None:
        text = bpy.data.texts.new("LoopSculpt_Debug")
    return text


def _debug_log(message):
    text = _debug_text()
    text.write(message + "\n")

def _edge_by_index(bm, index):
    try:
        return bm.edges[index]
    except (IndexError, TypeError):
        return None


def _edges_from_indices(bm, indices):
    edges = set()
    for idx in indices:
        e = _edge_by_index(bm, idx)
        if not e:
            return None
        edges.add(e)
    return edges


def _loop_indices(edges):
    return [e.index for e in edges]


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


def _is_deferred(value):
    return value is None or type(value).__name__ == "_PropertyDeferred"


def _settings_from_context(context):
    settings = getattr(context.scene, "loop_sculpt_settings", None)
    if _is_deferred(settings):
        return None
    return settings


def _prop_value(value, default):
    if _is_deferred(value):
        return default
    return value


def _snapshot_settings(settings):
    return {
        'step': int(_prop_value(settings.step, 2)),
        'include_start': bool(_prop_value(settings.include_start, False)),
        'limit_region': bool(_prop_value(settings.limit_region, True)),
        'vg_name': str(_prop_value(settings.vg_name, "")),
        'vg_threshold': float(_prop_value(settings.vg_threshold, 0.1)),
        'material_filter': str(_prop_value(settings.material_filter, "NONE")),
    }


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
            _debug_log("invoke: cancelled (no edit mesh)")
            return {'CANCELLED'}

        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        selected_edges = [e for e in bm.edges if e.select]
        _debug_log(
            "invoke: mode=%s obj=%s selected_edges=%d" %
            (context.mode, obj.name if obj else "None", len(selected_edges))
        )
        if len(selected_edges) < 2:
            self.report({'ERROR'}, "Select an entire edge loop (at least 2 edges)")
            _debug_log("invoke: cancelled (selection too small)")
            return {'CANCELLED'}

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
            _debug_log("invoke: cancelled (no start edge)")
            return {'CANCELLED'}

        base_loop = build_edge_loop(start_edge)
        if not base_loop or len(base_loop) < 2:
            self.report({'WARNING'}, "No valid edge loop found")
            _debug_log("invoke: cancelled (no valid loop)")
            return {'CANCELLED'}

        settings = _settings_from_context(context)
        if not settings:
            self.report({'ERROR'}, "Loop Sculpt settings missing. Reinstall the add-on.")
            _debug_log("invoke: cancelled (settings missing)")
            return {'CANCELLED'}
        if settings.vg_name and obj and not obj.vertex_groups.get(settings.vg_name):
            self.report({'WARNING'}, f"Vertex group '{settings.vg_name}' not found")
            _debug_log("invoke: warning (vertex group missing)")

        self._start_edge_index = start_edge.index
        self._base_loop_indices = _loop_indices(base_loop)
        self._orig_sel = {
            'edges': {e for e in bm.edges if e.select},
            'verts': {v for v in bm.verts if v.select},
            'faces': {f for f in bm.faces if f.select},
        }
        self._settings_snapshot = _snapshot_settings(settings)

        self.extend = 0
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        _debug_log("invoke: running (base_loop_edges=%d)" % len(self._base_loop_indices))
        return {'RUNNING_MODAL'}

    def _status_text(self):
        s = self._settings_snapshot
        return f"Loop Sculpt | Extend: {self.extend} | Step: {s.get('step', 1)}"

    def _candidate_loops(self, bm, max_offset):
        base_loop = _edges_from_indices(bm, self._base_loop_indices)
        if not base_loop:
            return None
        loops = expand_loops(base_loop, max_offset)
        return loops

    def _edges_to_dissolve(self, context, bm):
        s = self._settings_snapshot
        max_offset = max(0, self.extend * 2)
        loops = self._candidate_loops(bm, max_offset)
        if loops is None:
            return None
        if not loops:
            return set()
        # Always start from the selected loop and alternate outward:
        # selected, unselected, selected, unselected...

        obj = context.active_object
        if s.get('vg_name', "") and obj and not obj.vertex_groups.get(s.get('vg_name', "")):
            self.report({'WARNING'}, f"Vertex group '{s.get('vg_name', '')}' not found; filter disabled")
            s = dict(s)
            s['vg_name'] = ""
            _debug_log("filter: disabled missing vertex group")

        edges = set()
        for idx, loop in enumerate(loops):
            if idx % 2 == 0:
                edges.update(loop)

        class _SettingsView:
            pass
        sv = _SettingsView()
        sv.step = 2
        sv.include_start = True
        sv.limit_region = bool(s.get('limit_region', True))
        sv.vg_name = s.get('vg_name', "")
        sv.vg_threshold = float(s.get('vg_threshold', 0.0))
        sv.material_filter = s.get('material_filter', "NONE")

        start_edge = _edge_by_index(bm, self._start_edge_index)
        if not start_edge:
            return None
        return apply_filters(context, bm, edges, sv, start_edge)

    def _update_preview(self, context, bm):
        edges = self._edges_to_dissolve(context, bm)
        if edges is None:
            return False
        if not edges:
            return True
        _deselect_all(bm)
        for e in edges:
            if e.is_valid:
                e.select = True
        bmesh.update_edit_mesh(context.active_object.data, loop_triangles=False, destructive=False)
        return True

    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS'}:
            self.extend += 1
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("wheel up: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            if not self._update_preview(context, bm):
                self.report({'WARNING'}, "Edge loop data changed; restart the tool")
                _debug_log("wheel up: cancelled (edge data changed)")
                _clear_status(context)
                return {'CANCELLED'}
            _status(context, self._status_text())
            _debug_log("wheel up: extend=%d" % self.extend)
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELDOWNMOUSE', 'NUMPAD_MINUS'}:
            self.extend = max(0, self.extend - 1)
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("wheel down: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            if not self._update_preview(context, bm):
                self.report({'WARNING'}, "Edge loop data changed; restart the tool")
                _debug_log("wheel down: cancelled (edge data changed)")
                _clear_status(context)
                return {'CANCELLED'}
            _status(context, self._status_text())
            _debug_log("wheel down: extend=%d" % self.extend)
            return {'RUNNING_MODAL'}

        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("finish: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            edges = self._edges_to_dissolve(context, bm)
            if edges is None:
                self.report({'WARNING'}, "Edge loop data changed; restart the tool")
                _debug_log("finish: cancelled (edge data changed)")
                _clear_status(context)
                return {'CANCELLED'}
            if not edges:
                self.report({'WARNING'}, "No edges matched filters")
                _restore_selection(bm, self._orig_sel)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                _clear_status(context)
                _debug_log("finish: cancelled (no edges matched)")
                return {'CANCELLED'}
            bmesh.ops.dissolve_edges(bm, edges=list(edges), use_verts=True)
            bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=True)
            _clear_status(context)
            _debug_log("finish: applied")
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            obj, bm = _active_bm(context)
            if bm:
                _restore_selection(bm, self._orig_sel)
                bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            _clear_status(context)
            self.report({'INFO'}, "Loop Sculpt cancelled; selection restored")
            _debug_log("cancel: restored selection")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


class VIEW3D_PT_loop_sculpt(Panel):
    bl_label = "Loop Sculpt"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Loop Sculpt"
    bl_context = "mesh_edit"

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj and obj.type == 'MESH' and context.mode == 'EDIT_MESH'

    def draw(self, context):
        layout = self.layout
        settings = _settings_from_context(context)
        if not settings:
            layout.label(text="Settings missing; reinstall the add-on.")
            return

        layout.operator(MESH_OT_loop_sculpt.bl_idname, text="Loop Sculpt")
        layout.prop(settings, "step")
        layout.prop(settings, "include_start")
        layout.prop(settings, "limit_region")

        col = layout.column(align=True)
        col.label(text="Hair Filters")
        obj = context.active_object
        if settings.vg_name and obj and not obj.vertex_groups.get(settings.vg_name):
            col.label(text=f"Vertex group '{settings.vg_name}' not found", icon='ERROR')
        if obj:
            col.prop_search(settings, "vg_name", obj, "vertex_groups", text="Vertex Group")
        else:
            col.prop(settings, "vg_name")
        col.prop(settings, "vg_threshold")
        col.prop(settings, "material_filter")


def register():
    bpy.utils.register_class(LoopSculptSettings)
    bpy.utils.register_class(MESH_OT_loop_sculpt)
    bpy.utils.register_class(VIEW3D_PT_loop_sculpt)
    bpy.types.Scene.loop_sculpt_settings = PointerProperty(type=LoopSculptSettings)


def unregister():
    if hasattr(bpy.types.Scene, "loop_sculpt_settings"):
        del bpy.types.Scene.loop_sculpt_settings
    bpy.utils.unregister_class(VIEW3D_PT_loop_sculpt)
    bpy.utils.unregister_class(MESH_OT_loop_sculpt)
    bpy.utils.unregister_class(LoopSculptSettings)


if __name__ == "__main__":
    register()
