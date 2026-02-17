import bpy
import bmesh
import math
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import BoolProperty, IntProperty, PointerProperty


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


def _edge_key(edge):
    v1, v2 = edge.verts
    a = v1.index
    b = v2.index
    return (a, b) if a < b else (b, a)


def _edge_from_key(bm, key):
    a, b = key
    if a >= len(bm.verts) or b >= len(bm.verts):
        return None
    v1 = bm.verts[a]
    v2 = bm.verts[b]
    for e in v1.link_edges:
        if v2 in e.verts:
            return e
    return None


def _edges_from_keys(bm, keys):
    edges = set()
    for key in keys:
        e = _edge_from_key(bm, key)
        if not e:
            return None
        edges.add(e)
    return edges


def _loop_keys(edges):
    return {_edge_key(e) for e in edges}


def _opposite_edge_in_face(face, edge):
    v1, v2 = edge.verts
    for e in face.edges:
        if e is edge:
            continue
        if (v1 in e.verts) or (v2 in e.verts):
            continue
        return e
    return None


def _representative_edge(loop_edges):
    return min(loop_edges, key=lambda e: e.index)


def _is_boundary_edge(edge):
    return len(edge.link_faces) != 2


def _edge_dihedral_deg(edge):
    if len(edge.link_faces) != 2:
        return 180.0
    f1, f2 = edge.link_faces
    f1.normal_update()
    f2.normal_update()
    dot = f1.normal.dot(f2.normal)
    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return math.degrees(math.acos(dot))


def _is_protected_edge(edge, protect_angle_deg):
    if _is_boundary_edge(edge):
        return True
    if getattr(edge, "use_edge_sharp", False):
        return True
    if getattr(edge, "seam", False):
        return True
    if _edge_dihedral_deg(edge) >= protect_angle_deg:
        return True
    return False


def _loop_is_protected(loop_edges, protect_angle_deg):
    return any(_is_protected_edge(e, protect_angle_deg) for e in loop_edges)


def _sample_edges(loop_edges, max_items=3):
    result = []
    for e in loop_edges:
        result.append(e.index)
        if len(result) >= max_items:
            break
    return result


def _sample_protected(loop_edges, protect_angle_deg, max_items=5):
    edges = []
    angles = []
    for e in loop_edges:
        if _is_protected_edge(e, protect_angle_deg):
            edges.append(e.index)
            angles.append(_edge_dihedral_deg(e))
            if len(edges) >= max_items:
                break
    return edges, angles


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


def _settings_from_context(context):
    return getattr(context.scene, "loop_sculpt_settings", None)


def _validate_loop_edges(edges):
    if not edges or len(edges) < 2:
        return False, "selection too small"
    edge_set = set(edges)
    neighbors = {}
    for e in edge_set:
        linked = set()
        for v in e.verts:
            for le in v.link_edges:
                if le in edge_set and le is not e:
                    linked.add(le)
        neighbors[e] = linked
    counts = [len(neighbors[e]) for e in edge_set]
    if any(c > 2 for c in counts):
        return False, "branching selection"
    ones = sum(1 for c in counts if c == 1)
    twos = sum(1 for c in counts if c == 2)
    if not (ones == 0 or ones == 2):
        return False, "not a loop chain"
    if ones == 2 and twos + ones != len(edge_set):
        return False, "not a single chain"
    start = next(iter(edge_set))
    visited = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for n in neighbors[cur]:
            if n not in visited:
                visited.add(n)
                stack.append(n)
    if len(visited) != len(edge_set):
        return False, "selection not connected"
    return True, ""


def _get_loop_from_seed_edge(bm, obj, area, region, window, seed_edge):
    original = {e for e in bm.edges if e.select}
    _deselect_all(bm)
    seed_edge.select = True
    bm.select_history.clear()
    bm.select_history.add(seed_edge)
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

    with bpy.context.temp_override(
        window=window,
        area=area,
        region=region,
        active_object=obj,
        object=obj,
    ):
        bpy.ops.mesh.loop_multi_select(ring=False)

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    loop = {e for e in bm.edges if e.select}

    _deselect_all(bm)
    for e in original:
        if e.is_valid:
            e.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return loop


def _adjacent_loop_on_side(bm, obj, area, region, window, loop_edges, seed_face_index, prev_loop):
    seed_face = None
    for f in bm.faces:
        if f.index == seed_face_index:
            seed_face = f
            break
    if not seed_face:
        return None, None, None

    e0 = None
    for e in loop_edges:
        for f in e.link_faces:
            if f.index == seed_face_index:
                e0 = e
                break
        if e0:
            break
    if not e0:
        e0 = _representative_edge(loop_edges)

    if len(seed_face.verts) != 4:
        return None, None, None
    opp = _opposite_edge_in_face(seed_face, e0)
    if not opp:
        return None, None, None

    loop = _get_loop_from_seed_edge(bm, obj, area, region, window, opp)
    if not loop:
        return None, None, None

    if prev_loop and _loop_keys(loop) == _loop_keys(prev_loop):
        return None, None, None

    return loop, opp, (e0.verts[0].index, e0.verts[1].index)


class LoopSculptSettings(PropertyGroup):
    skip_loops: IntProperty(
        name="Skip Loops",
        description="Number of loops to skip between selected loops",
        default=1,
        min=1,
        max=5,
    )
    protect_angle_deg: IntProperty(
        name="Protect Angle",
        description="Edges with face angle >= this value are treated as silhouette and will not be selected",
        default=45,
        min=1,
        max=89,
        subtype='ANGLE',
    )
    disable_protection: BoolProperty(
        name="Disable Protection (Debug)",
        description="Ignore silhouette protection for debugging",
        default=False,
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

        self._win = context.window
        self._area = context.area
        self._region = context.region

        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        selected_edges = [e for e in bm.edges if e.select]
        _debug_log(
            "invoke: mode=%s area=%s region=%s selected_edges=%d" %
            (context.mode, self._area.type if self._area else None, self._region.type if self._region else None, len(selected_edges))
        )
        if len(selected_edges) < 2:
            self.report({'ERROR'}, "Select a full edge loop (at least 2 edges)")
            _debug_log("invoke: cancelled (selection too small)")
            return {'CANCELLED'}

        ok, reason = _validate_loop_edges(selected_edges)
        if not ok:
            self.report({'ERROR'}, "Selection must be a single edge loop")
            _debug_log("invoke: cancelled (selection invalid: %s)" % reason)
            return {'CANCELLED'}

        self._base_loop_keys = _loop_keys(selected_edges)
        self._orig_sel = {
            'edges': {e for e in bm.edges if e.select},
            'verts': {v for v in bm.verts if v.select},
            'faces': {f for f in bm.faces if f.select},
        }

        settings = _settings_from_context(context)
        self._skip_loops = settings.skip_loops if settings else 1
        self._protect_angle_deg = settings.protect_angle_deg if settings else 45
        self._disable_protection = settings.disable_protection if settings else False

        e0 = _representative_edge(selected_edges)
        faces = [f for f in e0.link_faces if len(f.verts) == 4]
        if len(faces) < 2:
            self.report({'WARNING'}, "Only one quad side found; expansion may be one-sided")
        self._sideA_face_index = faces[0].index if len(faces) > 0 else -1
        self._sideB_face_index = faces[1].index if len(faces) > 1 else -1

        self.extend = 0
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        _debug_log("invoke: base_loop_edges=%d" % len(self._base_loop_keys))
        return {'RUNNING_MODAL'}

    def _status_text(self):
        return f"Loop Sculpt | Extend: {self.extend} | Skip: {self._skip_loops}"

    def _edges_for_step(self, bm, obj, step, event_name):
        base_loop = _edges_from_keys(bm, self._base_loop_keys)
        if not base_loop:
            return None, "Base loop not found"

        hop = self._skip_loops + 1

        selected_loops = [base_loop]
        blocked = [False, False]
        added = [False, False]

        _debug_log("event: %s" % event_name)
        _debug_log("skip_loops=%d, hop=%d, protect_angle=%d, protection_disabled=%s" % (
            self._skip_loops,
            hop,
            self._protect_angle_deg,
            self._disable_protection,
        ))
        _debug_log("base_loop: edges=%d" % len(base_loop))

        side_faces = [self._sideA_face_index, self._sideB_face_index]
        side_counts = [0, 0]
        fail_reason = None

        for side_index in range(2):
            if side_faces[side_index] < 0:
                blocked[side_index] = True
                continue

            curr_loop = base_loop
            prev_loop = None
            final_loop = None
            seed_edge_key = None

            for _step in range(step):
                for _hop in range(hop):
                    nxt_loop, opp, seed_edge_key = _adjacent_loop_on_side(
                        bm,
                        obj,
                        self._area,
                        self._region,
                        self._win,
                        curr_loop,
                        side_faces[side_index],
                        prev_loop,
                    )
                    if not nxt_loop:
                        blocked[side_index] = True
                        fail_reason = "No adjacent loop found (topology/strip detection failed)"
                        break
                    prev_loop = curr_loop
                    curr_loop = nxt_loop
                    final_loop = curr_loop
                if blocked[side_index]:
                    break

            if blocked[side_index] or not final_loop:
                blocked[side_index] = True
                final_loop = None

            protected = False
            sample_edges = []
            sample_angles = []
            if final_loop and not self._disable_protection:
                protected = _loop_is_protected(final_loop, self._protect_angle_deg)
                sample_edges, sample_angles = _sample_protected(final_loop, self._protect_angle_deg)

            _debug_log("side%s_attempt:" % ("A" if side_index == 0 else "B"))
            _debug_log("    final_loop_edges=%d, protected=%s" % (
                len(final_loop) if final_loop else 0,
                protected,
            ))
            _debug_log("    sample_protected_edges=%s" % sample_edges)
            _debug_log("    sample_angles_deg=%s" % [round(a, 2) for a in sample_angles])
            _debug_log("    seed_face_index=%d, hop=%d, chosen_seed_edge=%s" % (
                side_faces[side_index],
                hop,
                seed_edge_key,
            ))

            if final_loop and (self._disable_protection or not protected):
                selected_loops.append(final_loop)
                added[side_index] = True
                side_counts[side_index] += 1
            else:
                if not self._disable_protection and protected:
                    fail_reason = "Reached protected silhouette loop"
                blocked[side_index] = True

        edges = set()
        for loop in selected_loops:
            edges.update(loop)

        _debug_log("result:")
        _debug_log("    added_sideA=%s, added_sideB=%s" % (added[0], added[1]))
        _debug_log("    selected_loops_sideA_count=%d" % side_counts[0])
        _debug_log("    selected_loops_sideB_count=%d" % side_counts[1])

        self._last_blocked = blocked
        self._last_added = added

        if blocked[0] and blocked[1] and step > 0:
            return None, fail_reason or "No adjacent loop found (topology/strip detection failed)"
        return edges, None

    def _update_preview(self, context, bm, obj, event_name):
        edges, err = self._edges_for_step(bm, obj, self.extend, event_name)
        if edges is None:
            return False, err
        if not edges:
            return True, None
        _deselect_all(bm)
        for e in edges:
            if e.is_valid:
                e.select = True
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return True, None

    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'LEFTMOUSE', 'ESC', 'RIGHTMOUSE'}:
            _debug_log("EVENT: %s %s" % (event.type, event.value))

        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS'}:
            self.extend += 1
            obj, bm = _active_bm(context)
            if not bm:
                self.report({'INFO'}, "Hover viewport + stay in Edit Mode")
                return {'RUNNING_MODAL'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            ok, err = self._update_preview(context, bm, obj, "WHEELUP")
            if not ok:
                if self._disable_protection:
                    self.report({'WARNING'}, err or "No adjacent loop found")
                else:
                    if err == "Reached protected silhouette loop":
                        self.report({'WARNING'}, "Reached protected silhouette loop; stopping.")
                    else:
                        self.report({'WARNING'}, err or "No adjacent loop found")
                self.extend = max(0, self.extend - 1)
                return {'RUNNING_MODAL'}
            if self._last_blocked[0] != self._last_blocked[1]:
                self.report({'INFO'}, "Side A blocked, side B extended.")
            _status(context, self._status_text())
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELDOWNMOUSE', 'NUMPAD_MINUS'}:
            self.extend = max(0, self.extend - 1)
            obj, bm = _active_bm(context)
            if not bm:
                self.report({'INFO'}, "Hover viewport + stay in Edit Mode")
                return {'RUNNING_MODAL'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            ok, err = self._update_preview(context, bm, obj, "WHEELDOWN")
            if not ok:
                if self._disable_protection:
                    self.report({'WARNING'}, err or "No adjacent loop found")
                else:
                    if err == "Reached protected silhouette loop":
                        self.report({'WARNING'}, "Reached protected silhouette loop; stopping.")
                    else:
                        self.report({'WARNING'}, err or "No adjacent loop found")
                return {'RUNNING_MODAL'}
            _status(context, self._status_text())
            return {'RUNNING_MODAL'}

        if event.type in {'LEFTMOUSE', 'RET', 'NUMPAD_ENTER'}:
            _clear_status(context)
            self.report({'INFO'}, "Loop Sculpt finished; selection kept")
            _debug_log("finish: leftmouse")
            return {'FINISHED'}

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            obj, bm = _active_bm(context)
            if bm:
                base_loop = _edges_from_keys(bm, self._base_loop_keys)
                if base_loop:
                    _deselect_all(bm)
                    for e in base_loop:
                        e.select = True
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
                else:
                    _restore_selection(bm, self._orig_sel)
                    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
            _clear_status(context)
            self.report({'INFO'}, "Loop Sculpt cancelled; base loop restored")
            _debug_log("cancel: %s" % event.type)
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
        layout.operator(MESH_OT_loop_sculpt.bl_idname, text="Loop Sculpt")
        if settings:
            layout.prop(settings, "skip_loops")
            layout.prop(settings, "protect_angle_deg")
            layout.prop(settings, "disable_protection")


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
