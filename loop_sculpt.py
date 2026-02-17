import bpy
import bmesh
import math
import mathutils
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


def _edge_by_key(bm):
    edge_map = {}
    for e in bm.edges:
        v1, v2 = e.verts
        a = v1.index
        b = v2.index
        key = (a, b) if a < b else (b, a)
        edge_map[key] = e
    return edge_map


def _edges_from_keys(edge_map, keys):
    edges = []
    for key in keys:
        e = edge_map.get(key)
        if e:
            edges.append(e)
    return edges


def _loop_keys(edges):
    return {_edge_key(e) for e in edges}


def _loop_centroid_from_keys(edge_map, loop_keys):
    total = None
    count = 0
    for key in loop_keys:
        e = edge_map.get(key)
        if not e:
            continue
        v1, v2 = e.verts
        mid = (v1.co + v2.co) * 0.5
        if total is None:
            total = mid.copy()
        else:
            total += mid
        count += 1
    if total is None or count == 0:
        return None
    return total / count


def _principal_axis(centroids):
    if not centroids:
        return mathutils.Vector((1.0, 0.0, 0.0))
    mean = mathutils.Vector((0.0, 0.0, 0.0))
    for c in centroids:
        mean += c
    mean /= len(centroids)
    cov = mathutils.Matrix(((0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0),
                            (0.0, 0.0, 0.0)))
    for c in centroids:
        d = c - mean
        cov[0][0] += d.x * d.x
        cov[0][1] += d.x * d.y
        cov[0][2] += d.x * d.z
        cov[1][0] += d.y * d.x
        cov[1][1] += d.y * d.y
        cov[1][2] += d.y * d.z
        cov[2][0] += d.z * d.x
        cov[2][1] += d.z * d.y
        cov[2][2] += d.z * d.z
    axis = mathutils.Vector((1.0, 0.0, 0.0))
    for _ in range(8):
        axis = cov @ axis
        if axis.length == 0.0:
            return mathutils.Vector((1.0, 0.0, 0.0))
        axis.normalize()
    return axis


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


def _loop_is_protected(loop_keys, edge_map, protect_angle_deg):
    for key in loop_keys:
        e = edge_map.get(key)
        if e and _is_protected_edge(e, protect_angle_deg):
            return True
    return False


def _sample_protected(loop_keys, edge_map, protect_angle_deg, max_items=5):
    edges = []
    angles = []
    for key in loop_keys:
        e = edge_map.get(key)
        if not e:
            continue
        if _is_protected_edge(e, protect_angle_deg):
            edges.append(e.index)
            angles.append(_edge_dihedral_deg(e))
            if len(edges) >= max_items:
                break
    return edges, angles


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


def _get_ring_edges_from_seed(bm, obj, area, region, window, seed_edge):
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
        bpy.ops.mesh.loop_multi_select(ring=True)

    bm = bmesh.from_edit_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    ring_edges = {e for e in bm.edges if e.select}

    _deselect_all(bm)
    for e in original:
        if e.is_valid:
            e.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return ring_edges


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

        self._base_loop_keys = list(_loop_keys(selected_edges))
        self._orig_sel = {
            'edges': {e for e in bm.edges if e.select},
            'verts': {v for v in bm.verts if v.select},
            'faces': {f for f in bm.faces if f.select},
        }

        settings = _settings_from_context(context)
        self._skip_loops = settings.skip_loops if settings else 1
        self._protect_angle_deg = settings.protect_angle_deg if settings else 45
        self._disable_protection = settings.disable_protection if settings else False

        base_rep = min(selected_edges, key=lambda e: e.index) if selected_edges else None
        if not base_rep:
            self.report({'ERROR'}, "No representative edge found")
            return {'CANCELLED'}

        ring_edges = _get_ring_edges_from_seed(bm, obj, self._area, self._region, self._win, base_rep)
        if not ring_edges:
            self.report({'ERROR'}, "Edge ring not found for base loop")
            return {'CANCELLED'}

        # Build loops from ring edges.
        unassigned = set(ring_edges)
        loops = []
        while unassigned:
            seed = next(iter(unassigned))
            loop = _get_loop_from_seed_edge(bm, obj, self._area, self._region, self._win, seed)
            if not loop:
                break
            loops.append(list(_loop_keys(loop)))
            unassigned -= loop

        if not loops:
            self.report({'ERROR'}, "Could not extract loops from ring")
            return {'CANCELLED'}

        # Order loops by projection along strip direction.
        edge_map = _edge_by_key(bm)
        centroids = [_loop_centroid_from_keys(edge_map, loop) for loop in loops]
        base_centroid = _loop_centroid_from_keys(edge_map, self._base_loop_keys)
        if base_centroid is None:
            self.report({'ERROR'}, "Base loop centroid not found")
            return {'CANCELLED'}

        projections = []
        axis = _principal_axis([c for c in centroids if c is not None])
        for loop, c in zip(loops, centroids):
            if c is None:
                proj = 0.0
            else:
                proj = (c - base_centroid).dot(axis)
            projections.append((proj, loop))
        projections.sort(key=lambda x: x[0])
        self._ordered_loops = [loop for _, loop in projections]

        # Determine base index.
        base_keys = set(self._base_loop_keys)
        base_index = -1
        for i, loop in enumerate(self._ordered_loops):
            if set(loop) == base_keys:
                base_index = i
                break
        if base_index < 0:
            self.report({'ERROR'}, "Base loop not found in ring order")
            return {'CANCELLED'}
        self._base_index = base_index
        self._step_count = 0
        hop = self._skip_loops + 1
        max_left = base_index
        max_right = (len(self._ordered_loops) - 1) - base_index
        self._max_steps = max(max_left, max_right) // hop
        self._notified_limit = False

        self.extend = 0
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        _debug_log("invoke: base_loop_edges=%d ordered_loops=%d" % (len(self._base_loop_keys), len(self._ordered_loops)))
        return {'RUNNING_MODAL'}

    def _status_text(self):
        return f"Loop Sculpt | Extend: {self.extend} | Skip: {self._skip_loops}"

    def _select_loops_by_step(self, bm, obj, event_name):
        hop = self._skip_loops + 1
        step_count = self._step_count
        base_index = self._base_index
        loops = self._ordered_loops

        edge_map = _edge_by_key(bm)

        selected = [loops[base_index]]

        _debug_log("event: %s" % event_name)
        _debug_log("base_i=%d total_loops=%d" % (base_index, len(loops)))
        _debug_log("skip_loops=%d, hop=%d, protect_angle=%d, protection_disabled=%s" % (
            self._skip_loops,
            hop,
            self._protect_angle_deg,
            self._disable_protection,
        ))
        _debug_log("base_loop: edges=%d" % len(loops[base_index]))

        chosen_indices = [base_index]
        blocked_indices = []

        def maybe_add(idx):
            if idx < 0 or idx >= len(loops):
                return False
            loop = loops[idx]
            if self._disable_protection:
                selected.append(loop)
                return True
            protected = _loop_is_protected(loop, edge_map, self._protect_angle_deg)
            if protected:
                blocked_indices.append(idx)
                return False
            selected.append(loop)
            return True

        added_a = False
        added_b = False
        added_current_a = False
        added_current_b = False
        blocked_current = False
        for k in range(1, step_count + 1):
            idx_pos = base_index + k * hop
            idx_neg = base_index - k * hop
            chosen_indices.extend([idx_pos, idx_neg])
            added_pos = maybe_add(idx_pos)
            added_neg = maybe_add(idx_neg)
            if added_pos:
                added_b = True
            if added_neg:
                added_a = True
            if k == step_count:
                added_current_b = added_pos
                added_current_a = added_neg
                blocked_current = not added_pos and not added_neg

        _debug_log("hop=%d step_count=%d" % (hop, step_count))
        _debug_log("chosen_indices=%s" % chosen_indices)
        for idx in blocked_indices:
            _debug_log("blocked idx=%d" % idx)

        if event_name in {"WHEELUP", "WHEELDOWN"} and not self._disable_protection:
            if step_count > 0 and blocked_current:
                if not self._notified_limit:
                    self.report({'INFO'}, "Reached silhouette; further expansion blocked on one or both sides.")
                    self._notified_limit = True

        edges = set()
        for loop in selected:
            for key in loop:
                e = edge_map.get(key)
                if e:
                    edges.add(e)

        _deselect_all(bm)
        for e in edges:
            if e.is_valid:
                e.select = True
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
        return True

    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'LEFTMOUSE', 'ESC', 'RIGHTMOUSE'}:
            _debug_log("EVENT: %s %s" % (event.type, event.value))

        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS'}:
            hop = self._skip_loops + 1
            next_step = self._step_count + 1
            idx_left = self._base_index - next_step * hop
            idx_right = self._base_index + next_step * hop
            in_left = 0 <= idx_left < len(self._ordered_loops)
            in_right = 0 <= idx_right < len(self._ordered_loops)
            obj, bm = _active_bm(context)
            if not bm:
                self.report({'INFO'}, "Hover viewport + stay in Edit Mode")
                return {'RUNNING_MODAL'}
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            edge_map = _edge_by_key(bm)
            left_protected = False
            right_protected = False
            if in_left and not self._disable_protection:
                left_protected = _loop_is_protected(self._ordered_loops[idx_left], edge_map, self._protect_angle_deg)
            if in_right and not self._disable_protection:
                right_protected = _loop_is_protected(self._ordered_loops[idx_right], edge_map, self._protect_angle_deg)
            left_ok = in_left and (self._disable_protection or not left_protected)
            right_ok = in_right and (self._disable_protection or not right_protected)

            _debug_log("limit_check: attempted_step=%d" % next_step)
            _debug_log("idx_left: in_range=%s protected=%s" % (in_left, left_protected))
            _debug_log("idx_right: in_range=%s protected=%s" % (in_right, right_protected))

            if not left_ok and not right_ok:
                if not self._notified_limit:
                    self.report({'INFO'}, "Reached silhouette; further expansion blocked on one or both sides.")
                    self._notified_limit = True
                return {'RUNNING_MODAL'}

            self._step_count = min(next_step, self._max_steps)
            obj, bm = _active_bm(context)
            if not bm:
                self.report({'INFO'}, "Hover viewport + stay in Edit Mode")
                return {'RUNNING_MODAL'}
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            self._select_loops_by_step(bm, obj, "WHEELUP")
            _status(context, self._status_text())
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELDOWNMOUSE', 'NUMPAD_MINUS'}:
            self._step_count = max(0, self._step_count - 1)
            self._notified_limit = False
            obj, bm = _active_bm(context)
            if not bm:
                self.report({'INFO'}, "Hover viewport + stay in Edit Mode")
                return {'RUNNING_MODAL'}
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            self._select_loops_by_step(bm, obj, "WHEELDOWN")
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
                edge_map = _edge_by_key(bm)
                base_loop = self._base_loop_keys
                if base_loop:
                    _deselect_all(bm)
                    for key in base_loop:
                        e = edge_map.get(key)
                        if e:
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
