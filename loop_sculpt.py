import bpy
import bmesh
import math
import mathutils
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import BoolProperty, IntProperty, PointerProperty, FloatProperty


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


def _edges_share_ring_vertices(loop_edges, ring_edges):
    ring_verts = set()
    for e in ring_edges:
        ring_verts.update(e.verts)
    for e in loop_edges:
        for v in e.verts:
            if v in ring_verts:
                return True
    return False


def _rep_edge_from_keys(edge_map, loop_keys):
    for key in loop_keys:
        e = edge_map.get(key)
        if e:
            return e
    return None


def _pick_side_face(edge, target_normal):
    best = None
    best_dot = -2.0
    for f in edge.link_faces:
        if len(f.verts) != 4:
            continue
        f.normal_update()
        dot = f.normal.dot(target_normal)
        if dot > best_dot:
            best_dot = dot
            best = f
    return best


def _opposite_edge_in_face(face, edge):
    for e in face.edges:
        if e is edge:
            continue
        if e.verts[0] not in edge.verts and e.verts[1] not in edge.verts:
            return e
    return None


def _loop_from_seed_keys(bm, obj, area, region, window, seed_edge):
    loop = _get_loop_from_seed_edge(bm, obj, area, region, window, seed_edge)
    if not loop:
        return None
    return set(_loop_keys(loop))


class LoopSculptSettings(PropertyGroup):
    skip_loops: int = IntProperty(
        name="Skip Loops",
        description="Number of loops to skip between selected loops",
        default=1,
        min=1,
        max=5,
    )
    protect_angle_deg: float = FloatProperty(
        name="Protect Angle",
        description="Edges with face angle >= this value are treated as silhouette and will not be selected",
        default=45,
        min=1,
        max=89,
    )
    disable_protection: bool = BoolProperty(
        name="Disable Protection (Debug)",
        description="Ignore silhouette protection for debugging",
        default=False,
    )


class MESH_OT_loop_sculpt(Operator):
    bl_idname = "mesh.loop_sculpt"
    bl_label = "Loop Sculpt"
    bl_options = {'REGISTER', 'UNDO', 'BLOCKING'}

    extend: int = IntProperty(default=0, min=0)

    def _loop_is_protected_keys(self, loop_keys, edge_map):
        if self._disable_protection:
            return False
        for key in loop_keys:
            e = edge_map.get(key)
            if not e:
                continue
            if _is_boundary_edge(e):
                return True
            if getattr(e, "use_edge_sharp", False) or getattr(e, "seam", False):
                return True
            if _edge_dihedral_deg(e) >= self._protect_angle_deg:
                return True
        return False

    def _step_adjacent_loop(self, bm, obj, edge_map, current_keys, side_normal):
        rep = _rep_edge_from_keys(edge_map, current_keys)
        if not rep:
            return None
        face = _pick_side_face(rep, side_normal)
        if not face:
            return None
        opp = _opposite_edge_in_face(face, rep)
        if not opp:
            return None
        next_keys = _loop_from_seed_keys(bm, obj, self._area, self._region, self._win, opp)
        if not next_keys:
            return None
        if len(next_keys) != self._base_loop_size:
            return None
        if next_keys == current_keys:
            return None
        return next_keys

    def _build_side_loops(self, bm, obj, edge_map, side_normal):
        loops = []
        seen = set()
        current = set(self._base_loop_keys)
        while True:
            bm = bmesh.from_edit_mesh(obj.data)
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            edge_map = _edge_by_key(bm)
            next_keys = self._step_adjacent_loop(bm, obj, edge_map, current, side_normal)
            if not next_keys:
                break
            key = frozenset(next_keys)
            if key in seen:
                break
            seen.add(key)
            if self._loop_is_protected_keys(next_keys, edge_map):
                break
            loops.append(list(next_keys))
            current = next_keys
        return loops

    def _apply_selection(self, bm, obj):
        hop = self._skip_loops + 1
        edge_map = _edge_by_key(bm)
        _deselect_all(bm)
        # Base loop always selected.
        for key in self._base_loop_keys:
            e = edge_map.get(key)
            if e:
                e.select = True
        # Add stepped loops symmetrically.
        for k in range(1, self._step_count + 1):
            idx = k * hop - 1
            if idx < len(self._left_loops):
                for key in self._left_loops[idx]:
                    e = edge_map.get(key)
                    if e:
                        e.select = True
            if idx < len(self._right_loops):
                for key in self._right_loops[idx]:
                    e = edge_map.get(key)
                    if e:
                        e.select = True
        bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)

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
        self._base_loop_size = len(self._base_loop_keys)
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

        quad_faces = [f for f in base_rep.link_faces if len(f.verts) == 4]
        if not quad_faces:
            self.report({'ERROR'}, "Base loop has no quad faces to step across")
            return {'CANCELLED'}
        if len(quad_faces) < 2:
            self.report({'WARNING'}, "Only one side available for expansion")

        edge_map = _edge_by_key(bm)
        left_normal = quad_faces[0].normal.copy()
        right_normal = quad_faces[1].normal.copy() if len(quad_faces) > 1 else quad_faces[0].normal.copy()

        self._left_loops = self._build_side_loops(bm, obj, edge_map, left_normal)
        self._right_loops = self._build_side_loops(bm, obj, edge_map, right_normal)

        self._step_count = 0
        self._notified_limit = False

        self.extend = 0
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        _debug_log("invoke: base_loop_edges=%d left_loops=%d right_loops=%d" % (
            len(self._base_loop_keys), len(self._left_loops), len(self._right_loops)))
        return {'RUNNING_MODAL'}

    def _status_text(self):
        return f"Loop Sculpt | Extend: {self.extend} | Skip: {self._skip_loops}"

    def _select_loops_by_step(self, bm, obj, event_name):
        hop = self._skip_loops + 1
        step_count = self._step_count

        _debug_log("event: %s" % event_name)
        _debug_log("hop=%d step_count=%d" % (hop, step_count))
        _debug_log("base_i=0 total_loops_left=%d total_loops_right=%d" % (
            len(self._left_loops), len(self._right_loops)))

        intended = [k * hop - 1 for k in range(1, step_count + 1)]
        selected = []
        blocked = []
        for k in range(1, step_count + 1):
            idx = k * hop - 1
            left_ok = idx < len(self._left_loops)
            right_ok = idx < len(self._right_loops)
            if left_ok:
                selected.append(("L", idx))
            else:
                blocked.append(("L", idx, "out_of_range"))
            if right_ok:
                selected.append(("R", idx))
            else:
                blocked.append(("R", idx, "out_of_range"))

        _debug_log("intended_indices=%s" % intended)
        _debug_log("selected_indices=%s" % selected)
        for side, idx, reason in blocked:
            _debug_log("blocked %s idx=%d reason=%s" % (side, idx, reason))

        self._apply_selection(bm, obj)
        return True

    def modal(self, context, event):
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE', 'LEFTMOUSE', 'ESC', 'RIGHTMOUSE'}:
            _debug_log("EVENT: %s %s" % (event.type, event.value))

        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS'}:
            obj, bm = _active_bm(context)
            if not bm:
                self.report({'INFO'}, "Hover viewport + stay in Edit Mode")
                return {'RUNNING_MODAL'}
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            hop = self._skip_loops + 1
            next_step = self._step_count + 1
            idx = next_step * hop - 1
            left_ok = idx < len(self._left_loops)
            right_ok = idx < len(self._right_loops)
            if not left_ok and not right_ok:
                if not self._notified_limit:
                    self.report({'INFO'}, "Reached silhouette; further expansion blocked on one or both sides.")
                    self._notified_limit = True
                return {'RUNNING_MODAL'}
            self._step_count = next_step
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
