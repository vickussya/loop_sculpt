import bpy
import bmesh
import mathutils
from bpy.types import Operator, Panel, PropertyGroup
from bpy.props import IntProperty, PointerProperty


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


def _loop_centroid(edge_map, loop_keys):
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


def _neighbor_loop(bm, loop_edges):
    candidates = set()
    for e in loop_edges:
        for f in e.link_faces:
            if len(f.verts) != 4:
                continue
            opp = _opposite_edge_in_face(f, e)
            if opp:
                candidates.add(opp)
    if not candidates:
        return None
    # Split candidates into connected components and keep one coherent loop.
    components = []
    remaining = set(candidates)
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        stack = [start]
        remaining.remove(start)
        while stack:
            cur = stack.pop()
            for v in cur.verts:
                for le in v.link_edges:
                    if le in remaining:
                        remaining.remove(le)
                        comp.add(le)
                        stack.append(le)
        components.append(comp)
    valid = []
    for comp in components:
        ok, _reason = _validate_loop_edges(comp)
        if ok:
            valid.append(comp)
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    # Deterministic pick if two sides exist.
    valid.sort(key=lambda c: sorted(_loop_keys(c))[0])
    return valid[0]


def _get_loop_from_seed_edge(bm, obj, area, region, window, seed_edge):
    orig_keys = {_edge_key(e) for e in bm.edges if e.select}
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
    loop_keys = {_edge_key(e) for e in bm.edges if e.select}

    _deselect_all(bm)
    edge_map = _edge_map(bm)
    for key in orig_keys:
        e = edge_map.get(key)
        if e:
            e.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return loop_keys


def _get_ring_edges_from_seed(bm, obj, area, region, window, seed_edge):
    orig_keys = {_edge_key(e) for e in bm.edges if e.select}
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
    ring_keys = {_edge_key(e) for e in bm.edges if e.select}

    _deselect_all(bm)
    edge_map = _edge_map(bm)
    for key in orig_keys:
        e = edge_map.get(key)
        if e:
            e.select = True
    bmesh.update_edit_mesh(obj.data, loop_triangles=False, destructive=False)
    return ring_keys


def _neighbor_loops(bm, loop_edges):
    candidates = set()
    for e in loop_edges:
        for f in e.link_faces:
            if len(f.verts) != 4:
                continue
            opp = _opposite_edge_in_face(f, e)
            if opp:
                candidates.add(opp)
    if not candidates:
        return []
    components = []
    remaining = set(candidates)
    while remaining:
        start = next(iter(remaining))
        comp = {start}
        stack = [start]
        remaining.remove(start)
        while stack:
            cur = stack.pop()
            for v in cur.verts:
                for le in v.link_edges:
                    if le in remaining:
                        remaining.remove(le)
                        comp.add(le)
                        stack.append(le)
        components.append(comp)
    valid = []
    for comp in components:
        ok, _reason = _validate_loop_edges(comp)
        if ok:
            valid.append(comp)
    valid.sort(key=lambda c: sorted(_loop_keys(c))[0])
    return valid


def _next_loop(bm, prev_loop, curr_loop):
    neighbors = _neighbor_loops(bm, curr_loop)
    if not neighbors:
        return None
    if not prev_loop:
        return neighbors[0]
    prev_keys = _loop_keys(prev_loop)
    for loop in neighbors:
        if _loop_keys(loop) != prev_keys:
            return loop
    return None


def _edge_map(bm):
    edge_map = {}
    for e in bm.edges:
        edge_map[_edge_key(e)] = e
    return edge_map


def _step_loop(bm, prev_loop, curr_loop, steps):
    prev = prev_loop
    curr = curr_loop
    for _i in range(1, steps):
        nxt = _next_loop(bm, prev, curr)
        if not nxt:
            return None
        prev, curr = curr, nxt
    return curr


class LoopSculptSettings(PropertyGroup):
    skip_loops: IntProperty(
        name="Skip Loops",
        description="Select every Nth loop (skip N-1 in between)",
        default=2,
        min=1,
        max=5,
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
        self._win = context.window
        self._area = context.area
        self._region = context.region

        selected_edges = [e for e in bm.edges if e.select]
        _debug_log(
            "invoke: mode=%s obj=%s selected_edges=%d" %
            (context.mode, obj.name if obj else "None", len(selected_edges))
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
        self._skip_loops = settings.skip_loops if settings else 2
        self._notified_limit = False

        base_loop = _edges_from_keys(bm, self._base_loop_keys)
        if not base_loop:
            self.report({'ERROR'}, "Base loop is not valid")
            _debug_log("invoke: cancelled (base loop invalid)")
            return {'CANCELLED'}

        rep_edge = min(selected_edges, key=lambda e: e.index)
        ring_keys = _get_ring_edges_from_seed(bm, obj, self._area, self._region, self._win, rep_edge)
        if not ring_keys:
            self.report({'ERROR'}, "Edge ring not found for base loop")
            _debug_log("invoke: cancelled (ring not found)")
            return {'CANCELLED'}

        edge_map = _edge_map(bm)
        unassigned = set(ring_keys)
        loops = []
        seen = set()
        base_size = len(self._base_loop_keys)
        while unassigned:
            seed_key = next(iter(unassigned))
            seed_edge = edge_map.get(seed_key)
            if not seed_edge:
                unassigned.remove(seed_key)
                continue
            loop_keys = _get_loop_from_seed_edge(bm, obj, self._area, self._region, self._win, seed_edge)
            if not loop_keys:
                break
            if len(loop_keys) != base_size:
                unassigned -= loop_keys
                continue
            key = frozenset(loop_keys)
            if key in seen:
                unassigned -= loop_keys
                continue
            seen.add(key)
            loops.append(list(loop_keys))
            unassigned -= loop_keys

        if not loops:
            self.report({'ERROR'}, "Could not extract loops from ring")
            _debug_log("invoke: cancelled (no loops)")
            return {'CANCELLED'}

        centroids = [_loop_centroid(edge_map, loop) for loop in loops]
        axis = _principal_axis([c for c in centroids if c is not None])
        projections = []
        for loop, c in zip(loops, centroids):
            proj = c.dot(axis) if c is not None else 0.0
            projections.append((proj, loop))
        projections.sort(key=lambda x: x[0])
        self._ordered_loops = [loop for _, loop in projections]

        base_set = set(self._base_loop_keys)
        self._base_index = -1
        for i, loop in enumerate(self._ordered_loops):
            if set(loop) == base_set:
                self._base_index = i
                break
        if self._base_index < 0:
            self.report({'ERROR'}, "Base loop not found in ordered loops")
            _debug_log("invoke: cancelled (base not in ordered loops)")
            return {'CANCELLED'}

        self.extend = 0
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        self.report({'INFO'}, "Loop Sculpt: started")
        _debug_log("invoke: running (base_loop_edges=%d valid=%s)" % (len(self._base_loop_keys), ok))
        return {'RUNNING_MODAL'}

    def _status_text(self):
        return f"Loop Sculpt | Extend: {self.extend} | Skip: {self._skip_loops}"

    def _edges_for_step(self, bm, step):
        if not hasattr(self, "_ordered_loops") or self._base_index < 0:
            return None
        edge_map = _edge_map(bm)
        edges = set()

        def add_loop(loop_keys):
            for key in loop_keys:
                e = edge_map.get(key)
                if e:
                    edges.add(e)

        add_loop(self._ordered_loops[self._base_index])
        if step == 0:
            return edges

        hop = self._skip_loops
        for k in range(1, step + 1):
            idx_pos = self._base_index + k * hop
            idx_neg = self._base_index - k * hop
            if 0 <= idx_pos < len(self._ordered_loops):
                add_loop(self._ordered_loops[idx_pos])
            if 0 <= idx_neg < len(self._ordered_loops):
                add_loop(self._ordered_loops[idx_neg])

        _debug_log("loops: step=%d total=%d" % (step, len(edges)))
        return edges

    def _update_preview(self, context, bm):
        edges = self._edges_for_step(bm, self.extend)
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
        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS', 'WHEELINMOUSE'}:
            if context.area and context.area.type != 'VIEW_3D':
                return {'RUNNING_MODAL'}
            hop = self._skip_loops
            next_step = self.extend + 1
            idx_pos = self._base_index + next_step * hop
            idx_neg = self._base_index - next_step * hop
            if idx_pos >= len(self._ordered_loops) and idx_neg < 0:
                if not self._notified_limit:
                    self.report({'INFO'}, "Reached end")
                    self._notified_limit = True
                return {'RUNNING_MODAL'}
            self.extend = next_step
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("wheel up: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            if not self._update_preview(context, bm):
                self.report({'WARNING'}, "Topology limits loop traversal; cannot extend further")
                _debug_log("wheel up: blocked (topology limit)")
                _clear_status(context)
                self.extend = max(0, self.extend - 1)
                return {'RUNNING_MODAL'}
            _status(context, self._status_text())
            self.report({'INFO'}, f"extend={self.extend} step={self._skip_loops}")
            _debug_log("wheel up: extend=%d added=%d" % (self.extend, self.extend))
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELDOWNMOUSE', 'NUMPAD_MINUS', 'WHEELOUTMOUSE'}:
            if context.area and context.area.type != 'VIEW_3D':
                return {'RUNNING_MODAL'}
            self.extend = max(0, self.extend - 1)
            if self.extend == 0:
                self._notified_limit = False
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("wheel down: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            if not self._update_preview(context, bm):
                self.report({'WARNING'}, "Topology limits loop traversal; cannot shrink further")
                _debug_log("wheel down: blocked (topology limit)")
                _clear_status(context)
                return {'RUNNING_MODAL'}
            _status(context, self._status_text())
            self.report({'INFO'}, f"extend={self.extend} step={self._skip_loops}")
            _debug_log("wheel down: extend=%d removed=%d" % (self.extend, self.extend))
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
