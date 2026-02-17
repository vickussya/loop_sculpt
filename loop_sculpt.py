import bpy
import bmesh
import math
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


def _loop_centroid(loop_edges):
    total = None
    count = 0
    for e in loop_edges:
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


def _extract_components(edges):
    components = []
    remaining = set(edges)
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
    return components


def _neighbor_loop_side(loop_edges, face_set, prev_loop, base_centroid, protect_angle_deg):
    candidates = set()
    used_faces = set()
    for e in loop_edges:
        face = None
        for f in e.link_faces:
            if f in face_set:
                face = f
                break
        if not face or len(face.verts) != 4:
            continue
        opp = _opposite_edge_in_face(face, e)
        if opp:
            candidates.add(opp)
            used_faces.add(face)
    if not candidates or not used_faces:
        return None, None, []
    components = _extract_components(candidates)
    infos = []
    prev_keys = _loop_keys(prev_loop) if prev_loop else None
    base_c = base_centroid
    for comp in components:
        ok, _reason = _validate_loop_edges(comp)
        if not ok:
            continue
        if prev_keys and _loop_keys(comp) == prev_keys:
            continue
        centroid = _loop_centroid(comp)
        dist = (centroid - base_c).length if centroid and base_c else 0.0
        prot = _loop_is_protected(comp, protect_angle_deg)
        infos.append((prot, dist, comp))
    if not infos:
        return None, None, []
    infos.sort(key=lambda x: (x[0], x[1]))
    return infos[0][2], used_faces, infos


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

        self._area = context.area
        self._region = context.region

        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

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
        self._skip_loops = settings.skip_loops if settings else 1
        self._protect_angle_deg = settings.protect_angle_deg if settings else 45

        self.extend = 0
        _status(context, self._status_text())
        context.window_manager.modal_handler_add(self)
        _debug_log("invoke: running (base_loop_edges=%d valid=%s)" % (len(self._base_loop_keys), ok))
        return {'RUNNING_MODAL'}

    def _status_text(self):
        return f"Loop Sculpt | Extend: {self.extend} | Skip: {self._skip_loops}"

    def _edges_for_step(self, bm, step):
        base_loop = _edges_from_keys(bm, self._base_loop_keys)
        if not base_loop:
            return None

        base_centroid = _loop_centroid(base_loop)
        hop = self._skip_loops + 1

        side_faces = []
        e0 = next(iter(base_loop))
        for f in e0.link_faces:
            if len(f.verts) == 4:
                side_faces.append(f)
        side_faces = side_faces[:2]

        side_seed = []
        for f in side_faces:
            side_seed.append({f})

        side_selected = [[], []]
        blocked = [False, False]
        applied = [False, False]

        for side_index in range(2):
            if side_index >= len(side_seed):
                blocked[side_index] = True
                continue
            curr_loop = base_loop
            face_set = side_seed[side_index]
            prev_loop = None
            for _step in range(step):
                for _hop in range(hop):
                    nxt_loop, next_face_set, infos = _neighbor_loop_side(
                        curr_loop,
                        face_set,
                        prev_loop,
                        base_centroid,
                        self._protect_angle_deg,
                    )
                    _debug_log(
                        "event=wheel hop=%d side=%d cand=%d" %
                        (_hop + 1, side_index, len(infos))
                    )
                    for prot, dist, _comp in infos:
                        _debug_log(
                            "  cand dist=%.4f protected=%s" %
                            (dist, prot)
                        )
                    if not nxt_loop or not next_face_set:
                        blocked[side_index] = True
                        break
                    prev_loop = curr_loop
                    curr_loop = nxt_loop
                    face_set = next_face_set
                if blocked[side_index]:
                    break
                # Only check protection on the loop we intend to SELECT.
                if _loop_is_protected(curr_loop, self._protect_angle_deg):
                    blocked[side_index] = True
                    break
                side_selected[side_index].append(curr_loop)
                applied[side_index] = True

        selected_loops = [base_loop]
        for side in side_selected:
            selected_loops.extend(side)

        edges = set()
        for loop in selected_loops:
            edges.update(loop)

        _debug_log(
            "step=%d skip=%d hop=%d protect=%d base=%d base_prot=%s sideA=%d sideA_prot=%s sideB=%d sideB_prot=%s selA=%d selB=%d" % (
                step,
                self._skip_loops,
                hop,
                self._protect_angle_deg,
                len(base_loop),
                _loop_is_protected(base_loop, self._protect_angle_deg),
                len(side_selected[0][-1]) if side_selected[0] else 0,
                _loop_is_protected(side_selected[0][-1], self._protect_angle_deg) if side_selected[0] else False,
                len(side_selected[1][-1]) if side_selected[1] else 0,
                _loop_is_protected(side_selected[1][-1], self._protect_angle_deg) if side_selected[1] else False,
                len(side_selected[0]),
                len(side_selected[1]),
            )
        )

        self._last_blocked = blocked
        self._last_applied = applied

        if blocked[0] and blocked[1] and step > 0:
            return None
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
        if context.area != self._area:
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELUPMOUSE', 'NUMPAD_PLUS'}:
            if context.area and context.area.type != 'VIEW_3D':
                return {'RUNNING_MODAL'}
            self.extend += 1
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("wheel up: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            if not self._update_preview(context, bm):
                self.report({'WARNING'}, "Reached protected silhouette loop; stopping.")
                _debug_log("wheel up: blocked (protected)")
                _clear_status(context)
                self.extend = max(0, self.extend - 1)
                return {'RUNNING_MODAL'}
            if self._last_blocked[0] and self._last_blocked[1]:
                self.report({'WARNING'}, "Reached protected silhouette loop; stopping.")
            elif self._last_blocked[0] != self._last_blocked[1]:
                self.report({'INFO'}, "Side A/B blocked by silhouette; other side extended.")
            _status(context, self._status_text())
            _debug_log("wheel up: extend=%d" % self.extend)
            return {'RUNNING_MODAL'}

        if event.type in {'WHEELDOWNMOUSE', 'NUMPAD_MINUS'}:
            if context.area and context.area.type != 'VIEW_3D':
                return {'RUNNING_MODAL'}
            self.extend = max(0, self.extend - 1)
            obj, bm = _active_bm(context)
            if not bm:
                _debug_log("wheel down: cancelled (no edit mesh)")
                return {'CANCELLED'}
            bm.edges.ensure_lookup_table()
            bm.verts.ensure_lookup_table()
            bm.faces.ensure_lookup_table()
            if not self._update_preview(context, bm):
                self.report({'WARNING'}, "Reached protected silhouette loop; stopping.")
                _debug_log("wheel down: blocked (protected)")
                _clear_status(context)
                return {'RUNNING_MODAL'}
            _status(context, self._status_text())
            _debug_log("wheel down: extend=%d" % self.extend)
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
