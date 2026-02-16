# AGENTS.md

## Purpose
This repository contains a Blender add-on. Follow these best practices when editing or extending it.

## Contributing
- See `CONTRIBUTING.md` for contribution and testing guidelines.

## General
- Keep the add-on installable as a single folder named `loop_sculpt` with `__init__.py` as the entry point.
- Avoid breaking Blender 3.6 compatibility; prefer API that works in 3.6 and 4.x.
- Keep the add-on self-contained; avoid external dependencies.
- Use ASCII in source files unless there is a strong reason not to.

## Code Style
- Prefer small, focused functions and clear naming.
- Add short comments only where logic is non-obvious.
- Keep modal logic and data preparation separate for readability.
- Use `bmesh` for edit operations and update the mesh after edits.

## Blender Add-on Conventions
- Maintain accurate `bl_info` metadata.
- Register/unregister all classes and keymaps cleanly.
- Use `PointerProperty` on `Scene` for settings and keep defaults sensible.
- Ensure operators fail gracefully with clear `report` messages.

## Modal Operator Guidelines
- Never modify the mesh data in preview; only adjust selection.
- Restore original selection on cancel.
- Update status text during modal and clear it on exit.
- Keep undo support enabled on final apply.

## UX / UI
- Keep the panel simple and grouped (primary action, then options/filters).
- Expose only necessary options and document behavior in tooltips.
- Keep keymap bindings scoped to Mesh Edit Mode.

## Testing Checklist
- Test in Blender 3.6 LTS and a 4.x version.
- Verify modal wheel behavior, confirm/cancel, and undo.
- Verify filters (vertex group, material, connected region).
- Verify that the add-on installs from a zipped `loop_sculpt` folder.
