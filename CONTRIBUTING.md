# Contributing

Thanks for contributing to Loop Sculpt! This add-on targets Blender 3.6 LTS and 4.x.

## Quick Start
- Clone the repo and keep the add-on layout as `loop_sculpt/__init__.py`.
- Test in Blender by installing a zip containing the `loop_sculpt` folder.

## Coding Guidelines
- Keep compatibility with Blender 3.6 and 4.x APIs.
- Use `bmesh` for edit-mode operations and update the mesh after changes.
- Keep modal preview selection-only; apply edits on confirm.
- Add short comments only when the logic is not obvious.

## Testing Checklist
- Modal wheel behavior (extend up/down).
- Confirm/cancel selection restore and undo.
- Filters: vertex group, material, connected region.
- Install from a zipped `loop_sculpt` folder.

## Submitting Changes
- Keep commits focused and descriptive.
- If you change behavior, note it in your PR description or commit message.
