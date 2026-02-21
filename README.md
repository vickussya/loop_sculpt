# Loop Sculpt

Loop Sculpt is a selection-only Blender add-on for stepping edge loop selection in Edit Mode.

## Features
- Selection-only loop stepping (no geometry changes)
- Mouse wheel expands and shrinks symmetrically on both sides
- Skip Loops setting (1-5) to control spacing
- UI button activation (no shortcuts)
- Border/corner loop exclusion (boundary loops are never selected)

## Installation (Blender 3.6+ / 4.x)
1. Download `loop_sculpt.zip` from the GitHub Releases page or use the GitHub **Download ZIP** button.
   - The ZIP contains: `__init__.py` and `loop_sculpt.py` at the top level.
2. In Blender, go to **Edit > Preferences > Add-ons**.
3. Click **Install...** and select the ZIP file.
4. Enable the add-on by checking **Loop Sculpt** in the add-ons list.

## Usage
1. Enter **Edit Mode** on a mesh and use **Edge Select**.
2. Alt+Click to select a single edge loop.
3. Open **View3D > Sidebar > Loop Sculpt** and click **Loop Sculpt**.
4. Hover the mouse over the 3D Viewport and use the mouse wheel:
   - Wheel Up: add outer loops symmetrically on both sides using skip spacing.
   - Wheel Down: remove the last added loops.
5. Confirm with **Left Mouse** (selection stays).
6. Cancel with **Esc** or **Right Mouse** (selection returns to the original loop).

## Options
- **Skip Loops (1-5)**: controls spacing between selected loops (default: 2).
  - Example: Skip=2 selects every third loop.

## Notes
- Best on quad-based topology.
- The tool only changes selection; it never modifies geometry.
- Boundary/border loops are never selected.
