# Loop Sculpt

Loop Sculpt is a Blender add-on designed to help artists clean up heavy topology faster without ruining the form of the mesh. It is best for cases where you need extra loop cuts to shape the form, then reduce the loop count to make rendering easier. It is especially helpful for dense meshes such as hair. The add-on was created from real-world modeling experience with the goal of speeding up the workflow.

## Features
- Modal edge-loop dissolve controlled by mouse wheel
- Step-based dissolve (e.g., dissolve every other loop)
- Filters for hair-only workflows (vertex group, material, connected region)
- Undo-friendly, non-destructive preview

## Installation (Blender 3.6+ / 4.x)
1. Download `loop_sculpt.zip` from the GitHub Releases page.
   - The ZIP contains: `__init__.py` and `loop_sculpt.py` at the top level.
2. Note: the GitHub "Download ZIP" is **not** installable directly because GitHub wraps the repo in a `loop_sculpt-main/` folder, which Blender treats as an invalid module name.
3. In Blender, go to **Edit > Preferences > Add-ons**.
4. Click **Install...** and select the ZIP file.
5. Enable the add-on by checking **Loop Sculpt** in the add-ons list.

## Usage
1. Enter **Edit Mode** on a mesh and use **Edge Select**.
2. Select a single edge that belongs to an edge loop.
3. Press **Ctrl+X** to start the modal tool.
4. Use the mouse wheel to increase/decrease the number of loops previewed.
5. Confirm with **Left Mouse** or **Enter**, or cancel with **Right Mouse** / **Esc**.
6. Settings and filters are available in **View3D > Sidebar > Edit > Retopo Cleanup**.
