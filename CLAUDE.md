# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SketchImpact** is a university group project (Group 2: Rene Sebastian, Shabaz Malik, Frederick David Vandermoeten) built in **Godot 4.7** with **GDScript**. The goal is to let players draw freeform sprites that collide on a 2D billiard table, and to use a CNN model to filter expensive narrow-phase collision checks between those user-drawn sprites.

Development workflow (in order):
1. Build baseline game without AI — **done**
2. Generate labeled dataset of sprite pairs, relative offsets, and collision outcomes
3. Train a CNN model
4. Integrate the model into the game
5. Evaluate performance vs. baseline

## Running the Project

There is no CLI build command — open and run entirely through the **Godot 4.7 editor**.

Launch the editor:
```
& "E:\Steam\steamapps\common\Godot Engine\godot.windows.opt.tools.64.exe" --path "C:\Users\add-iv\Documents\uni\ss2026\cg\SketchImpact"
```

Then press **F5** (or the Play button) to run the main scene (`sketch_impact.tscn`). No separate compile step; GDScript is interpreted at runtime. There is currently no test framework — all testing is manual through the Godot editor.

## Architecture

### Scene Tree

```
sketch_impact.tscn  (Node2D)
├── BilliardTable   → billiard_table.tscn / billiard_table.gd
│   ├── Background  (ColorRect 820×620, green)
│   ├── Sprites     (Node2D — runtime container for sprite_body instances)
│   └── HUD         (Label — live collision stats)
└── DrawWindow      → draw_window.tscn / save_image.gd
    └── (UI: TextureRect canvas, ColorPickerButton, GridContainer preview, buttons)
```

`sprite_body.tscn` is a minimal scene (Node2D + Sprite2D) instantiated at runtime by `billiard_table.gd`.

### Data Flow: Drawing → Simulation

1. User draws on the 16×16 `Image` in `texture_rect.gd` (pixel painting via `_gui_input`).
2. **Save button** (`save_image.gd::_on_button_pressed`): duplicates the image, saves it to `user://sprites/sprite_NNNN.png`, and shifts the preview grid.
3. **Spawn button** (`save_image.gd::_on_spawn_pressed`): calls `billiard_table.spawn_sprite(image)`, which extracts the alpha channel into a `PackedByteArray`, assigns a random position/velocity, and instantiates a `sprite_body` node.

### Collision Pipeline (`billiard_table.gd::_process`)

Each frame, for all sprite pairs:

1. **Quadtree broad-phase** — `_QuadNode` (inner class, `MAX_ITEMS=4`, `MAX_DEPTH=5`) collects candidate pairs. Each pair is keyed as `[min_idx, max_idx]` in a `Dictionary` to deduplicate across quadrants.
2. **AABB check** — `_aabb(i).intersects(_aabb(j))` using square bounding boxes sized `SPRITE_PIXEL_SIZE * DISPLAY_SCALE`.
3. **Pixel-perfect narrow-phase** — `_pixel_collision(i, j)` computes sub-pixel overlap by checking both `floor` and `ceil` candidate pixels of sprite `j` for each opaque pixel of sprite `i`.
4. **Elastic resolution** — `_resolve(i, j)` swaps velocity components along the collision normal and separates positions.

Stats are shown in the HUD: `Sprites | Quadtree pairs | AABB hits | Pixel checks`.

### Sprite Representation

`billiard_table.gd` uses a **struct-of-arrays** layout:
- `positions: Array[Vector2]` — screen-space center
- `velocities: Array[Vector2]`
- `pixels: Array[PackedByteArray]` — alpha channel only, `SPRITE_PIXEL_SIZE²` bytes (0 = transparent, 255 = opaque)
- `display_nodes: Array[Node2D]` — corresponding scene nodes

All arrays are indexed in sync; there is no per-sprite object.

### Key Constants (`billiard_table.gd`)

```
SPRITE_PIXEL_SIZE = 16    # native image resolution — change together with DISPLAY_SCALE
DISPLAY_SCALE     = 3     # screen pixels per sprite pixel
TABLE_WIDTH       = 820   # must match Background ColorRect size
TABLE_HEIGHT      = 620
```

### Engine Configuration (`project.godot`)
- Rendering: Forward Plus, D3D12 (Windows)
- Stretch mode: `canvas_items` / `expand` (UI scales with window)

## Next Steps (Planned)

- **Dataset generation**: export `(image_i, image_j, dx, dy, collided)` tuples during simulation for CNN training input.
- **CNN integration**: replace or gate the `_pixel_collision` call with a model inference that predicts collision probability from two 16×16 images + relative offset.
- **Performance HUD**: extend the existing HUD label to show skipped narrow-phase checks once the CNN filter is active.
