# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**SketchImpact** is a university group project (Group 2: Rene Sebastian, Shabaz Malik, Frederick David Vandermoeten) built in **Godot 4.7** with **GDScript**. The goal is to let players draw freeform sprites that collide on a 2D billiard table, and to use a CNN model to filter expensive narrow-phase collision checks between those user-drawn sprites.

Development workflow (in order):
1. Build baseline game without AI (current stage)
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

Then press **F5** (or the Play button) to run the main scene (`draw_window.tscn`). No separate compile step; GDScript is interpreted at runtime.

There is currently no test framework. All testing is manual through the Godot editor.

## Architecture

### Current Scene Tree
```
draw_window.tscn  (Node2D)
└── TextureRect   → texture_rect.gd
```

### Key Script: `texture_rect.gd`
Extends `TextureRect`. Owns an in-memory 16×16 `Image` pixel buffer (`FORMAT_RGBA8`). Mouse input is handled in `_gui_input()` — left-click/drag calls `paint_at_mouse()`, which maps screen coordinates to pixel coordinates via a scale factor and writes directly into the `Image`, then pushes the update to the `ImageTexture` each frame.

The 16×16 resolution is intentional: it produces compact binary representations suitable as CNN input for collision prediction.

### Engine Configuration (`project.godot`)
- Rendering: Forward Plus, D3D12 (Windows)
- Physics: Jolt Physics (3D engine setting; 2D will use Godot2D/Jolt as configured)
- Stretch mode: `canvas_items` / `expand` (UI scales with window)

### Planned Components (not yet implemented)
- Physics simulation: sprites moving and bouncing on a billiard table
- Broad-phase: AABB / OBB spatial data structure
- Narrow-phase: per-pixel or polygon collision
- AI filter: CNN that takes two 16×16 sprite images + relative offset and predicts collision probability, used to skip narrow-phase checks
- HUD: stats overlay showing skipped collision check counts
