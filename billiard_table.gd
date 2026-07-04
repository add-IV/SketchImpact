extends Node2D

const TABLE_WIDTH := 820.0
const TABLE_HEIGHT := 620.0

const SPRITE_PIXEL_SIZE := 32
const IMPULSE_SPEED := 200.0

const ZOOM_MIN := 1
const ZOOM_MAX := 16

# Mutable at runtime via zoom in/out hotkeys
var _display_scale := 2

# Struct-of-arrays: all sprite state lives here
var positions: Array[Vector2] = []
var velocities: Array[Vector2] = []
var pixels: Array[PackedByteArray] = []   # alpha channel, SPRITE_PIXEL_SIZE² bytes each
var display_nodes: Array[Node2D] = []
var aabb_offsets: Array[Rect2] = []  # tight pixel-space bbox relative to sprite center, constant per sprite
var aabbs: Array[Rect2] = []         # current frame screen-space bbox, updated each frame

signal scenario_loaded(images: Array[Image])

var broad_checks := 0   # unique pairs returned by quadtree
var aabb_checks := 0    # pairs that passed AABB overlap test
var narrow_checks := 0  # pairs that reached pixel-perfect check
var model_skips := 0    # pairs the CNN filter skipped without a pixel check

# Eval mode also runs the ground-truth pixel check on pairs the model skipped,
# so it can tell false skips from true ones. This cancels the CNN's speed-up
# while active — it's a measurement tool, not the default play mode.
var eval_mode := false
var false_positives := 0  # model flagged the pair, pixel check found no collision
var false_negatives := 0  # model skipped the pair, but a collision was actually there
var eval_flagged_total := 0  # pairs the model flagged, seen while eval mode was on
var eval_skipped_total := 0  # pairs the model skipped, seen while eval mode was on

var _collision_model_scene := preload("res://collision_model.gd")
var collision_model: Node

var _show_bounding_boxes := false
var _show_quadtree := false
var _qt_root = null

@onready var sprites_container: Node2D = $Sprites
@onready var help_overlay: Panel = $HelpOverlay

var _sprite_scene: PackedScene = preload("res://sprite_body.tscn")


func _ready() -> void:
	add_to_group("billiard_table")
	collision_model = _collision_model_scene.new()
	add_child(collision_model)


func _set_display_scale(new_scale: int) -> void:
	_display_scale = clampi(new_scale, ZOOM_MIN, ZOOM_MAX)
	for node in display_nodes:
		node.get_node("Sprite2D").scale = Vector2(_display_scale, _display_scale)


func spawn_sprite(image: Image) -> void:
	if image.is_invisible():
		return
	if image.get_width() != SPRITE_PIXEL_SIZE or image.get_height() != SPRITE_PIXEL_SIZE:
		return
	var px := _extract_pixels(image)
	var offset := _tight_aabb_offset(px)
	var scale := float(_display_scale)
	var pos := Vector2(
		randf_range(-offset.position.x * scale + 10.0, TABLE_WIDTH  - (offset.position.x + offset.size.x) * scale - 10.0),
		randf_range(-offset.position.y * scale + 10.0, TABLE_HEIGHT - (offset.position.y + offset.size.y) * scale - 10.0)
	)
	var vel := Vector2(cos(randf() * TAU), sin(randf() * TAU)) * randf_range(80.0, 160.0)
	_do_spawn(image, px, offset, pos, vel)


func _do_spawn(image: Image, px: PackedByteArray, offset: Rect2, pos: Vector2, vel: Vector2) -> void:
	var node: Node2D = _sprite_scene.instantiate()
	var sprite: Sprite2D = node.get_node("Sprite2D")
	sprite.texture = ImageTexture.create_from_image(image)
	sprite.scale = Vector2(_display_scale, _display_scale)
	sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	node.position = pos
	sprites_container.add_child(node)

	var idx := positions.size()
	positions.append(pos)
	velocities.append(vel)
	pixels.append(px)
	display_nodes.append(node)
	aabb_offsets.append(offset)
	aabbs.append(Rect2(pos + offset.position * _display_scale, offset.size * _display_scale))
	collision_model.register_sprite(idx, px)


func clear_sprites() -> void:
	for node in display_nodes:
		node.queue_free()
	positions.clear()
	velocities.clear()
	pixels.clear()
	display_nodes.clear()
	aabb_offsets.clear()
	aabbs.clear()
	collision_model.clear()


func _parse_scenario_json(file_path: String) -> Variant:
	if not FileAccess.file_exists(file_path):
		return null
	var file := FileAccess.open(file_path, FileAccess.READ)
	if file == null:
		return null
	var json := JSON.new()
	if json.parse(file.get_as_text()) != OK:
		return null
	return json.data


func _load_sprites_from_data(data: Dictionary) -> Array[Image]:
	var loaded_images: Array[Image] = []
	for sprite_data in data.get("sprites", []):
		var sprite_path: String = sprite_data.get("file", "")
		if sprite_path.is_empty() or not FileAccess.file_exists(sprite_path):
			continue
		var img := Image.load_from_file(sprite_path)
		if img == null:
			continue
		if img.get_width() != SPRITE_PIXEL_SIZE or img.get_height() != SPRITE_PIXEL_SIZE:
			continue
		var pos := Vector2(float(sprite_data.get("x", 0.0)), float(sprite_data.get("y", 0.0)))
		var vel := Vector2(float(sprite_data.get("vx", 0.0)), float(sprite_data.get("vy", 0.0)))
		var px := _extract_pixels(img)
		_do_spawn(img, px, _tight_aabb_offset(px), pos, vel)
		loaded_images.append(img)
	return loaded_images


func _load_images_from_data(data: Dictionary) -> Array[Image]:
	var loaded_images: Array[Image] = []
	for entry in data.get("images", []):
		var image_path: String = (entry.get("file", "") if entry is Dictionary else entry) as String
		var spawn_count: int = int(entry.get("count", 0)) if entry is Dictionary else 0
		if image_path.is_empty() or not FileAccess.file_exists(image_path):
			continue
		var img := Image.load_from_file(image_path)
		if img == null:
			continue
		if img.get_width() != SPRITE_PIXEL_SIZE or img.get_height() != SPRITE_PIXEL_SIZE:
			img.resize(SPRITE_PIXEL_SIZE, SPRITE_PIXEL_SIZE, Image.INTERPOLATE_NEAREST)
		loaded_images.append(img)
		for i in range(spawn_count):
			spawn_sprite(img.duplicate())
	return loaded_images


func load_scenario(file_path: String) -> bool:
	var parsed: Variant = _parse_scenario_json(file_path)
	if parsed == null:
		return false
	var data: Dictionary = parsed
	clear_sprites()
	if data.has("display_scale"):
		_set_display_scale(int(data["display_scale"]))
	var loaded_images: Array[Image] = []
	loaded_images.append_array(_load_sprites_from_data(data))
	loaded_images.append_array(_load_images_from_data(data))
	scenario_loaded.emit(loaded_images)
	return true


func _physics_process(delta: float) -> void:
	var n := positions.size()
	broad_checks = 0
	aabb_checks = 0
	narrow_checks = 0
	model_skips = 0

	for i in range(n):
		positions[i] += velocities[i] * delta
		_bounce_walls(i)
		aabbs[i] = Rect2(positions[i] + aabb_offsets[i].position * _display_scale, aabb_offsets[i].size * _display_scale)

	_qt_root = null
	if n > 1:
		var qt := _QuadNode.new(Rect2(0.0, 0.0, TABLE_WIDTH, TABLE_HEIGHT), 0, aabbs)
		for i in range(n):
			qt.insert(i)
		_qt_root = qt
		var pairs: Dictionary = {}
		qt.collect_pairs(pairs)
		broad_checks = pairs.size()

		var candidates: Array = []
		for key in pairs:
			var i: int = key[0]
			var j: int = key[1]
			if aabbs[i].intersects(aabbs[j]):
				aabb_checks += 1
				candidates.append([i, j])

		var queries: Array = []
		for pair in candidates:
			var i: int = pair[0]
			var j: int = pair[1]
			var dx := (positions[j].x - positions[i].x) / float(_display_scale)
			var dy := (positions[j].y - positions[i].y) / float(_display_scale)
			queries.append([i, j, dx, dy])
		var maybe: Array = collision_model.query_pairs(queries)

		for k in range(candidates.size()):
			var predicted_collision: bool = maybe[k]
			if not predicted_collision:
				model_skips += 1
				if not eval_mode:
					continue

			var i: int = candidates[k][0]
			var j: int = candidates[k][1]
			if predicted_collision:
				narrow_checks += 1
			var col_pt := _pixel_collision(i, j)
			var actual_collision := col_pt.is_finite()

			if eval_mode:
				if predicted_collision:
					eval_flagged_total += 1
					if not actual_collision:
						false_positives += 1
				else:
					eval_skipped_total += 1
					if actual_collision:
						false_negatives += 1

			if predicted_collision and actual_collision:
				_resolve(i, j, col_pt)

	for i in range(n):
		display_nodes[i].position = positions[i]

	queue_redraw()


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("help"):
		help_overlay.visible = not help_overlay.visible
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("show bounding boxes"):
		_show_bounding_boxes = not _show_bounding_boxes
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("draw quadtree"):
		_show_quadtree = not _show_quadtree
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("freeze"):
		for i in range(velocities.size()):
			velocities[i] = Vector2.ZERO
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("toggle model"):
		collision_model.enabled = not collision_model.enabled
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("toggle eval"):
		eval_mode = not eval_mode
		if eval_mode:
			false_positives = 0
			false_negatives = 0
			eval_flagged_total = 0
			eval_skipped_total = 0
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("zoom in"):
		_set_display_scale(_display_scale * 2)
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("zoom out"):
		@warning_ignore("integer_division")
		_set_display_scale(_display_scale / 2)
		get_viewport().set_input_as_handled()
		return
	if event is InputEventKey and event.pressed and not event.echo:
		var key_event := event as InputEventKey
		if key_event.keycode >= KEY_0 and key_event.keycode <= KEY_5:
			load_scenario("res://scenarios/scenario_%d.json" % (key_event.keycode - KEY_0))
			get_viewport().set_input_as_handled()
			return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		var click := get_local_mouse_position()
		if not Rect2(0.0, 0.0, TABLE_WIDTH, TABLE_HEIGHT).has_point(click):
			return
		for i in range(positions.size()):
			if aabbs[i].has_point(click):
				var angle := randf() * TAU
				velocities[i] = Vector2(cos(angle), sin(angle)) * IMPULSE_SPEED
				get_viewport().set_input_as_handled()
				break


func _draw() -> void:
	if _show_bounding_boxes:
		for i in range(positions.size()):
			draw_rect(aabbs[i], Color(1.0, 1.0, 0.0, 0.8), false, 1.0)
	if _show_quadtree and _qt_root != null:
		_draw_qt_node(_qt_root)


func _draw_qt_node(node) -> void:
	draw_rect(node.bounds, Color(0.0, 0.8, 1.0, 0.4), false, 1.0)
	for child in node.children:
		_draw_qt_node(child)


func _tight_aabb_offset(px: PackedByteArray) -> Rect2:
	var s := SPRITE_PIXEL_SIZE
	var min_x := s; var max_x := -1
	var min_y := s; var max_y := -1
	for y in range(s):
		for x in range(s):
			if px[y * s + x] > 0:
				if x < min_x: min_x = x
				if x > max_x: max_x = x
				if y < min_y: min_y = y
				if y > max_y: max_y = y
	var half := s * 0.5
	return Rect2(
		Vector2(min_x - half, min_y - half),
		Vector2(max_x - min_x + 1, max_y - min_y + 1)
	)


func _bounce_walls(i: int) -> void:
	var pos := positions[i]
	var vel := velocities[i]
	var o := aabb_offsets[i]
	var scale := float(_display_scale)
	var left   := o.position.x * scale
	var top    := o.position.y * scale
	var right  := left + o.size.x * scale
	var bottom := top  + o.size.y * scale
	if pos.x + left < 0.0:
		pos.x = -left
		vel.x = absf(vel.x)
	elif pos.x + right > TABLE_WIDTH:
		pos.x = TABLE_WIDTH - right
		vel.x = -absf(vel.x)
	if pos.y + top < 0.0:
		pos.y = -top
		vel.y = absf(vel.y)
	elif pos.y + bottom > TABLE_HEIGHT:
		pos.y = TABLE_HEIGHT - bottom
		vel.y = -absf(vel.y)
	positions[i] = pos
	velocities[i] = vel


func _extract_pixels(image: Image) -> PackedByteArray:
	var s := SPRITE_PIXEL_SIZE
	var px := PackedByteArray()
	px.resize(s * s)
	for y in range(s):
		for x in range(s):
			px[y * s + x] = int(image.get_pixel(x, y).a * 255.0)
	return px


func _find_overlapping_j_pixel(pxa: int, pya: int, px_j: PackedByteArray, dx: float, dy: float) -> Vector2i:
	var s := SPRITE_PIXEL_SIZE
	var pxb_f := float(pxa) - dx
	var pyb_f := float(pya) - dy
	for pxb in [int(floor(pxb_f)), int(ceil(pxb_f))]:
		if pxb < 0 or pxb >= s:
			continue
		for pyb in [int(floor(pyb_f)), int(ceil(pyb_f))]:
			if pyb < 0 or pyb >= s:
				continue
			if px_j[pyb * s + pxb] > 0:
				return Vector2i(pxb, pyb)
	return Vector2i(-1, -1)


func _pixel_collision(i: int, j: int) -> Vector2:
	# Returns the screen-space collision point, or Vector2.INF if no collision.
	var s := SPRITE_PIXEL_SIZE
	var dx := (positions[j].x - positions[i].x) / float(_display_scale)
	var dy := (positions[j].y - positions[i].y) / float(_display_scale)
	var px_i := pixels[i]
	var px_j := pixels[j]
	var pxa_lo := clampi(int(floor(dx)),        0, s - 1)
	var pxa_hi := clampi(int(ceil(dx)) + s - 1, 0, s - 1)
	var pya_lo := clampi(int(floor(dy)),        0, s - 1)
	var pya_hi := clampi(int(ceil(dy)) + s - 1, 0, s - 1)
	var pxa_step := -1 if dx > 0 else 1
	var pya_step := -1 if dy > 0 else 1
	var pxa_start := pxa_hi if pxa_step < 0 else pxa_lo
	var pya_start := pya_hi if pya_step < 0 else pya_lo
	var half := s * 0.5
	for pya in range(pya_start, pya_start + (pya_hi - pya_lo + 1) * pya_step, pya_step):
		for pxa in range(pxa_start, pxa_start + (pxa_hi - pxa_lo + 1) * pxa_step, pxa_step):
			if px_i[pya * s + pxa] == 0:
				continue
			var hit := _find_overlapping_j_pixel(pxa, pya, px_j, dx, dy)
			if hit.x != -1:
				var pt_i := positions[i] + Vector2(pxa + 0.5 - half, pya + 0.5 - half) * _display_scale
				var pt_j := positions[j] + Vector2(hit.x + 0.5 - half, hit.y + 0.5 - half) * _display_scale
				return (pt_i + pt_j) * 0.5
	return Vector2.INF


const SEPARATION_PUSH := 1.0  # native sprite pixels to push each sprite apart per resolved collision

func _resolve(i: int, j: int, collision_pt: Vector2) -> void:
	# Center-to-center first: near a full overlap, dx/dy hover around 0, and
	# _pixel_collision's scan direction flips on the sign of dx/dy, so
	# collision_pt can jump to a different part of the sprite between frames.
	# Using it as the push direction made separation flip-flop instead of
	# converging. Center-to-center moves smoothly, so once the first push
	# nudges the centers apart it keeps pointing the same way.
	var sep := positions[j] - positions[i]
	if sep.length_squared() < 0.0001:
		sep = positions[j] - collision_pt
	if sep.length_squared() < 0.0001:
		sep = Vector2(1.0, 0.0)
	else:
		sep = sep.normalized()
	# Only swap velocities if sprites are approaching each other along the normal.
	# Skipping this check caused repeated swaps on overlapping frames, producing jitter.
	var vi_dot := velocities[i].dot(sep)
	var vj_dot := velocities[j].dot(sep)
	if vi_dot > vj_dot:
		velocities[i] += sep * (vj_dot - vi_dot)
		velocities[j] += sep * (vi_dot - vj_dot)
	# Push pixel masks apart along the normal so they don't stay interlocked;
	# unlike the old circle-based formula, pixel-perfect masks have no closed-form
	# overlap depth, so this nudges them apart gradually over a few frames instead.
	positions[i] -= sep * SEPARATION_PUSH * _display_scale
	positions[j] += sep * SEPARATION_PUSH * _display_scale


class _QuadNode:
	const MAX_ITEMS := 4
	const MAX_DEPTH := 5

	var bounds: Rect2
	var depth: int
	var aabbs: Array[Rect2]  # shared reference, owned by _physics_process
	var items: Array[int]
	var children: Array  # Array of _QuadNode, empty when leaf

	func _init(b: Rect2, d: int, a: Array[Rect2]) -> void:
		bounds = b
		depth = d
		aabbs = a
		items = []
		children = []

	func insert(idx: int) -> void:
		if not bounds.intersects(aabbs[idx]):
			return
		if children.is_empty():
			items.append(idx)
			if items.size() > MAX_ITEMS and depth < MAX_DEPTH:
				_subdivide()
		else:
			for child in children:
				child.insert(idx)

	func _subdivide() -> void:
		var half := bounds.size * 0.5
		children = [
			_QuadNode.new(Rect2(bounds.position, half), depth + 1, aabbs),
			_QuadNode.new(Rect2(bounds.position + Vector2(half.x, 0.0), half), depth + 1, aabbs),
			_QuadNode.new(Rect2(bounds.position + Vector2(0.0, half.y), half), depth + 1, aabbs),
			_QuadNode.new(Rect2(bounds.position + half, half), depth + 1, aabbs),
		]
		var old := items.duplicate()
		items = []
		for it in old:
			for child in children:
				child.insert(it)

	func collect_pairs(out: Dictionary) -> void:
		if children.is_empty():
			for a in range(items.size()):
				for b in range(a + 1, items.size()):
					var ia: int = items[a]
					var ib: int = items[b]
					if ia > ib:
						var t := ia
						ia = ib
						ib = t
					out[[ia, ib]] = true
		else:
			for child in children:
				child.collect_pairs(out)
