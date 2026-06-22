extends Node2D

const TABLE_WIDTH := 820.0
const TABLE_HEIGHT := 620.0

const SPRITE_PIXEL_SIZE := 32
const IMPULSE_SPEED := 200.0

const ZOOM_MIN := 1
const ZOOM_MAX := 16

# Mutable at runtime via zoom in/out hotkeys
var _display_scale := 2
var _sprite_screen_size: float = SPRITE_PIXEL_SIZE * 2
var _sprite_half: float = SPRITE_PIXEL_SIZE * 2 / 2.0

# Struct-of-arrays: all sprite state lives here
var positions: Array[Vector2] = []
var velocities: Array[Vector2] = []
var pixels: Array[PackedByteArray] = []   # alpha channel, SPRITE_PIXEL_SIZE² bytes each
var display_nodes: Array[Node2D] = []

var broad_checks := 0   # unique pairs returned by quadtree
var aabb_checks := 0    # pairs that passed AABB overlap test
var narrow_checks := 0  # pairs that reached pixel-perfect check

var _show_bounding_boxes := false
var _show_quadtree := false
var _qt_root = null

@onready var hud_label: Label = $HUD
@onready var sprites_container: Node2D = $Sprites
@onready var help_overlay: Panel = $HelpOverlay

var _sprite_scene: PackedScene = preload("res://sprite_body.tscn")


func _set_display_scale(new_scale: int) -> void:
	_display_scale = clampi(new_scale, ZOOM_MIN, ZOOM_MAX)
	_sprite_screen_size = float(SPRITE_PIXEL_SIZE * _display_scale)
	_sprite_half = _sprite_screen_size / 2.0
	for node in display_nodes:
		node.get_node("Sprite2D").scale = Vector2(_display_scale, _display_scale)


func spawn_sprite(image: Image) -> void:
	if image.is_invisible():
		return
	var total := SPRITE_PIXEL_SIZE * SPRITE_PIXEL_SIZE
	var px := PackedByteArray()
	px.resize(total)
	for y in range(SPRITE_PIXEL_SIZE):
		for x in range(SPRITE_PIXEL_SIZE):
			px[y * SPRITE_PIXEL_SIZE + x] = int(image.get_pixel(x, y).a * 255.0)

	var pos := Vector2(
		randf_range(_sprite_half + 10.0, TABLE_WIDTH - _sprite_half - 10.0),
		randf_range(_sprite_half + 10.0, TABLE_HEIGHT - _sprite_half - 10.0)
	)
	var angle := randf() * TAU
	var vel := Vector2(cos(angle), sin(angle)) * randf_range(80.0, 160.0)

	var node: Node2D = _sprite_scene.instantiate()
	var sprite: Sprite2D = node.get_node("Sprite2D")
	sprite.texture = ImageTexture.create_from_image(image)
	sprite.scale = Vector2(_display_scale, _display_scale)
	sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	node.position = pos
	sprites_container.add_child(node)

	positions.append(pos)
	velocities.append(vel)
	pixels.append(px)
	display_nodes.append(node)


func _physics_process(delta: float) -> void:
	var n := positions.size()
	broad_checks = 0
	aabb_checks = 0
	narrow_checks = 0
	
	for i in range(n):
		positions[i] += velocities[i] * delta
		_bounce_walls(i)

	_qt_root = null
	if n > 1:
		var qt := _QuadNode.new(Rect2(0.0, 0.0, TABLE_WIDTH, TABLE_HEIGHT), 0)
		for i in range(n):
			qt.insert(i, _aabb(i))
		_qt_root = qt
		var pairs: Dictionary = {}
		qt.collect_pairs(pairs)

		for key in pairs:
			broad_checks += 1
			var i: int = key[0]
			var j: int = key[1]
			if _aabb(i).intersects(_aabb(j)):
				aabb_checks += 1
				narrow_checks += 1
				var col_pt := _pixel_collision(i, j)
				if col_pt.is_finite():
					_resolve(i, j, col_pt)

	for i in range(n):
		display_nodes[i].position = positions[i]

	hud_label.text = "Sprites: %d  |  Quadtree pairs: %d  |  AABB hits: %d  |  Pixel checks: %d" % [
		n, broad_checks, aabb_checks, narrow_checks
	]
	queue_redraw()


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("help"):
		help_overlay.visible = not help_overlay.visible
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("show stats"):
		hud_label.visible = not hud_label.visible
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
	if event.is_action_pressed("zoom in"):
		_set_display_scale(_display_scale * 2)
		get_viewport().set_input_as_handled()
		return
	if event.is_action_pressed("zoom out"):
		@warning_ignore("integer_division")
		_set_display_scale(_display_scale / 2)
		get_viewport().set_input_as_handled()
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		var click := get_local_mouse_position()
		if not Rect2(0.0, 0.0, TABLE_WIDTH, TABLE_HEIGHT).has_point(click):
			return
		for i in range(positions.size()):
			if _aabb(i).has_point(click):
				var angle := randf() * TAU
				velocities[i] = Vector2(cos(angle), sin(angle)) * IMPULSE_SPEED
				get_viewport().set_input_as_handled()
				break


func _draw() -> void:
	if _show_bounding_boxes:
		for i in range(positions.size()):
			draw_rect(_aabb(i), Color(1.0, 1.0, 0.0, 0.8), false, 1.0)
	if _show_quadtree and _qt_root != null:
		_draw_qt_node(_qt_root)


func _draw_qt_node(node) -> void:
	draw_rect(node.bounds, Color(0.0, 0.8, 1.0, 0.4), false, 1.0)
	for child in node.children:
		_draw_qt_node(child)


func _aabb(i: int) -> Rect2:
	return Rect2(
		positions[i] - Vector2(_sprite_half, _sprite_half),
		Vector2(_sprite_screen_size, _sprite_screen_size)
	)


func _bounce_walls(i: int) -> void:
	var pos := positions[i]
	var vel := velocities[i]
	if pos.x - _sprite_half < 0.0:
		pos.x = _sprite_half
		vel.x = absf(vel.x)
	elif pos.x + _sprite_half > TABLE_WIDTH:
		pos.x = TABLE_WIDTH - _sprite_half
		vel.x = -absf(vel.x)
	if pos.y - _sprite_half < 0.0:
		pos.y = _sprite_half
		vel.y = absf(vel.y)
	elif pos.y + _sprite_half > TABLE_HEIGHT:
		pos.y = TABLE_HEIGHT - _sprite_half
		vel.y = -absf(vel.y)
	positions[i] = pos
	velocities[i] = vel


func _pixel_collision(i: int, j: int) -> Vector2:
	# Returns the screen-space collision point, or Vector2.INF if no collision.
	var dx := (positions[j].x - positions[i].x) / float(_display_scale)
	var dy := (positions[j].y - positions[i].y) / float(_display_scale)
	var px_i := pixels[i]
	var px_j := pixels[j]
	var s := SPRITE_PIXEL_SIZE
	for pya in range(s):
		for pxa in range(s):
			if px_i[pya * s + pxa] == 0:
				continue
			# Candidate pixels of j whose screen rect overlaps pixel (pxa, pya) of i
			var pxb_f := float(pxa) - dx
			var pyb_f := float(pya) - dy
			for pxb in [int(floor(pxb_f)), int(ceil(pxb_f))]:
				if pxb < 0 or pxb >= s:
					continue
				for pyb in [int(floor(pyb_f)), int(ceil(pyb_f))]:
					if pyb < 0 or pyb >= s:
						continue
					if px_j[pyb * s + pxb] > 0:
						var half := s * 0.5
						var pt_i := positions[i] + Vector2(pxa + 0.5 - half, pya + 0.5 - half) * _display_scale
						var pt_j := positions[j] + Vector2(pxb + 0.5 - half, pyb + 0.5 - half) * _display_scale
						return (pt_i + pt_j) * 0.5
	return Vector2.INF


func _resolve(i: int, j: int, collision_pt: Vector2) -> void:
	var sep := positions[j] - collision_pt
	if sep.length_squared() < 0.0001:
		sep = positions[j] - positions[i]
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


class _QuadNode:
	const MAX_ITEMS := 4
	const MAX_DEPTH := 5

	var bounds: Rect2
	var depth: int
	var items: Array   # Array of [idx: int, aabb: Rect2]
	var children: Array  # Array of _QuadNode, empty when leaf

	func _init(b: Rect2, d: int) -> void:
		bounds = b
		depth = d
		items = []
		children = []

	func insert(idx: int, aabb: Rect2) -> void:
		if not bounds.intersects(aabb):
			return
		if children.is_empty():
			items.append([idx, aabb])
			if items.size() > MAX_ITEMS and depth < MAX_DEPTH:
				_subdivide()
		else:
			for child in children:
				child.insert(idx, aabb)

	func _subdivide() -> void:
		var half := bounds.size * 0.5
		children = [
			_QuadNode.new(Rect2(bounds.position, half), depth + 1),
			_QuadNode.new(Rect2(bounds.position + Vector2(half.x, 0.0), half), depth + 1),
			_QuadNode.new(Rect2(bounds.position + Vector2(0.0, half.y), half), depth + 1),
			_QuadNode.new(Rect2(bounds.position + half, half), depth + 1),
		]
		var old := items.duplicate()
		items = []
		for it in old:
			for child in children:
				child.insert(it[0], it[1])

	func collect_pairs(out: Dictionary) -> void:
		if children.is_empty():
			for a in range(items.size()):
				for b in range(a + 1, items.size()):
					var ia: int = items[a][0]
					var ib: int = items[b][0]
					if ia > ib:
						var t := ia
						ia = ib
						ib = t
					out[[ia, ib]] = true
		else:
			for child in children:
				child.collect_pairs(out)
