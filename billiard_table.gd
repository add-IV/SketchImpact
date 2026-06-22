extends Node2D

const TABLE_WIDTH := 820.0
const TABLE_HEIGHT := 620.0

# Adjust these two constants together when changing sprite resolution
const SPRITE_PIXEL_SIZE := 16   # native image resolution (e.g. 16 or 32)
const DISPLAY_SCALE := 3        # screen pixels per sprite pixel

const SPRITE_SCREEN_SIZE := SPRITE_PIXEL_SIZE * DISPLAY_SCALE
const SPRITE_HALF := SPRITE_SCREEN_SIZE / 2

const IMPULSE_SPEED := 200.0

# Struct-of-arrays: all sprite state lives here
var positions: Array[Vector2] = []
var velocities: Array[Vector2] = []
var pixels: Array[PackedByteArray] = []   # alpha channel, SPRITE_PIXEL_SIZE² bytes each
var display_nodes: Array[Node2D] = []

var broad_checks := 0   # unique pairs returned by quadtree
var aabb_checks := 0    # pairs that passed AABB overlap test
var narrow_checks := 0  # pairs that reached pixel-perfect check

var _show_bounding_boxes := false

@onready var hud_label: Label = $HUD
@onready var sprites_container: Node2D = $Sprites
@onready var help_overlay: Panel = $HelpOverlay

var _sprite_scene: PackedScene = preload("res://sprite_body.tscn")


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
		randf_range(SPRITE_HALF + 10.0, TABLE_WIDTH - SPRITE_HALF - 10.0),
		randf_range(SPRITE_HALF + 10.0, TABLE_HEIGHT - SPRITE_HALF - 10.0)
	)
	var angle := randf() * TAU
	var vel := Vector2(cos(angle), sin(angle)) * randf_range(80.0, 160.0)

	var node: Node2D = _sprite_scene.instantiate()
	var sprite: Sprite2D = node.get_node("Sprite2D")
	sprite.texture = ImageTexture.create_from_image(image)
	sprite.scale = Vector2(DISPLAY_SCALE, DISPLAY_SCALE)
	sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	node.position = pos
	sprites_container.add_child(node)

	positions.append(pos)
	velocities.append(vel)
	pixels.append(px)
	display_nodes.append(node)


func _process(delta: float) -> void:
	var n := positions.size()
	broad_checks = 0
	aabb_checks = 0
	narrow_checks = 0

	for i in range(n):
		positions[i] += velocities[i] * delta
		_bounce_walls(i)

	if n > 1:
		var qt := _QuadNode.new(Rect2(0.0, 0.0, TABLE_WIDTH, TABLE_HEIGHT), 0)
		for i in range(n):
			qt.insert(i, _aabb(i))
		var pairs: Dictionary = {}
		qt.collect_pairs(pairs)

		for key in pairs:
			broad_checks += 1
			var i: int = key[0]
			var j: int = key[1]
			if _aabb(i).intersects(_aabb(j)):
				aabb_checks += 1
				narrow_checks += 1
				if _pixel_collision(i, j):
					_resolve(i, j)

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
	if not _show_bounding_boxes:
		return
	for i in range(positions.size()):
		draw_rect(_aabb(i), Color(1.0, 1.0, 0.0, 0.8), false, 1.0)


func _aabb(i: int) -> Rect2:
	return Rect2(
		positions[i] - Vector2(SPRITE_HALF, SPRITE_HALF),
		Vector2(SPRITE_SCREEN_SIZE, SPRITE_SCREEN_SIZE)
	)


func _bounce_walls(i: int) -> void:
	var pos := positions[i]
	var vel := velocities[i]
	if pos.x - SPRITE_HALF < 0.0:
		pos.x = SPRITE_HALF
		vel.x = absf(vel.x)
	elif pos.x + SPRITE_HALF > TABLE_WIDTH:
		pos.x = TABLE_WIDTH - SPRITE_HALF
		vel.x = -absf(vel.x)
	if pos.y - SPRITE_HALF < 0.0:
		pos.y = SPRITE_HALF
		vel.y = absf(vel.y)
	elif pos.y + SPRITE_HALF > TABLE_HEIGHT:
		pos.y = TABLE_HEIGHT - SPRITE_HALF
		vel.y = -absf(vel.y)
	positions[i] = pos
	velocities[i] = vel


func _pixel_collision(i: int, j: int) -> bool:
	# Offset of j relative to i in sprite-pixel coordinates
	var dx := (positions[j].x - positions[i].x) / float(DISPLAY_SCALE)
	var dy := (positions[j].y - positions[i].y) / float(DISPLAY_SCALE)
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
						return true
	return false


func _resolve(i: int, j: int) -> void:
	var sep := positions[j] - positions[i]
	if sep.length_squared() < 0.0001:
		sep = Vector2(1.0, 0.0)
	else:
		sep = sep.normalized()
	# Elastic collision: swap velocity components along collision normal
	var vi_dot := velocities[i].dot(sep)
	var vj_dot := velocities[j].dot(sep)
	velocities[i] += sep * (vj_dot - vi_dot)
	velocities[j] += sep * (vi_dot - vj_dot)
	# Separate overlapping sprites
	var overlap := float(SPRITE_SCREEN_SIZE) - positions[i].distance_to(positions[j])
	if overlap > 0.0:
		positions[i] -= sep * overlap * 0.5
		positions[j] += sep * overlap * 0.5


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
