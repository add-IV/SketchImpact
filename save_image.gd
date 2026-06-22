extends Node2D

@onready var drawWindow = $MarginContainer/VBoxContainer/Control/TextureRect
@onready var grid = $MarginContainer/VBoxContainer/GridContainer

var save_count := 0
var billiard_table: Node
var saved_images: Array[Image] = []
var selected_slot: int = -1

func _ready():
	DirAccess.make_dir_recursive_absolute("user://sprites")
	billiard_table = get_parent().get_node_or_null("BilliardTable")
	if billiard_table:
		billiard_table.scenario_loaded.connect(_on_scenario_loaded)
	for i in grid.get_child_count():
		var slot: TextureRect = grid.get_child(i)
		slot.mouse_filter = Control.MOUSE_FILTER_STOP
		slot.gui_input.connect(_on_slot_gui_input.bind(i))

		var highlight := Panel.new()
		highlight.name = "Highlight"
		highlight.anchor_right = 1.0
		highlight.anchor_bottom = 1.0
		highlight.mouse_filter = Control.MOUSE_FILTER_IGNORE
		highlight.visible = false
		var style := StyleBoxFlat.new()
		style.bg_color = Color(0, 0, 0, 0)
		style.border_color = Color(1, 1, 0, 1)
		style.set_border_width_all(3)
		highlight.add_theme_stylebox_override("panel", style)
		slot.add_child(highlight)

	drawWindow.drew.connect(_clear_selection)

func _on_down_pressed():
	var image: Image = drawWindow.image.duplicate()

	var children := grid.get_child_count()
	for i in range(children - 1, 0, -1):
		var slot: TextureRect = grid.get_child(i)
		slot.texture = (grid.get_child(i - 1) as TextureRect).texture

	var first: TextureRect = grid.get_child(0)
	first.texture = ImageTexture.create_from_image(image)
	first.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	first.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED

	saved_images.push_front(image)
	if saved_images.size() > children:
		saved_images.pop_back()

	var path := "user://sprites/sprite_%04d.png" % save_count
	image.save_png(path)

	save_count += 1
	drawWindow.clear_image()
	_clear_selection()

func _on_up_pressed():
	if selected_slot == -1 or selected_slot >= saved_images.size():
		return
	drawWindow.load_image(saved_images[selected_slot])
	_clear_selection()

func _on_slot_gui_input(event: InputEvent, index: int):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		if index >= saved_images.size():
			return
		if selected_slot == index:
			_clear_selection()
		else:
			_set_selection(index)

func _set_selection(index: int):
	selected_slot = index
	for i in grid.get_child_count():
		grid.get_child(i).get_node("Highlight").visible = (i == index)

func _clear_selection():
	selected_slot = -1
	for i in grid.get_child_count():
		grid.get_child(i).get_node("Highlight").visible = false

func _on_scenario_loaded(images: Array[Image]) -> void:
	saved_images.clear()
	var children := grid.get_child_count()
	for i in range(children):
		(grid.get_child(i) as TextureRect).texture = null
	for i in range(min(images.size(), children)):
		saved_images.append(images[i])
		var slot := grid.get_child(i) as TextureRect
		slot.texture = ImageTexture.create_from_image(images[i])
		slot.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		slot.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	_clear_selection()

func _on_spawn_pressed():
	if billiard_table == null:
		return
	if selected_slot != -1 and selected_slot < saved_images.size():
		billiard_table.spawn_sprite(saved_images[selected_slot].duplicate())
	else:
		billiard_table.spawn_sprite(drawWindow.image.duplicate())
