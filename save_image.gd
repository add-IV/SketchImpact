extends Node2D

@onready var drawWindow = $MarginContainer/VBoxContainer/Control/TextureRect
@onready var grid = $MarginContainer/VBoxContainer/GridContainer

var save_count := 0
var billiard_table: Node
var saved_images: Array[Image] = []

func _ready():
	DirAccess.make_dir_recursive_absolute("user://sprites")
	billiard_table = get_parent().get_node_or_null("BilliardTable")
	for i in grid.get_child_count():
		var slot: TextureRect = grid.get_child(i)
		slot.mouse_filter = Control.MOUSE_FILTER_STOP
		slot.gui_input.connect(_on_slot_gui_input.bind(i))

func _on_button_pressed():
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

func _on_slot_gui_input(event: InputEvent, index: int):
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT and event.pressed:
		if index < saved_images.size():
			drawWindow.load_image(saved_images[index])

func _on_spawn_pressed():
	if billiard_table == null:
		return
	billiard_table.spawn_sprite(drawWindow.image.duplicate())
