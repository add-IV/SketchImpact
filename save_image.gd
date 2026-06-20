extends Node2D

@onready var drawWindow = $MarginContainer/VBoxContainer/Control/TextureRect
@onready var grid = $MarginContainer/VBoxContainer/GridContainer

var save_count := 0

func _ready():
	DirAccess.make_dir_recursive_absolute("user://sprites")

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

	var path := "user://sprites/sprite_%04d.png" % save_count
	image.save_png(path)

	save_count += 1
	drawWindow.clear_image()
