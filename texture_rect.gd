extends TextureRect

var image_size := Vector2i(16, 16)
var image: Image
var texture_buffer: ImageTexture
var is_drawing: bool = false
var brush_color: Color = Color.BLACK

func _ready():
	clear_image()

func clear_image():
	image = Image.create(image_size.x, image_size.y, false, Image.FORMAT_RGBA8)
	image.fill(Color.TRANSPARENT)

	texture_buffer = ImageTexture.create_from_image(image)
	self.texture = texture_buffer

func load_image(img: Image):
	image = img.duplicate()
	texture_buffer = ImageTexture.create_from_image(image)
	self.texture = texture_buffer

func _gui_input(event):
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			is_drawing = event.pressed
			if is_drawing:
				paint_at_mouse(event.position)
	elif event is InputEventMouseMotion and is_drawing:
		paint_at_mouse(event.position)

func paint_at_mouse(mouse_pos: Vector2):
	var scale_factor = Vector2(image_size) / size
	var pixel_pos = Vector2i(mouse_pos * scale_factor)
	
	image.set_pixel(pixel_pos.x, pixel_pos.y, $"../../ColorPickerButton".color)
	texture_buffer.update(image)
