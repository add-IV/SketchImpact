extends Label

var billiard_table: Node


func _ready() -> void:
	billiard_table = get_tree().get_first_node_in_group("billiard_table")


func _process(_delta: float) -> void:
	if billiard_table == null:
		return
	text = _format_stats()


func _format_stats() -> String:
	var t: Node = billiard_table
	var eval_text := ""
	if t.eval_mode:
		var eval_total: int = t.eval_flagged_total + t.eval_skipped_total
		var fp_rate: float = 100.0 * t.false_positives / t.eval_flagged_total if t.eval_flagged_total > 0 else 0.0
		var fn_rate: float = 100.0 * t.false_negatives / t.eval_skipped_total if t.eval_skipped_total > 0 else 0.0
		var skip_rate: float = 100.0 * t.eval_skipped_total / eval_total if eval_total > 0 else 0.0
		eval_text = "\nFP: %d/%d (%.1f%%)\nFN: %d/%d (%.1f%%)\nSkip rate: %d/%d (%.1f%%)" % [
			t.false_positives, t.eval_flagged_total, fp_rate,
			t.false_negatives, t.eval_skipped_total, fn_rate,
			t.eval_skipped_total, eval_total, skip_rate
		]
	return "Sprites: %d\nQuadtree pairs: %d\nAABB hits: %d\nPixel checks: %d\nModel skips: %d%s%s" % [
		t.positions.size(), t.broad_checks, t.aabb_checks, t.narrow_checks, t.model_skips,
		"" if t.collision_model.is_ready() else "  (model off)",
		eval_text
	]


func _input(event: InputEvent) -> void:
	if event.is_action_pressed("show stats"):
		visible = not visible
		get_viewport().set_input_as_handled()
