extends Node

## Talks to inference/inference_server.py over stdin/stdout to gate the
## expensive pixel-perfect narrow-phase check with a CNN collision prediction.
## Fails open: if the server can't be started or stops responding, every
## query returns "maybe" so billiard_table.gd falls back to checking
## everything exactly, matching pre-CNN behaviour.

const UV_EXECUTABLE := "uv"
const INFERENCE_SCRIPT := "inference_server.py"

var enabled := true

var _pipe: FileAccess = null
var _pid := -1
var _ready_ok := false
var _next_query_id := 0


func _ready() -> void:
	_start_server()


func _start_server() -> void:
	var inference_dir := ProjectSettings.globalize_path("res://inference")
	var script_path := inference_dir.path_join(INFERENCE_SCRIPT)
	if not FileAccess.file_exists(script_path):
		push_warning("CollisionModel: %s not found — CNN filter disabled." % script_path)
		return

	var args := PackedStringArray(["run", "--project", inference_dir, "python", script_path])
	var result := OS.execute_with_pipe(UV_EXECUTABLE, args)
	if result.is_empty():
		push_warning("CollisionModel: failed to spawn 'uv' (is it installed and on PATH?) — CNN filter disabled.")
		return

	_pipe = result["stdio"]
	_pid = result["pid"]

	var line := _pipe.get_line()
	var parsed: Variant = JSON.parse_string(line)
	if parsed is Dictionary and parsed.get("ready", false):
		_ready_ok = true
		print("CollisionModel: ready (%s)" % str(parsed.get("provider", "?")))
	else:
		push_warning("CollisionModel: unexpected startup response: %s" % line)


## True once the server has replied to its startup handshake and hasn't
## dropped since. Independent of the `enabled` toggle — sprites must stay
## registered with the server even while filtering is toggled off, so
## re-enabling doesn't leave stale/missing entries on the Python side.
func is_server_connected() -> bool:
	return _ready_ok


func is_ready() -> bool:
	return _ready_ok and enabled


func _send(msg: Dictionary) -> Variant:
	if _pipe == null:
		return null
	_pipe.store_string(JSON.stringify(msg) + "\n")
	var line := _pipe.get_line()
	if line.is_empty():
		push_warning("CollisionModel: lost connection to inference server.")
		_ready_ok = false
		return null
	var response: Variant = JSON.parse_string(line)
	if response is Dictionary and response.has("error"):
		push_warning("CollisionModel: server error for %s: %s" % [msg.get("cmd", "?"), response["error"]])
	return response


func register_sprite(idx: int, alpha: PackedByteArray) -> void:
	if not is_server_connected():
		return
	_send({"cmd": "register", "idx": idx, "alpha": Array(alpha)})


func clear() -> void:
	if not is_server_connected():
		return
	_send({"cmd": "clear"})


## pairs: Array of [i, j, dx, dy]. Returns a same-length Array of bool —
## true means "the model isn't confident enough this misses, run the exact
## pixel check". Fails open (all true) if the server is unavailable.
func query_pairs(pairs: Array) -> Array:
	if pairs.is_empty():
		return []
	if not is_ready():
		return _all_true(pairs.size())

	var id := _next_query_id
	_next_query_id += 1
	var response: Variant = _send({"cmd": "query", "id": id, "pairs": pairs})
	if not (response is Dictionary) or not response.has("maybe"):
		if response is Dictionary and not response.has("error"):
			push_warning("CollisionModel: query response missing 'maybe': %s" % str(response))
		return _all_true(pairs.size())

	var maybe: Array = response["maybe"]
	var out: Array = []
	out.resize(maybe.size())
	for k in range(maybe.size()):
		out[k] = maybe[k] != 0
	return out


func _all_true(n: int) -> Array:
	var out: Array = []
	out.resize(n)
	out.fill(true)
	return out


func _exit_tree() -> void:
	if _pid >= 0:
		OS.kill(_pid)
