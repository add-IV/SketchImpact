"""
Persistent collision-classifier inference server for Godot.

Godot spawns this process once (via OS.execute_with_pipe) and keeps it alive
for the whole session, talking to it over stdin/stdout with one line-delimited
JSON message per request/response. This avoids reloading the ONNX model or
paying process-spawn cost every frame.

Protocol (one JSON object per line, newline-terminated, stdout flushed after
every response):

  -> {"cmd": "register", "idx": <int>, "alpha": [1024 ints, 0-255]}
  <- {"ok": true}

  -> {"cmd": "clear"}
  <- {"ok": true}

  -> {"cmd": "query", "id": <int>, "pairs": [[i, j, dx, dy], ...]}
  <- {"id": <int>, "maybe": [0/1, ...]}   # one flag per pair, same order

  -> {"cmd": "ping"}
  <- {"pong": true}

On startup, prints one line: {"ready": true, "provider": "<execution provider>"}
"""

import sys
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

SPRITE_SIZE = 32
MODEL_PATH = Path(__file__).resolve().parent.parent / "dataset" / "model.onnx"

# Operating point picked in dataset/design.md: threshold at ~99.9% recall on the
# validation set (eval.py), i.e. only skip the exact pixel check when the model
# is confident enough that fewer than 0.1% of true collisions are missed.
DEFAULT_THRESHOLD = -8.512

_xs = np.linspace(0, 1, SPRITE_SIZE, dtype=np.float32)
_ys = np.linspace(0, 1, SPRITE_SIZE, dtype=np.float32)
_X_GRID, _Y_GRID = np.meshgrid(_xs, _ys)  # (32, 32) each


def _build_session() -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers = []
    if "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    return ort.InferenceSession(str(MODEL_PATH), providers=providers)


def _make_batch(sprites: dict, pairs: list) -> np.ndarray:
    n = len(pairs)
    batch = np.empty((n, 6, SPRITE_SIZE, SPRITE_SIZE), dtype=np.float32)
    for k, (i, j, dx, dy) in enumerate(pairs):
        batch[k, 0] = sprites[i]
        batch[k, 1] = sprites[j]
        batch[k, 2] = dx / SPRITE_SIZE
        batch[k, 3] = dy / SPRITE_SIZE
    batch[:, 4] = _X_GRID
    batch[:, 5] = _Y_GRID
    return batch


def main() -> None:
    session = _build_session()
    input_name = session.get_inputs()[0].name
    provider = session.get_providers()[0]

    # Warm up so the first real query doesn't pay lazy-init cost mid-frame.
    session.run(None, {input_name: np.zeros((1, 6, SPRITE_SIZE, SPRITE_SIZE), dtype=np.float32)})

    sprites: dict[int, np.ndarray] = {}

    print(json.dumps({"ready": True, "provider": provider}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg["cmd"]

            if cmd == "register":
                alpha = np.array(msg["alpha"], dtype=np.float32).reshape(SPRITE_SIZE, SPRITE_SIZE) / 255.0
                sprites[msg["idx"]] = alpha
                print(json.dumps({"ok": True}), flush=True)

            elif cmd == "clear":
                sprites.clear()
                print(json.dumps({"ok": True}), flush=True)

            elif cmd == "query":
                pairs = msg["pairs"]
                threshold = msg.get("threshold", DEFAULT_THRESHOLD)
                if not pairs:
                    print(json.dumps({"id": msg["id"], "maybe": []}), flush=True)
                    continue
                batch = _make_batch(sprites, pairs)
                logits = session.run(None, {input_name: batch})[0]
                maybe = (logits > threshold).astype(int).tolist()
                print(json.dumps({"id": msg["id"], "maybe": maybe}), flush=True)

            elif cmd == "ping":
                print(json.dumps({"pong": True}), flush=True)

            else:
                print(json.dumps({"error": f"unknown cmd: {cmd}"}), flush=True)

        except Exception as e:
            print(json.dumps({"error": str(e)}), flush=True)


if __name__ == "__main__":
    main()
