"""Tiny dependency-free serving utility for the TorchScript model.

Examples
--------
One local prediction using the built-in sklearn digits dataset:

    python serving.py --model model.pt --index 7

Start a minimal HTTP server:

    python serving.py --model model.pt --serve --port 8080

Then POST JSON:

    curl -X POST http://localhost:8080/predict \
      -H 'Content-Type: application/json' \
      -d '{"pixels": [0, 0, ... 64 total values ...]}'

Pixels may be:
  - 64 flat values, or
  - an 8x8 nested list.

Values can be either normalized [0, 1] or raw sklearn-digits values [0, 16].
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np
import torch


def load_model(model_path: str) -> torch.jit.ScriptModule:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    return model


def to_tensor(pixels: Any) -> torch.Tensor:
    array = np.asarray(pixels, dtype=np.float32)

    if array.size != 64:
        raise ValueError(
            f"Expected exactly 64 pixel values for an 8x8 image, got {array.size}."
        )

    array = array.reshape(8, 8)

    # Accept raw sklearn-digits intensity values [0, 16].
    if float(array.max()) > 1.0:
        array = array / 16.0

    return torch.from_numpy(array).reshape(1, 1, 8, 8)


def predict(model: torch.jit.ScriptModule, pixels: Any) -> dict[str, Any]:
    tensor = to_tensor(pixels)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = int(torch.argmax(probabilities).item())

    return {
        "prediction": predicted_class,
        "probabilities": [float(value) for value in probabilities.tolist()],
    }


def run_example(model: torch.jit.ScriptModule, index: int) -> None:
    from sklearn.datasets import load_digits

    digits = load_digits()
    index = index % len(digits.images)

    result = predict(model, digits.images[index])

    print(
        json.dumps(
            {
                "sample_index": index,
                "actual": int(digits.target[index]),
                **result,
            },
            indent=2,
        )
    )


def run_server(model: torch.jit.ScriptModule, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._send_json(404, {"error": "not found"})
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))

                if "pixels" not in payload:
                    raise ValueError("JSON body must contain a 'pixels' field.")

                result = predict(model, payload["pixels"])
                self._send_json(200, result)
            except (ValueError, json.JSONDecodeError) as exc:
                self._send_json(400, {"error": str(exc)})
            except Exception as exc:
                self._send_json(500, {"error": f"prediction failed: {exc}"})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[http] {self.address_string()} - {format % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving on http://{host}:{port}")
    print("Health: GET /health")
    print("Predict: POST /predict")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to TorchScript model.pt")
    parser.add_argument("--index", type=int, default=0, help="Example digit index")
    parser.add_argument("--serve", action="store_true", help="Start HTTP server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    model = load_model(args.model)

    if args.serve:
        run_server(model, args.host, args.port)
    else:
        run_example(model, args.index)


if __name__ == "__main__":
    main()
