"""Read-only local dashboard. Run in a second terminal beside training."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import shutil
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "dashboard"


class RunReader:
    def __init__(self, folder):
        self.folder = folder
        self.key = self.cached = None
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            metrics = self.folder / "metrics.jsonl"
            checkpoint = self.folder / "last.pt"
            key = tuple(p.stat().st_mtime_ns if p.exists() else None for p in (metrics, checkpoint))
            if key != self.key or self.cached is None:
                events = []
                if metrics.exists():
                    with metrics.open(encoding="utf-8") as f:
                        for line in f:
                            try:
                                event = json.loads(line)
                            except json.JSONDecodeError:
                                continue  # Writer may be in the middle of the last line.
                            if event["event"] == "resume":
                                events = [e for e in events if e["step"] <= event["step"]]
                            events.append(event)
                if events:
                    latest = events[-1]
                    mode = "metrics"
                elif checkpoint.exists():
                    import torch
                    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    ident = state["identity"]
                    tokens = state["token_metadata"]["splits"]
                    config = state["config"]
                    n = tokens["train"]["tokens"]
                    windows = (n - 1 + ident["block_size"] - 1) // ident["block_size"]
                    steps_per_epoch = (windows + ident["batch_size"] - 1) // ident["batch_size"]
                    latest = {"event":"legacy_checkpoint", "status":"checkpoint_only",
                              "step":state["steps"], "epoch_completed":state["epoch"],
                              "next_batch":state["next_batch"], "steps_per_epoch":steps_per_epoch,
                              "epochs":max(1, state["epoch"] + bool(state["next_batch"])),
                              "train_tokens":n, "val_tokens":tokens["val"]["tokens"],
                              "parameters":sum(v.numel() for k,v in state["model"].items() if k != "lm_head.weight"),
                              "device":state["device"], "batch_size":ident["batch_size"],
                              "block_size":ident["block_size"], "learning_rate":ident["lr"],
                              "tokenizer_vocab":config["vocab_size"],
                              "checkpoint_step":state["steps"], "checkpoint_time":checkpoint.stat().st_mtime,
                              "time":checkpoint.stat().st_mtime}
                    mode = "checkpoint"
                else:
                    latest, mode = None, "empty"
                self.cached = {"latest":latest, "events":events, "mode":mode}
                self.key = key
            return {**self.cached, "server_time":time.time(),
                    "disk_free_bytes":shutil.disk_usage(self.folder if self.folder.exists() else ROOT).free,
                    "run_folder":str(self.folder)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/wikipedia_v4_epochs")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    reader = RunReader(args.run_dir.resolve())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?", 1)[0]
            try:
                if route in ("/api/run", "/api/export"):
                    body = json.dumps(reader.read(), allow_nan=False).encode()
                    content_type = "application/json; charset=utf-8"
                elif route in ("/", "/style.css", "/app.js", "/assessment.js"):
                    name = "index.html" if route == "/" else route[1:]
                    body = (WEB / name).read_bytes()
                    content_type = {"index.html":"text/html", "style.css":"text/css", "app.js":"text/javascript", "assessment.js":"text/javascript"}[name] + "; charset=utf-8"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                if route == "/api/export":
                    self.send_header("Content-Disposition", 'attachment; filename="amc-training-metrics.json"')
                self.end_headers()
                self.wfile.write(body)
            except (OSError, ValueError, KeyError) as exc:
                self.send_error(503, "Run data unavailable; retry shortly")
                print("Dashboard read error:", exc, flush=True)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"AMC training dashboard: http://127.0.0.1:{args.port}", flush=True)
    print("Read-only. Close this server without stopping training.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
