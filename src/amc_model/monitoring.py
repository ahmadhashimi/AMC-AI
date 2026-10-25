"""Small durable training events; no dependency on a dashboard process."""
import json
import os
from pathlib import Path
import time


class RunMonitor:
    def __init__(self, folder, details):
        self.folder = Path(folder)
        self.details = details

    def emit(self, event, **values):
        record = {**self.details, "event": event, "time": time.time(), **values}
        encoded = json.dumps(record, allow_nan=False)
        with (self.folder / "metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(encoded + "\n")
        partial = self.folder / "status.json.partial"
        partial.write_text(encoded + "\n", encoding="utf-8")
        os.replace(partial, self.folder / "status.json")
