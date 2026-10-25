"""Temporary measurements only: no server or training processes."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from amc_model.monitoring import RunMonitor

spec = importlib.util.spec_from_file_location("dashboard", ROOT / "scripts/training_dashboard.py")
dashboard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashboard)


class MonitoringChecks(unittest.TestCase):
    def test_log_and_status_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            monitor = RunMonitor(folder, {"epochs": 1})
            monitor.emit("train", step=100, train_loss=5.2)
            status = json.loads((folder / "status.json").read_text())
            events = (folder / "metrics.jsonl").read_text().splitlines()
            self.assertEqual(status, json.loads(events[-1]))
            self.assertEqual(status["train_loss"], 5.2)
            monitor.emit("validation", step=1000, validation_loss=4.8)
            self.assertEqual(len((folder / "metrics.jsonl").read_text().splitlines()), 2)

    def test_resume_filters_rolled_back_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            monitor = RunMonitor(folder, {})
            for step in (100, 200, 300):
                monitor.emit("train", step=step, train_loss=5.0)
            monitor.emit("resume", step=200)
            monitor.emit("train", step=250, train_loss=4.9)
            with (folder / "metrics.jsonl").open("a") as f:
                f.write('{"incomplete":')
            result = dashboard.RunReader(folder).read()
            self.assertEqual([e["step"] for e in result["events"]], [100, 200, 200, 250])
            self.assertEqual(result["latest"]["train_loss"], 4.9)
            self.assertEqual(result["mode"], "metrics")

    def test_missing_run_has_no_invented_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = dashboard.RunReader(Path(tmp)).read()
            self.assertIsNone(result["latest"])
            self.assertEqual(result["events"], [])
            self.assertEqual(result["mode"], "empty")


if __name__ == "__main__":
    unittest.main()
