from __future__ import annotations

import importlib.util
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch


MODULE_PATH = Path(__file__).with_name("bench_h3.py")
SPEC = importlib.util.spec_from_file_location("bench_h3", MODULE_PATH)
bench_h3 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_h3)


class HistoryTest(unittest.TestCase):
    def test_list_shaped_history_prompt_has_no_node_errors(self):
        self.assertTrue(hasattr(bench_h3, "extract_node_errors"))
        history = {"prompt": ["id", 0, {}, {}, []]}
        self.assertEqual(bench_h3.extract_node_errors(history), {})

    def test_execution_times_use_history_timestamps(self):
        self.assertTrue(hasattr(bench_h3, "execution_times"))
        history = {
            "status": {
                "messages": [
                    ["execution_start", {"timestamp": 1_000}],
                    ["execution_success", {"timestamp": 3_500}],
                ]
            }
        }
        self.assertEqual(bench_h3.execution_times(history), (1.0, 3.5))


class ArtifactTest(unittest.TestCase):
    def test_workflow_hash_is_content_addressed(self):
        self.assertTrue(hasattr(bench_h3, "_sha256_file"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.json"
            path.write_bytes(b"abc")
            self.assertEqual(
                bench_h3._sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223"
                "b00361a396177a9cb410ff61f20015ad",
            )

    def test_json_artifact_replaces_atomically(self):
        self.assertTrue(hasattr(bench_h3, "_write_json_atomic"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            path.write_text("{\"old\": true}\\n")
            bench_h3._write_json_atomic(path, {"new": [1, 2, 3]})
            self.assertEqual(
                path.read_text(), '{\n  "new": [\n    1,\n    2,\n    3\n  ]\n}\n'
            )
            self.assertFalse(path.with_name("result.json.tmp").exists())


class TelemetryTest(unittest.TestCase):
    def test_device_free_vram_wins_when_torch_telemetry_is_zero(self):
        payload = {
            "devices": [{"vram_free": 31_000, "torch_vram_free": 0}]
        }
        self.assertEqual(bench_h3._free_vram(payload), 31_000)

    def test_minimum_vram_ignores_missing_samples(self):
        self.assertTrue(hasattr(bench_h3, "_minimum_vram"))
        self.assertEqual(bench_h3._minimum_vram(20_000, None), 20_000)
        self.assertEqual(bench_h3._minimum_vram(None, 15_000), 15_000)

    def test_preflight_rejects_low_or_unknown_free_vram(self):
        self.assertTrue(hasattr(bench_h3, "require_free_vram"))
        low = {"devices": [{"vram_free": 10 * 1024**3}]}
        with patch.object(bench_h3, "api_json", return_value=low):
            with self.assertRaises(RuntimeError):
                bench_h3.require_free_vram(
                    "http://127.0.0.1:8188", 30 * 1024**3
                )
        with patch.object(bench_h3, "api_json", return_value={}):
            with self.assertRaises(RuntimeError):
                bench_h3.require_free_vram(
                    "http://127.0.0.1:8188", 30 * 1024**3
                )


class QueueSafetyTest(unittest.TestCase):
    def test_queue_error_is_not_treated_as_idle(self):
        with patch.object(bench_h3, "api_json", side_effect=OSError("down")):
            with self.assertRaises(OSError):
                bench_h3.queue_state("http://127.0.0.1:8188")


    def test_malformed_queue_payload_is_not_treated_as_idle(self):
        with patch.object(bench_h3, "api_json", return_value={}):
            with self.assertRaises(RuntimeError):
                bench_h3.queue_state("http://127.0.0.1:8188")

    def test_queue_and_history_waits_have_deadlines(self):
        self.assertIn(
            "timeout", inspect.signature(bench_h3.wait_for_idle).parameters
        )
        self.assertIn("timeout", inspect.signature(bench_h3.one_run).parameters)

    def test_timeout_cancel_targets_only_its_prompt(self):
        self.assertTrue(hasattr(bench_h3, "cancel_prompt"))
        empty = {"queue_running": [], "queue_pending": []}
        with (
            patch.object(
                bench_h3,
                "api_json",
                side_effect=[{"cancelled": True}, {"deleted": ["prompt-1"]}],
            ) as api,
            patch.object(bench_h3, "queue_state", return_value=empty),
        ):
            self.assertTrue(
                bench_h3.cancel_prompt(
                    "http://127.0.0.1:8188",
                    "prompt-1",
                    poll=0.01,
                    timeout=1.0,
                )
            )
        self.assertEqual(
            api.call_args_list,
            [
                call(
                    "http://127.0.0.1:8188",
                    "/interrupt",
                    {"prompt_id": "prompt-1"},
                    request_timeout=1.0,
                ),
                call(
                    "http://127.0.0.1:8188",
                    "/queue",
                    {"delete": ["prompt-1"]},
                    request_timeout=1.0,
                ),
            ],
        )


class RunStatusTest(unittest.TestCase):
    def test_only_completed_success_allows_next_case(self):
        self.assertTrue(hasattr(bench_h3, "run_succeeded"))
        self.assertTrue(
            bench_h3.run_succeeded(
                {"status": "success", "completed": True}
            )
        )
        self.assertFalse(
            bench_h3.run_succeeded(
                {"status": "success", "completed": False}
            )
        )
        self.assertFalse(
            bench_h3.run_succeeded(
                {"status": "timeout", "completed": False}
            )
        )


class MatrixTest(unittest.TestCase):
    def test_anchor_is_one_864x480_two_second_case(self):
        anchor = bench_h3.cases("anchor", steps=20, repeat=1)
        self.assertEqual(len(anchor), 1)
        self.assertEqual(
            (anchor[0]["width"], anchor[0]["height"], anchor[0]["frames"]),
            (864, 480, 56),
        )
        self.assertEqual(anchor[0]["steps"], 20)

    def test_sweeps_cover_requested_resolution_and_duration_ranges(self):
        resolution = bench_h3.cases("resolution", steps=1, repeat=1)
        duration = bench_h3.cases("duration", steps=1, repeat=1)
        self.assertEqual([row["target_mp"] for row in resolution], [
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ])
        self.assertEqual(
            [row["requested_seconds"] for row in duration], list(range(1, 16))
        )

    def test_aspect_sweep_covers_common_landscape_square_and_portrait_formats(self):
        aspect = bench_h3.cases("aspect", steps=1, repeat=1)
        self.assertEqual(
            [row["aspect_ratio"] for row in aspect],
            ["16:9", "4:3", "1:1", "3:4", "9:16"],
        )
        self.assertEqual(
            [(row["width"], row["height"]) for row in aspect],
            [(864, 480), (736, 544), (640, 640), (544, 736), (480, 864)],
        )
        self.assertTrue(all(row["target_mp"] == 0.4 for row in aspect))
        self.assertTrue(all(row["frames"] == 56 for row in aspect))

    def test_all_matrix_includes_resolution_duration_aspect_and_anchor(self):
        all_cases = bench_h3.cases("all", steps=1, repeat=1)
        self.assertEqual(
            {row["kind"] for row in all_cases},
            {"resolution", "duration", "aspect", "anchor"},
        )
        self.assertEqual(len(all_cases), 31)

    def test_zero_step_or_repeat_matrix_is_rejected(self):
        with self.assertRaises(ValueError):
            bench_h3.cases("anchor", steps=0, repeat=1)
        with self.assertRaises(ValueError):
            bench_h3.cases("anchor", steps=1, repeat=0)


if __name__ == "__main__":
    unittest.main()
