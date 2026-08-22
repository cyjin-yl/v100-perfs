from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import subprocess
from unittest import mock


MODULE_PATH = Path(__file__).parent / "scripts" / "fastllm_native_profile.py"
SPEC = importlib.util.spec_from_file_location("fastllm_native_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
native_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native_profile)


class NativeManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = native_profile.load_manifest()

    def test_six_profiles_have_unique_ports_and_absolute_argv(self):
        profiles = self.manifest["profiles"]
        self.assertEqual(len(profiles), 6)
        self.assertEqual(len({item["name"] for item in profiles}), 6)
        self.assertEqual(len({item["port"] for item in profiles}), 6)
        for item in profiles:
            self.assertTrue(Path(item["argv"][0]).is_absolute())
            self.assertEqual(item["argv"][0], self.manifest["binary"])
            self.assertEqual(
                native_profile._argv_value(item["argv"], "--path", item["name"]),
                item["model_path"],
            )

    def test_local_artifact_report_matches_declared_profiles(self):
        report = native_profile.artifact_report(self.manifest)
        self.assertTrue(report["valid"])
        self.assertTrue(report["profiles"]["thinkingcap-q4"]["size_match"])
        self.assertTrue(report["profiles"]["thinkingcap-q6"]["size_match"])
        self.assertTrue(report["profiles"]["heretic-q6"]["size_match"])
        self.assertTrue(report["profiles"]["heretic-q8"]["size_match"])
        self.assertTrue(report["profiles"]["thinkingcap-q8"]["accepted_missing"])
        self.assertTrue(report["profiles"]["heretic-q4"]["accepted_missing"])

    def test_manager_requires_external_proxy_mode(self):
        manager = native_profile.NativeProfileManager()
        with mock.patch.dict(native_profile.os.environ, {"FASTLLM_OWNED": "1"}, clear=False):
            with self.assertRaisesRegex(native_profile.NativeProfileError, "FASTLLM_OWNED"):
                manager._assert_external_proxy()

    def test_free_vram_parser_uses_lowest_gpu_value(self):
        result = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout="32012\n31744\n", stderr=""
        )
        with mock.patch.object(native_profile.subprocess, "run", return_value=result):
            self.assertEqual(native_profile.NativeProfileManager._free_vram_mib(), 31744)

    def test_preflight_rejects_below_minimum_free_vram(self):
        manager = native_profile.NativeProfileManager()
        profile = manager._profile("thinkingcap-q4")
        with mock.patch.object(manager, "_free_vram_mib", return_value=511):
            with self.assertRaisesRegex(native_profile.NativeProfileError, "pre-start VRAM gate"):
                manager._preflight(profile)

    def test_unknown_top_level_key_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["unexpected"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(native_profile.NativeProfileError):
                native_profile.load_manifest(path)

    def test_duplicate_port_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["profiles"][1]["port"] = broken["profiles"][0]["port"]
        broken["profiles"][1]["argv"][broken["profiles"][1]["argv"].index("--port") + 1] = str(
            broken["profiles"][0]["port"]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(native_profile.NativeProfileError):
                native_profile.load_manifest(path)

    def test_relative_executable_is_rejected(self):
        broken = copy.deepcopy(self.manifest)
        broken["profiles"][0]["argv"][0] = "./apiserver"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(native_profile.NativeProfileError):
                native_profile.load_manifest(path)


class FingerprintTest(unittest.TestCase):
    def test_argv_fingerprint_is_stable_and_order_sensitive(self):
        first = ["/bin/true", "--a", "1"]
        second = ["/bin/true", "--a", "2"]
        self.assertEqual(
            native_profile._argv_fingerprint(first),
            native_profile._argv_fingerprint(first),
        )
        self.assertNotEqual(
            native_profile._argv_fingerprint(first),
            native_profile._argv_fingerprint(second),
        )


if __name__ == "__main__":
    unittest.main()
