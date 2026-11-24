#!/usr/bin/env python3
"""Host-only orchestration checks. Every subprocess and GPU command is mocked."""
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[5]
PAYLOAD = REPO / "tools/mxfp4_training/crater_remote_run.py"
ORIGINAL_STAGES = ["gpu_environment", "cuda_build", "native_baseline", "native_tiled",
                   "native_certified", "native_benchmark", "torch_integration",
                   "torch_benchmark", "smoke_cuda", "smoke_cuda_tiled"]


def simulate(name, *, nvcc=True, torch=True, failures=None, expire_after=None,
             launch_error=None, timeout_stage=None):
    failures = failures or {}
    spec = importlib.util.spec_from_file_location("payload_under_test", PAYLOAD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [0.0]
    started, kills, build_variants = [], [], []
    with tempfile.TemporaryDirectory(prefix="mxfp4-payload-host-") as temporary:
        base = Path(temporary)
        root = base / "run"
        root.mkdir()
        archive = base / "source.tar.gz"
        with tarfile.open(archive, "w:gz") as packed:
            content = b"Host simulation fixture; no CUDA sources or GPU execution.\n"
            member = tarfile.TarInfo("fixture.txt")
            member.size = len(content)
            packed.addfile(member, io.BytesIO(content))
        fake_nvcc = base / "cuda-12.9/bin/nvcc"
        fake_nvcc.parent.mkdir(parents=True)
        fake_nvcc.write_text("not executable; probe is mocked\n")

        class Process:
            def __init__(self, argv, *, cwd, env, stdout, stderr, start_new_session):
                self.name = Path(stdout.name).stem
                if self.name == launch_error:
                    raise FileNotFoundError("host-simulated missing executable")
                assert start_new_session
                self.argv, self.returncode, self.waits = argv, None, 0
                self.pid = 100000000 + len(started)
                started.append({"name": self.name, "argv": argv,
                                "framework_library": env.get("MXFP4_CUDA_LIBRARY"), "pid": self.pid})
                stdout.write((json.dumps({"host_simulation": True, "name": self.name}) + "\n").encode())
                if "--certified-variant" in argv:
                    variant = int(argv[argv.index("--certified-variant") + 1])
                    build_variants.append(variant)
                    build = Path(argv[argv.index("--output-dir") + 1])
                    build.mkdir()
                    (build / "compile-evidence.json").write_text(json.dumps({
                        "host_simulation": True, "certified_variant": variant,
                        "passed": not failures.get(self.name, 0), "gpu_executed": False}))
                    (build / "shared_library.log").write_text("mock compiler log\n")
                if "--output" in argv:
                    output = Path(argv[argv.index("--output") + 1])
                    output.write_text(json.dumps({"host_simulation": True, "status": "passed"}) + "\n")

            def wait(self, timeout=None):
                self.waits += 1
                if self.name == timeout_stage and self.waits == 1:
                    raise subprocess.TimeoutExpired(self.argv, timeout)
                self.returncode = (-15 if self.name == timeout_stage else failures.get(self.name, 0))
                if self.name == expire_after:
                    clock[0] = 2701.0
                return self.returncode

        def probe(argv, **_kwargs):
            if argv[-1] == "--version":
                return subprocess.CompletedProcess(argv, 0, "Cuda compilation tools, release 12.9, V12.9.86", "")
            assert argv[1] == "-c"
            return subprocess.CompletedProcess(argv, 0 if torch else 2, "host-simulated torch probe", "")

        original_glob = Path.glob
        with ExitStack() as stack:
            stack.enter_context(patch.object(module.sys, "argv", [str(PAYLOAD), str(archive), str(root)]))
            stack.enter_context(patch.object(module.subprocess, "Popen", Process))
            stack.enter_context(patch.object(module.subprocess, "run", probe))
            stack.enter_context(patch.object(module.shutil, "which", lambda tool:
                str(fake_nvcc) if tool == "nvcc" and nvcc else
                (sys.executable if tool in ("python", "python3") else None)))
            stack.enter_context(patch.object(Path, "glob", lambda path, pattern:
                iter(()) if path == Path("/usr/local") and pattern == "cuda*/bin/nvcc"
                else original_glob(path, pattern)))
            stack.enter_context(patch.object(module.time, "monotonic", lambda: clock[0]))
            stack.enter_context(patch.object(module.os, "killpg", lambda pid, sig: kills.append((pid, sig))))
            stack.enter_context(patch.object(module.signal, "signal", lambda *_args: None))
            stack.enter_context(redirect_stdout(io.StringIO()))
            exit_code = module.main()
        results = root / "results"
        report = json.loads((results / "completion.json").read_text())
        stages = {item["name"]: item for item in report["stages"]}
        assert len(stages) == len(report["stages"]), "stage names must remain unique"
        assert report["deadline_seconds"] == 2700
        expected = report["fused_comparison"]["stage_order"]
        assert len(expected) == 2 + 4 * len(module.FUSED_VARIANTS) and all(label in stages for label in expected)
        assert all(stages[label]["status"] != "running" for label in stages)
        for item in report["stages"]:
            if item["status"] == "blocked":
                assert item["executed"] is False and item["gpu_executed"] is False
        for item in started:
            assert kills.count((item["pid"], signal.SIGTERM)) == 1
            assert kills.count((item["pid"], signal.SIGKILL)) == 1
            if item["name"] in expected:
                assert item["framework_library"] == str(root / "build/libmxfp4_sm70.so")
            if "benchmark_" in item["name"] and item["name"] in expected:
                argv = item["argv"]
                size = item["name"].rsplit("_", 1)[1]
                required = {"--m": size, "--n": size, "--k": size,
                            "--mode": "random_block_scales", "--seed": "20261123",
                            "--iters": "10", "--rounds": "5", "--warmup": "3"}
                assert all(argv[argv.index(flag) + 1] == value for flag, value in required.items())
                assert "--quick" not in argv and "--reserve-mib" not in argv and "--max-mib" not in argv
            if item["name"].startswith("fused_") and item["name"].endswith("_correctness"):
                assert item["argv"][-3:] == ["--device", "0", "--correctness-only"]
        for variant in module.FUSED_VARIANTS:
            prefix = "fused_c" + str(variant)
            if prefix + "_build" in [item["name"] for item in started]:
                build_evidence = results / ("fused-c" + str(variant)) / "build/compile-evidence.json"
                assert json.loads(build_evidence.read_text())["certified_variant"] == variant
            for label in expected:
                if label.startswith(prefix) and stages[label]["executed"]:
                    assert stages[label]["stdout"].startswith("fused-c" + str(variant) + "/")
            if stages[prefix + "_correctness"]["status"] != "passed":
                assert all(stages[prefix + "_benchmark_" + str(size)]["status"] == "blocked"
                           for size in (2048, 4096))
        if name == "all_pass":
            assert list(stages) == ORIGINAL_STAGES + expected
            assert report["status"] == "passed" and exit_code == 0
            assert build_variants == [2, *module.FUSED_VARIANTS]
        elif name == "no_toolchain":
            assert report["status"] == "blocked" and exit_code == 2
            assert all(stages[label]["status"] == "blocked" for label in expected)
        elif name == "no_torch":
            assert report["status"] == "blocked" and exit_code == 2
            assert all(stages[label]["status"] == "passed" for label in expected)
        elif name == "deadline_after_original_stages":
            assert report["status"] == "blocked" and exit_code == 2
            assert [item["name"] for item in started] == ORIGINAL_STAGES
            assert all(stages[label]["status"] == "blocked" for label in expected)
        elif name == "deadline_after_candidate_build":
            assert report["status"] == "blocked" and exit_code == 2
            assert started[-1]["name"] == "fused_c3_build"
        else:
            assert report["status"] == "failed" and exit_code == 1
        assert (root / "results.tar.gz").is_file()
        return {"scenario": name, "assertions_passed": True, "payload_status": report["status"],
                "payload_exit_code": exit_code, "mock_processes_started": len(started),
                "cleanup_signals_recorded": len(kills), "compiled_variants": build_variants,
                "stage_statuses": {label: stage["status"] for label, stage in stages.items()}}


def main():
    cases = [simulate("all_pass"), simulate("no_toolchain", nvcc=False),
             simulate("no_torch", torch=False),
             simulate("native_correctness_failed", failures={"native_certified": 1}),
             simulate("candidate_build_failed", failures={"fused_c3_build": 1}),
             simulate("candidate_correctness_failed", failures={"fused_c4_correctness": 1}),
             simulate("candidate_correctness_timeout", timeout_stage="fused_c4_correctness"),
             simulate("deadline_after_original_stages", expire_after="smoke_cuda_tiled"),
             simulate("deadline_after_candidate_build", expire_after="fused_c3_build"),
             simulate("missing_initial_executable", launch_error="gpu_environment")]
    result = {"schema_version": 1, "evidence_type": "host_only_payload_orchestration_simulation",
              "gpu_executed": False, "all_subprocesses_mocked": True,
              "passed": all(case["assertions_passed"] for case in cases),
              "payload_sha256": hashlib.sha256(PAYLOAD.read_bytes()).hexdigest(),
              "simulation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "cases": cases}
    output = Path(__file__).with_name("payload-host-simulation.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "gpu_executed": False,
                      "mocked_scenarios": len(cases), "output": str(output)}))


if __name__ == "__main__":
    main()
