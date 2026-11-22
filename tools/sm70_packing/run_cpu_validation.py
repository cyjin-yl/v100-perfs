#!/usr/bin/env python3
"""Replay the CPU evidence in an isolated directory; never mutate source evidence.

Default requirements: Python 3.10+, NumPy, and a C++17 compiler. No GPU, CUDA,
network, Octave, or simulator binary is used by default. An architecture-log
consistency check is explicitly not a new architecture simulation.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def find_study_root() -> Path | None:
    for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
        for candidate in [parent, parent / "evidence/benchmarks/sm70-packing/study"]:
            if (candidate / "math/packing_exhaustive.py").is_file() and (candidate / "simulation/numerics/v100_model.py").is_file():
                return candidate
    return None


class Replay:
    def __init__(self, study: Path, output: Path, args: argparse.Namespace):
        self.study, self.output, self.args = study, output, args
        self.stage = output / "isolated-study"
        self.stage.mkdir()
        self.logs = output / "logs"
        self.logs.mkdir()
        self.protected: dict[str, str] = {}
        self.records: list[dict] = []
        self.environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONOPTIMIZE="0")

    def copy(self, relative: str) -> Path:
        src, dst = self.study / relative, self.stage / relative
        require(src.is_file(), f"Missing required evidence input: {relative}")
        self.protected[relative] = digest(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return dst

    def copy_tree(self, relative: str) -> None:
        directory = self.study / relative
        require(directory.is_dir(), f"Missing required evidence directory: {relative}")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                self.copy(str(path.relative_to(self.study)))

    def protect(self, relative: str) -> None:
        path = self.study / relative
        if path.is_file():
            self.protected[relative] = digest(path)

    def load(self, relative: str):
        return json.loads((self.stage / relative).read_text())

    def command(self, name: str, argv: list[str], *, cwd: Path | None = None,
                timeout: int = 300, env: dict | None = None) -> dict:
        log = self.logs / f"{name}.log"
        start = time.monotonic()
        with log.open("w") as handle:
            result = subprocess.run(argv, cwd=cwd or self.stage, env=env or self.environment,
                                    stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
        record = {"name": name, "argv": argv, "log": str(log.relative_to(self.output)),
                  "returncode": result.returncode, "wall_seconds": time.monotonic() - start,
                  "log_sha256": digest(log)}
        self.records.append(record)
        require(result.returncode == 0, f"{name} failed with exit {result.returncode}; see {log}")
        return record

    def python(self, name: str, relative: str, *extra: str) -> dict:
        return self.command(name, [sys.executable, str(self.stage / relative), *extra])

    def validate_originals_unchanged(self) -> None:
        changed = [p for p, expected in self.protected.items()
                   if not (self.study / p).is_file() or digest(self.study / p) != expected]
        require(not changed, f"Source evidence changed during replay: {changed}")

    def extra_checks(self) -> int:
        """Run explicit round-three checks; arbitrary missing checks never pass.

        Manifest entries declare immutable input files/directories, argv with
        {python}/{stage}/{output} placeholders, and exact JSON assertions.
        These are repository-owned test commands, not downloaded plugins.
        """
        path = self.args.extra_checks_manifest
        if path is None:
            candidates = [self.study / "simulation/round3/review/ci-checks.json",
                          self.study / "simulation/round3/ci-checks.json"]
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            require(not self.args.require_round3, "Round-three checks are required; provide --extra-checks-manifest")
            return 0
        path = Path(path).resolve()
        if path.is_relative_to(self.study):
            self.protect(str(path.relative_to(self.study)))
        manifest = json.loads(path.read_text())
        require(manifest.get("schema_version") == 1, "Unsupported extra-checks manifest schema")
        checks = manifest.get("checks", [])
        require(bool(checks), "Extra-checks manifest has no checks")
        for check in checks:
            name = check["name"]
            require(name.replace("-", "").replace("_", "").isalnum(), f"Invalid check name: {name}")
            for relative in check.get("files", []):
                require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                        f"Check input must be study-relative: {relative}")
                self.copy(relative)
            for relative in check.get("directories", []):
                require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                        f"Check directory must be study-relative: {relative}")
                self.copy_tree(relative)
            replacements = {"python": sys.executable, "stage": str(self.stage), "output": str(self.output)}
            argv = [argument.format(**replacements) for argument in check["argv"]]
            record = self.command(name, argv, timeout=int(check.get("timeout_seconds", 300)))
            data = self.load(check["result_json"])
            assertions = check.get("assertions", [])
            require(bool(assertions), f"{name} does not define any result assertions")
            for test in assertions:
                value = data
                for key in test["path"]:
                    value = value[key]
                require(any(key in test for key in ["equals", "minimum", "maximum", "length"]),
                        f"{name}: result assertion has no predicate")
                if "equals" in test:
                    require(value == test["equals"], f"{name}: {test['path']} expected {test['equals']!r}, got {value!r}")
                if "minimum" in test:
                    require(value >= test["minimum"], f"{name}: {test['path']} is below minimum")
                if "maximum" in test:
                    require(value <= test["maximum"], f"{name}: {test['path']} exceeds maximum")
                if "length" in test:
                    require(len(value) == test["length"], f"{name}: {test['path']} has wrong length")
            record.update(scope=check["scope"], passed=True, result_json=check["result_json"],
                          manifest_sha256=digest(path))
        return len(checks)

    def execute(self) -> dict:
        # Historical records are protected but are never used to manufacture
        # the new numerical test output.
        for p in ["math/packing_results.json", "simulation/numerics/numerical_results.json",
                  "simulation/numerics/original_model_cases.json", "simulation/numerics/fixture_comparison.csv",
                  "simulation/numerics/original_octave_results.json", "simulation/search/structural-search-results.json",
                  "simulation/cost/magic-decode-validation.json", "simulation/cost/static-cost-results.json",
                  "simulation/architecture/synthetic-results.json", "simulation/architecture/official-lud64-results.json",
                  "simulation/round3/math/round3_math_results.json", "simulation/round3/cuda/cpu-fragment-validation.json",
                  "simulation/round3/cuda/fragment-map.json", "simulation/round3/cuda/compile-evidence.json",
                  "simulation/round3/middle_decode/cpu-decoder-validation.json", "simulation/round3/middle_decode/compile-evidence.json",
                  "simulation/round3/review/preparation-review-results.json", "simulation/round3/review/layout-contract-results.json"]:
            self.protect(p)

        self.copy("math/packing_exhaustive.py")
        r = self.python("exact-integer-and-ieee-model", "math/packing_exhaustive.py")
        data = self.load("math/packing_results.json")
        require(not data["metadata"]["gpu_executed"], "CPU result unexpectedly claims GPU execution")
        require(data["narrow_e2m1_exact_k16"]["wrong_output_pairs"] == 0, "Narrow proof regression")
        for key in ["two_sided_ternary_exact_k16", "two_sided_ternary_exact_k32"]:
            require(data[key]["wrong_coefficient_triples"] == 0, f"{key} proof regression")
        collision = data["full_e2m1_k2_lookup_collision"]
        require(collision["case_a"]["fp32_accumulator"] == collision["case_b"]["fp32_accumulator"]
                and collision["case_a"]["target_q_dots"] != collision["case_b"]["target_q_dots"],
                "The expected counterexample disappeared")
        r.update(scope="CPU integer proof boxes, scalar IEEE model, and negative controls", passed=True)

        for p in ["run_numerical_study.py", "v100_model.py", "run_original_octave.m"]:
            self.copy("simulation/numerics/" + p)
        self.copy_tree("simulation/numerics/upstream-snapshot")
        r = self.python("published-v100-numerical-model", "simulation/numerics/run_numerical_study.py",
                        "--random-cases", str(self.args.random_cases), "--upstream",
                        str(self.stage / "simulation/numerics/upstream-snapshot"))
        data = self.load("simulation/numerics/numerical_results.json")
        fixture = data["upstream_hardware_fixture_validation"]
        require(fixture["fixture_count"] == 5000 and fixture["mismatches"]["ported_v100"] == 0,
                "Published V100 fixture validation failed")
        require(len(data["packing_schemes"]) == 6, "Unexpected numerical scheme coverage")
        require(all(x["v100_decode_failures"] == x["packed_value_vs_exact_integer_mismatches"]
                    == x["bfma_groups_that_discarded_alignment_bits"] == 0 for x in data["packing_schemes"]),
                "A safe-domain V100 numerical-model check failed")
        r.update(scope="Fresh CPU numerical-model execution against published hardware fixtures; no new GPU run",
                 passed=True, published_fixture_count=5000,
                 packing_cases=sum(x["total_cases"] for x in data["packing_schemes"]))

        if self.args.octave_bin:
            env = dict(self.environment, V100_MODEL_UPSTREAM=str(self.stage / "simulation/numerics/upstream-snapshot"))
            r = self.command("original-octave-model", [self.args.octave_bin, "--quiet", "--no-gui",
                             str(self.stage / "simulation/numerics/run_original_octave.m")], timeout=1800, env=env)
            data = self.load("simulation/numerics/original_octave_results.json")
            require(data["fixture_bit_mismatches"] == data["packing_and_control_bit_mismatches"] == 0,
                    "Original Octave model mismatch")
            r.update(scope="Fresh execution of unmodified upstream model in Octave", passed=True)

        self.copy("simulation/search/structural_search.py")
        r = self.python("linear-family-sufficient-bound-search", "simulation/search/structural_search.py")
        data = self.load("simulation/search/structural-search-results.json")
        require(data["candidates_checked"] == 21504 and all(x["sufficient_conditions_pass"]
                for x in data["selected"].values()), "Structural search regression")
        r.update(scope="Sufficient-bound search only; failed bounds are not impossibility proofs", passed=True,
                 candidates_checked=data["candidates_checked"])

        self.copy("simulation/cost/validate_magic_decode.py")
        r = self.python("magic-decoder", "simulation/cost/validate_magic_decode.py")
        data = self.load("simulation/cost/magic-decode-validation.json")
        require(all(v["low_errors"] == v["high_errors"] == 0 for v in data["validation"]),
                "Magic decoder regression")
        r.update(scope="CPU float32/int32 coefficient-box checks", passed=True,
                 coefficient_cases=sum(v["cases"] for v in data["validation"]))

        for p in ["analyze_cost.py", "resources.txt", "sm70-sass.txt", "packing_cost_kernels.cu",
                  "packing_cost_kernels.o", "m8n8k4_probe.cu", "m8n8k4_probe.o", "m8n8k4-sass.txt"]:
            self.copy("simulation/cost/" + p)
        r = self.python("recorded-sm70-static-cost", "simulation/cost/analyze_cost.py")
        data = self.load("simulation/cost/static-cost-results.json")
        kernels = {x["kernel"]: x for x in data["kernels"]}
        for name, expected_hmma in [("baseline_two_K32_w4", 64), ("two_sided_float_K32_w4", 32),
                                    ("two_sided_magic_K32_w4", 32)]:
            require(kernels[name]["opcode_counts"]["HMMA"] == expected_hmma, f"SASS count mismatch for {name}")
        require(len(kernels) == 9, "Unexpected static kernel coverage")
        r.update(scope="Reparse recorded SASS/resource evidence; no recompilation or GPU timing", passed=True,
                 kernels=len(kernels))

        # Fresh trace generation is deterministic. Reading old simulation logs
        # does not claim to have rerun the simulator.
        arch = "simulation/architecture/"
        self.copy(arch + "make_synthetic_traces.py")
        self.protect(arch + "synthetic/manifest.json")
        expected_traces = json.loads((self.study / arch / "synthetic/manifest.json").read_text())
        r = self.python("synthetic-trace-generator", arch + "make_synthetic_traces.py")
        generated = self.load(arch + "synthetic/manifest.json")
        require(generated == expected_traces, "Synthetic trace generator changed versus recorded manifest")
        for case in generated:
            trace = self.stage / arch / "synthetic" / case["name"] / "kernel-1.traceg"
            require(digest(trace) == case["sha256"], "Generated synthetic trace hash mismatch")
        r.update(scope="Fresh generation of artificial dependency graphs only; no simulation", passed=True,
                 traces=len(generated))

        self.copy(arch + "summarize_evidence.py")
        self.copy(arch + "logs/official-lud64-qv100.log")
        r = self.python("recorded-official-trace-log-parser", arch + "summarize_evidence.py")
        data = self.load(arch + "official-lud64-results.json")
        require(data["kernel_count"] == 10 and data["gpu_tot_sim_cycle"] == 127309
                and data["gpu_tot_sim_insn"] == 418884, "Recorded official trace log consistency failed")
        r.update(scope="Parse a recorded simulation log; not a new cycle simulation", passed=True)

        if self.args.simulate:
            for p in ["run_synthetic.py", "configs/qv100-upstream.config", "configs/qv100-upstream-trace.config",
                      "configs/latency8-whatif.config"]:
                self.copy(arch + p)
            self.copy_tree(arch + "official-lud64")
            if self.args.simulator_bin:
                executable = Path(self.args.simulator_bin).resolve()
                require(executable.is_file(), "Explicit simulator binary is missing")
                target = self.stage / arch / "build/accel-sim.out"
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(executable, target)
            else:
                target = self.copy(arch + "build/accel-sim.out")
            local_arch = self.stage / arch
            r = self.command("fresh-official-lud64-cycle-simulation", [str(target), "-trace",
                str(local_arch / "official-lud64/kernelslist.g"), "-config",
                str(local_arch / "configs/qv100-upstream.config"), "-config",
                str(local_arch / "configs/qv100-upstream-trace.config")], cwd=local_arch)
            fresh_log = (self.output / r["log"]).read_text()
            require("*** exit detected ***" in fresh_log, "Simulator did not complete")
            fresh_cycles = re.findall(r"^gpu_tot_sim_cycle\s*=\s*(\d+)", fresh_log, re.M)
            fresh_insns = re.findall(r"^gpu_tot_sim_insn\s*=\s*(\d+)", fresh_log, re.M)
            require(bool(fresh_cycles and fresh_insns), "Fresh simulator statistics are missing")
            require(int(fresh_cycles[-1]) == 127309 and int(fresh_insns[-1]) == 418884,
                    "Fresh QV100 replay differs from the pinned reference model")
            r.update(scope="Fresh cycle-model replay of published LUD trace; no packing kernel or numeric validation",
                     passed=True, binary_sha256=digest(target),
                     gpu_tot_sim_cycle=int(fresh_cycles[-1]), gpu_tot_sim_insn=int(fresh_insns[-1]))
            r = self.python("fresh-synthetic-cycle-simulations", arch + "run_synthetic.py")
            data = self.load(arch + "synthetic-results.json")
            require(len(data) == 10 and all(x["completed"] for x in data), "Synthetic model runs failed")
            r.update(scope="Fresh simulation of artificial dependency graphs; not kernel timing", passed=True)

        for p in ["ptx_accumulator_formula_check.cpp", "source-evidence/half.hpp"]:
            self.copy(arch + p)
        cxx = shutil.which(self.args.cxx)
        require(cxx is not None, f"C++ compiler not available: {self.args.cxx}")
        formula = self.stage / arch / "ptx_accumulator_formula_check"
        r = self.command("compile-isolated-half-formula", [cxx, "-O2", "-std=c++17",
                         "-I" + str(self.stage / arch / "source-evidence"),
                         str(self.stage / arch / "ptx_accumulator_formula_check.cpp"), "-o", str(formula)])
        r.update(scope="Compile isolated upstream half expression", passed=True)
        r = self.command("run-isolated-half-formula", [str(formula)])
        data = json.loads((self.output / r["log"]).read_text())
        require(data["K1_result"] == 4616 and data["K16_is_inf"], "Expected half-formula discrepancy disappeared")
        r.update(scope="Isolated arithmetic expression only, not a full PTX/GPU run", passed=True)

        extra_count = self.extra_checks()
        self.validate_originals_unchanged()
        return {"passed": True, "new_gpu_execution": False,
                "fresh_architecture_cycle_simulation": self.args.simulate,
                "fresh_original_octave_model": bool(self.args.octave_bin),
                "extra_round3_checks": extra_count,
                "random_cases_per_scheme": self.args.random_cases,
                "source_files_verified_unchanged": len(self.protected),
                "source_sha256": self.protected, "steps": self.records}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=find_study_root(),
                        help="Directory containing math/ and simulation/; supports the v100-perfs study layout")
    parser.add_argument("--output-dir", type=Path,
                        help="New directory for logs, isolated inputs, outputs and summary; must not already exist")
    parser.add_argument("--random-cases", type=int, default=64,
                        help="Random tests per scheme; all public fixtures and structured controls always run")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--octave-bin", help="Optional existing Octave executable; no installation is attempted")
    parser.add_argument("--simulate", action="store_true", help="Opt in to running the existing CPU architecture simulator")
    parser.add_argument("--simulator-bin", help="Compatible simulator binary for --simulate, replacing the recorded host binary")
    parser.add_argument("--extra-checks-manifest", type=Path,
                        help="Repository-owned JSON check list for new experiments; see the round3 review example")
    parser.add_argument("--require-round3", action="store_true",
                        help="Fail if no explicit round-three checks are configured")
    args = parser.parse_args()
    if args.study_root is None:
        parser.error("Cannot locate study inputs; pass --study-root")
    if args.random_cases < 0:
        parser.error("--random-cases must be nonnegative")
    if args.simulator_bin and not args.simulate:
        parser.error("--simulator-bin requires --simulate")
    study = args.study_root.resolve()
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        context = nullcontext(str(output))
    else:
        context = tempfile.TemporaryDirectory(prefix="v100-cpu-evidence-")
    with context as raw_output:
        output = Path(raw_output)
        replay = Replay(study, output, args)
        start = time.monotonic()
        try:
            summary = replay.execute()
        except Exception as error:
            try:
                replay.validate_originals_unchanged()
                unchanged = True
            except Exception:
                unchanged = False
            summary = {"passed": False, "error": str(error), "source_files_unchanged": unchanged,
                       "new_gpu_execution": False, "steps": replay.records}
        summary["runner_wall_seconds"] = time.monotonic() - start
        summary["runner_sha256"] = digest(Path(__file__))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: summary[k] for k in ["passed", "new_gpu_execution", "runner_wall_seconds"]}, indent=2))
        for row in summary.get("steps", []):
            print(json.dumps({k: row.get(k) for k in ["name", "passed", "scope", "wall_seconds"]}))
        if args.output_dir:
            print(f"Evidence replay output: {output}")
        if not summary["passed"]:
            print(summary["error"], file=sys.stderr)
        return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
