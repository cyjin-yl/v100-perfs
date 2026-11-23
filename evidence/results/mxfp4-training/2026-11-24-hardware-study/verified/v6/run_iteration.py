"""Bounded C6/C8 compile, full exact checks, and paired packed benchmarks.

Run from a newly extracted, immutable source directory containing source.tar.gz.
No service, clock, power, queue, or GPU-memory policy is changed by this driver.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback


parser = argparse.ArgumentParser()
parser.add_argument("--source-sha256", required=True)
parser.add_argument("--side", type=int, default=1536)
args = parser.parse_args()
root = Path(__file__).resolve().parent
archive_hash = hashlib.sha256((root / "source.tar.gz").read_bytes()).hexdigest()
assert archive_hash == args.source_sha256, "Frozen source archive differs"
assert args.side in (1408, 1536), "Only pre-budgeted bounded shapes are supported"
state_path = root / "driver-state.json"
assert not state_path.exists(), "Use a fresh directory, preserving earlier evidence"
started = time.monotonic()
deadline = started + 1200
record = {
    "schema_version": 2,
    "source_archive_sha256": archive_hash,
    "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "jobs": [],
    "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "completed": False,
    "gpu_payload_attempted": False,
    "full_training_executed": False,
    "side": args.side,
    "variants": [6, 8],
    "benchmark_order": [6, 8, 8, 6],
    "ordinary_memory_reserve_changed": False,
}
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")


def save():
    temp = state_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(record, indent=2) + "\n")
    temp.replace(state_path)


def execute(label, variant, command, outfile, timeout):
    remaining = deadline - time.monotonic()
    if remaining < 1:
        raise TimeoutError("The total driver deadline expired")
    start = time.monotonic()
    with (root / outfile).open("x") as so, (root / (outfile + ".stderr")).open("x") as se:
        try:
            result = subprocess.run(command, stdout=so, stderr=se, env=env,
                                    timeout=min(timeout, remaining))
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    return {"stage": label, "variant": variant, "returncode": rc,
            "seconds": time.monotonic() - start, "argv": command, "stdout": outfile}


def build(variant):
    command = ["python3", str(root / "tools/mxfp4_training/build_cuda.py"),
               "--nvcc", "/usr/local/cuda-12.9/bin/nvcc", "--output-dir",
               str(root / f"build-c{variant}"), "--with-certified",
               "--with-fused-benchmark", "--tiled-variant", "1",
               "--certified-variant", str(variant)]
    return execute("build", variant, command, f"build-c{variant}.stdout", 900)


logger = None
try:
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for item in pool.map(build, (6, 8)):
            record["jobs"].append(item)
            save()
    assert all(item["returncode"] == 0 for item in record["jobs"]), "CUDA build failed"
    for variant in (6, 8):
        manifest = json.loads((root / f"build-c{variant}/compile-evidence.json").read_text())
        assert manifest["passed"] and manifest["certified_variant"] == variant
        for name, expected in manifest["source_sha256"].items():
            actual = hashlib.sha256((root / "tools/mxfp4_training" / name).read_bytes()).hexdigest()
            assert actual == expected, f"Source changed after build: {name}"
    record["gpu_payload_attempted"] = True
    save()
    for variant in (6, 8):
        command = [str(root / f"build-c{variant}/check_mxfp4_certified"),
                   "--correctness-only", "--small-memory-correctness", "--reserve-mib", "48"]
        item = execute("correctness", variant, command, f"correctness-c{variant}.jsonl", 180)
        record["jobs"].append(item)
        save()
        assert item["returncode"] == 0, "Full exact correctness failed"
        messages = [json.loads(line) for line in (root / item["stdout"]).read_text().splitlines()
                    if line.startswith("{")]
        original = [message for message in messages if message.get("type") == "certified_summary"]
        assert len(original) == 1 and original[0].get("passed") is True
        if variant == 8:
            extra = [message for message in messages if message.get("type") == "c8_fixture_summary"]
            assert len(extra) == 1 and extra[0].get("passed") is True
            assert extra[0]["cases"] == 9 and extra[0]["values"] == 180224
    with (root / "background-gpu-samples.csv").open("x") as telemetry:
        logger = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.free,clocks.current.sm,power.draw",
             "--format=csv", "--loop-ms=200"], stdout=telemetry, stderr=subprocess.STDOUT)
        for seq, variant in enumerate((6, 8, 8, 6)):
            command = [str(root / f"build-c{variant}/bench_mxfp4_fused"),
                       "--m", str(args.side), "--n", str(args.side), "--k", str(args.side),
                       "--mode", "random_block_scales", "--seed", "20261123",
                       "--iters", "10", "--rounds", "5", "--warmup", "3",
                       "--max-mib", "12", "--reserve-mib", "48"]
            item = execute("benchmark_abba", variant, command,
                           f"abba-{seq:02d}-c{variant}-{args.side}.jsonl", 300)
            record["jobs"].append(item)
            save()
            assert item["returncode"] == 0, "Benchmark failed; inspect evidence before changing any shape"
    record["completed"] = True
except Exception as error:
    record["error"] = str(error)
    record["traceback"] = traceback.format_exc()
finally:
    if logger is not None:
        logger.terminate()
        try:
            logger.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.kill()
            logger.wait()
    record["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save()

