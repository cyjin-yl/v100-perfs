#!/usr/bin/env python3
"""Execute the real harness host logic with CUDA declarations/snapshots stubbed.

No CUDA library, driver, remote host, or GPU is used. The full new host driver
is syntax-checked against the original host helpers after omitting the old
driver and device-only dequantization kernel. Parser, guard, JSON, and finite
baseline checks compile the actual current C++ fragments, not reimplementations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import traceback


def fragment(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth, end = 1, opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


def compile_cpp(path: Path, binary: Path, record: dict) -> None:
    argv = ["g++", "-std=c++17", "-O2", str(path), "-o", str(binary)]
    result = run(argv)
    record.setdefault("compilations", []).append({
        "argv": argv, "returncode": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr,
    })
    assert result.returncode == 0, result.stderr


def record_execution(binary: Path, args: list[str], record: dict, **kwargs):
    result = run([str(binary)] + args, **kwargs)
    observed = json.loads(result.stdout)
    record.setdefault("executions", []).append({
        "args": args, "test_environment": {
            k: v for k, v in kwargs.get("env", {}).items() if k.startswith("TEST_")
        }, "returncode": result.returncode, "stdout_json": observed,
        "stderr": result.stderr,
    })
    return result, observed


def check_profile(src: Path, temp: Path, result: dict) -> None:
    source = (src / "check_mxfp4.cu").read_text()
    wrapper = (src / "check_mxfp4_certified.cu").read_text()
    globals_ = source[source.index("int reserve_mib = 128;"):source.index("void checked(")]
    options = source[source.index("struct Options {"):source.index("Options parse(")]
    allowed_macro = next(line for line in wrapper.splitlines()
                         if line.startswith("#define MXFP4_CHECK_SMALL_MEMORY_ALLOWED "))
    env_start = source.index('    std::cout << "{\\\"type\\\":\\\"environment\\\"')
    env_end = source.index("    require_free_memory(65536);", env_start)
    environment = source[env_start:env_end]
    prelude = r'''
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
using cudaError_t=int; constexpr int cudaSuccess=0;
size_t fake_free=65208320; int queries=0;
int cudaMemGetInfo(size_t* free,size_t* total){++queries;*free=fake_free;*total=17179869184ull;return 0;}
const char* cudaGetErrorString(int){return "host-only stub";}
const char* mxfp4_cuda_backend(){return "host-only parser and guard test";}
'''
    main = r'''
int main(int argc,char** argv){try {
  const Options options=parse(argc,argv);
  if(const char* f=std::getenv("TEST_FREE"))fake_free=std::stoull(f);
  if(const char* b=std::getenv("TEST_BYTES"))require_free_memory(std::stoull(b));
  const struct {int major=7,minor=0;const char* name="host-only stub";} properties;
  bool actual_v100=false;int driver_version=0,runtime_version=0;
  size_t free_bytes=fake_free,total_bytes=17179869184ull;
''' + environment + r'''
  std::cerr<<queries;return 0;
}catch(const std::exception& e){std::cout<<"{\"error\":"<<json_string(e.what())<<"}\n";std::cerr<<queries;return 1;}}
'''
    body = (globals_ + fragment(source, "void checked(")
            + "\n#define CUDA_CHECK(expression) checked((expression), #expression)\n"
            + fragment(source, "std::string json_string(") + "\n"
            + fragment(source, "void require_free_memory(") + "\n"
            + options + fragment(source, "Options parse(") + "\n" + main)
    cases = [
        ([], True, 128, False), (["--reserve-mib", "64"], True, 64, False),
        (["--reserve-mib", "48"], False, None, None),
        (["--small-memory-correctness"], False, None, None),
        (["--small-memory-correctness", "--reserve-mib", "48"], None, 48, True),
        (["--reserve-mib", "48", "--small-memory-correctness"], None, 48, True),
        (["--small-memory-correctness", "--reserve-mib", "64"], None, 64, True),
        (["--reserve-mib", "47", "--small-memory-correctness"], False, None, None),
        (["--benchmark", "--correctness-only", "--small-memory-correctness", "--reserve-mib", "48"], False, None, None),
        (["--small-memory-correctness", "--reserve-mib", "48", "--benchmark"], False, None, None),
        (["--reserve-mib"], False, None, None),
        (["--reserve-mib", "-1"], False, None, None),
        (["--reserve-mib", "48x", "--small-memory-correctness"], False, None, None),
        (["--small-memory-correctness", "--reserve-mib", "48", "--iters", "0"], False, None, None),
        (["--wrong"], False, None, None),
    ]
    result["correctness_profile"] = records = []
    for label, variant, certified in [
            ("base", None, False), ("base_variant3", 3, False), ("c2", 2, True),
            ("f3", 3, True), ("f4", 4, True), ("f5", 5, True)]:
        record = {"configuration": label}
        records.append(record)
        defines = ("" if variant is None else f"#define MXFP4_CERTIFIED_VARIANT {variant}\n")
        defines += allowed_macro + "\n" if certified else ""
        path, binary = temp / (label + ".cpp"), temp / label
        path.write_text(prelude + defines + body)
        compile_cpp(path, binary, record)
        allowed = certified and variant >= 3
        for args, expected, reserve, small in cases:
            expected = allowed if expected is None else expected
            observed, data = record_execution(binary, args, record)
            assert (observed.returncode == 0) == expected, (label, args, data)
            if expected:
                assert data["reserve_mib"] == reserve
                assert data["memory_profile"] == ("small_memory_correctness" if small else "ordinary")
                assert data["max_explicit_allocation_bytes"] == (8388608 if small else None)
                assert data["gpu_executed"] is False
            assert observed.stderr == "0"
        if allowed:
            args = ["--small-memory-correctness", "--reserve-mib", "48"]
            guards = [(3014876, 65208320, True, 1), (8388608, 65208320, True, 1),
                      (8388609, 65208320, False, 0), (11338200, 65208320, False, 0),
                      (3014876, 50331648 + 3014876, True, 1),
                      (3014876, 50331648 + 3014876 - 1, False, 1),
                      (0, 50331648 - 1, False, 1)]
        else:
            args = ["--reserve-mib", "64"]
            guards = [(1, 65208320, False, 1), (11338200, (64 << 20) + 11338200, True, 1),
                      (11338200, (64 << 20) + 11338200 - 1, False, 1)]
        for size, free, expected, queries in guards:
            env = dict(os.environ, TEST_BYTES=str(size), TEST_FREE=str(free))
            observed, data = record_execution(binary, args, record, env=env)
            assert (observed.returncode == 0) == expected, (label, size, free, data)
            assert observed.stderr == str(queries)
        record["checks_passed"] = len(record["executions"])


def check_packed(src: Path, temp: Path, result: dict) -> None:
    base = (src / "bench_mxfp4.cu").read_text()
    packed = (src / "bench_mxfp4_fused.cu").read_text()
    core = (src / "mxfp4_gemm_certified.cu").read_text()
    helpers = base[:base.index("void run_case(const Input&")] + "\n} // namespace\n"
    start = helpers.index("__global__ void dequantize_both(")
    end = helpers.index("struct Options {", start)
    helpers = helpers[:start] + helpers[end:]
    cuda = r'''#pragma once
#include <cstddef>
using cudaError_t=int; using cudaStream_t=void*; using cudaEvent_t=void*;
constexpr int cudaSuccess=0,cudaStreamNonBlocking=1,cudaMemcpyDeviceToHost=2,cudaMemcpyHostToDevice=3;
struct cudaDeviceProp{int major,minor,multiProcessorCount,clockRate,memoryClockRate,memoryBusWidth,ECCEnabled;char name[256];};
#define __device__
#define __constant__
const char* cudaGetErrorString(int);
''' + "".join(f"int {name}(...);\n" for name in [
        "cudaMalloc", "cudaFree", "cudaStreamCreateWithFlags", "cudaStreamDestroy", "cudaEventCreate",
        "cudaEventDestroy", "cudaSetDevice", "cudaGetDeviceProperties", "cudaDeviceGetPCIBusId",
        "cudaDriverGetVersion", "cudaRuntimeGetVersion", "cudaMemGetInfo", "cudaMemcpyAsync",
        "cudaStreamSynchronize", "cudaMemcpy", "cudaMemsetAsync", "cudaEventRecord",
        "cudaEventSynchronize", "cudaEventElapsedTime", "cudaGetDevice"])
    (temp / "cuda_runtime_api.h").write_text(cuda)
    (temp / "cuda_runtime.h").write_text('#pragma once\n#include "cuda_runtime_api.h"\n')
    (temp / "cuda_fp16.h").write_text('#pragma once\n#include <cstdint>\nusing half=uint16_t;\n')
    (temp / "cublas_v2.h").write_text('''#pragma once
using cublasStatus_t=int;using cublasHandle_t=void*;
constexpr int CUBLAS_STATUS_SUCCESS=0,CUBLAS_TENSOR_OP_MATH=1;
int cublasCreate(...);int cublasSetStream(...);int cublasSetMathMode(...);int cublasDestroy(...);
''')
    (temp / "bench_helpers_host.hpp").write_text(helpers)
    syntax = temp / "packed_syntax.cpp"
    syntax.write_text(packed.replace('#include "bench_mxfp4.cu"', '#include "bench_helpers_host.hpp"'))
    result["packed_host_driver_syntax"] = records = []
    for variant in [3, 4, 5, 0, 2]:
        argv = ["g++", "-std=c++17", "-fsyntax-only", f"-DMXFP4_CERTIFIED_VARIANT={variant}",
                "-I" + str(temp), "-I" + str(src), str(syntax)]
        observed = run(argv)
        records.append({"variant": variant, "argv": argv, "returncode": observed.returncode,
                        "stdout": observed.stdout, "stderr": observed.stderr})
        if variant >= 3:
            assert observed.returncode == 0, observed.stderr
        else:
            assert observed.returncode != 0 and "requires a compiled certified variant" in observed.stderr
    result["packed_parser_and_guard"] = record = {}
    options = base[base.index("struct Options {"):base.index("Options parse(")]
    prelude = r'''
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <limits>
size_t fake_free=65208320;int queries=0;
int cudaMemGetInfo(size_t* f,size_t* t){++queries;*f=fake_free;*t=17179869184ull;return 0;}
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1;
constexpr bool kFused=true;constexpr size_t kBlasWorkspace=4u*1024u*1024u;
#define CUDA(x) do{if(x)throw std::runtime_error("mock CUDA error");}while(0)
#define CERTIFIED(x) CUDA(x)
constexpr int kPackedMaxMiB=12,kPackedMinimumReserveMiB=48;
'''
    # Workspace calculation is the actual pure-host C API implementation.
    workspace = (fragment(core, "struct RowData {") + ";\n"
                 + fragment(core, "struct DeviceMetadata {") + ";\n"
                 + fragment(core, "int validate_shape(") + "\n"
                 + fragment(core, 'extern "C" int mxfp4_certified_workspace_bytes(') + "\n")
    main = r'''
int main(int argc,char** argv){try{const auto o=parse_packed(argc,argv);const auto b=packed_budget(o.m,o.n,(o.k+31)/32*32);
if(const char* f=std::getenv("TEST_FREE"))fake_free=std::stoull(f);
if(std::getenv("TEST_GUARD"))packed_memory_guard(b,o,std::getenv("TEST_RESIDENT")!=nullptr,"test");
std::cout<<"{\"M\":"<<o.m<<",\"N\":"<<o.n<<",\"K\":"<<o.k<<",\"rounds\":"<<o.rounds<<",\"iters\":"<<o.iters<<",\"warmup\":"<<o.warmup<<",\"mode\":"<<quote(o.mode)<<",\"budget\":"<<b.json()<<"}\n";std::cerr<<queries;return 0;
}catch(const std::exception& e){std::cout<<"{\"error\":"<<quote(e.what())<<"}\n";std::cerr<<queries;return 1;}}
'''
    source = (prelude + workspace + options + fragment(base, "std::string quote(") + "\n"
              + fragment(packed, "struct PackedBudget {") + ";\n"
              + fragment(packed, "PackedBudget packed_budget(") + "\n"
              + fragment(packed, "struct PackedMemory {") + ";\n"
              + fragment(packed, "PackedMemory packed_memory_guard(") + "\n"
              + fragment(packed, "Options parse_packed(") + "\n" + main)
    path, binary = temp / "packed_parser.cpp", temp / "packed_parser"
    path.write_text(source)
    compile_cpp(path, binary, record)
    valid = ["--max-mib", "12", "--reserve-mib", "48"]
    cases = [([], False), (["--max-mib", "12"], False), (["--reserve-mib", "48"], False),
             (valid, True), (["--reserve-mib", "48", "--max-mib", "12"], True),
             (["--max-mib", "13", "--reserve-mib", "48"], False),
             (["--max-mib", "0", "--reserve-mib", "48"], False),
             (["--max-mib", "12", "--reserve-mib", "47"], False),
             (valid + ["--m", "17"], False),
             (valid + ["--m", "17", "--n", "19", "--k", "65"], True),
             (valid + ["--m", "65537", "--n", "1", "--k", "32"], False),
             (valid + ["--rounds", "2"], False), (valid + ["--iters", "0"], False),
             (valid + ["--warmup", "10001"], False), (valid + ["--mode", "unknown"], False),
             (valid + ["--correctness-only"], True)]
    for args, expected in cases:
        observed, data = record_execution(binary, args, record)
        assert (observed.returncode == 0) == expected, (args, data)
        assert observed.stderr == "0"
        if expected and "--m" not in args:
            assert tuple(data[k] for k in ["M", "N", "K", "rounds", "iters", "warmup", "mode"]) == (
                1536, 1536, 1536, 5, 10, 3, "random_block_scales")
            assert data["budget"] == {
                "packed_nibbles_and_scales": 2506752, "single_fp32_output": 9437184,
                "certified_workspace": 52296, "total_explicit_gpu_bytes": 11996232}
    for free, resident, expected in [
            (65208320, False, True), (50331648 + 11996232, False, True),
            (50331648 + 11996232 - 1, False, False),
            (50331648, True, True), (50331648 - 1, True, False)]:
        env = dict(os.environ, TEST_FREE=str(free), TEST_GUARD="1")
        if resident:
            env["TEST_RESIDENT"] = "1"
        observed, data = record_execution(binary, valid, record, env=env)
        assert (observed.returncode == 0) == expected, (free, resident, data)
        assert observed.stderr == "1"
    observed, data = record_execution(binary, valid + ["--m", "2048", "--n", "2048", "--k", "2048"],
                                      record, env=dict(os.environ, TEST_GUARD="1"))
    assert observed.returncode != 0 and observed.stderr == "0" and "cap refused" in data["error"]
    record["checks_passed"] = len(record["executions"])
    check_finite_baseline(base, packed, temp, result)


def check_finite_baseline(base: str, packed: str, temp: Path, result: dict) -> None:
    result["finite_baseline_fault_injection"] = record = {}
    start = packed.index("  ErrorMetrics baseline_cpu, tiled_cpu, certified_cpu,")
    end = packed.index("  const std::string prefix = packed_case_prefix", start)
    actual_comparison = packed[start:end]
    prelude = r'''
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <iomanip>
#include <string>
#include <vector>
struct Input{};const Input in;
float cpu_reference(const Input&,size_t index){return float(index+1);}
'''
    main = r'''
int main(int argc,char** argv){
  const std::string mode=argc>1?argv[1]:"finite";
  const size_t output_count=4;const std::vector<size_t> indices={0};
  std::vector<float> baseline={1,2,3,4},tiled=baseline,certified=baseline;
  if(mode=="shared_nan_hole")baseline[3]=tiled[3]=certified[3]=std::numeric_limits<float>::quiet_NaN();
  if(mode=="shared_inf_hole")baseline[3]=tiled[3]=certified[3]=std::numeric_limits<float>::infinity();
  if(mode=="candidate_nan_hole")certified[3]=std::numeric_limits<float>::quiet_NaN();
''' + actual_comparison + r'''
  std::cout<<"{\"case\":\""<<mode<<"\",\"passed\":"<<(passed?"true":"false")
    <<",\"baseline_nonfinite_values_all_outputs\":"<<baseline_nonfinite_values
    <<",\"tiled_exact_mismatches\":"<<tiled_baseline.exact_mismatches
    <<",\"certified_exact_mismatches\":"<<certified_baseline.exact_mismatches<<"}\n";
  return 0;
}
'''
    source = (prelude + fragment(base, "std::string number(") + "\n"
              + fragment(base, "bool exact_equal(") + "\n"
              + fragment(base, "struct ErrorMetrics {") + ";\n" + main)
    path, binary = temp / "finite_baseline.cpp", temp / "finite_baseline"
    path.write_text(source)
    compile_cpp(path, binary, record)
    for mode, expected in [("finite", True), ("shared_nan_hole", False),
                           ("shared_inf_hole", False), ("candidate_nan_hole", False)]:
        observed, data = record_execution(binary, [mode], record)
        assert observed.returncode == 0 and data["passed"] == expected
        if mode.startswith("shared_"):
            assert data["baseline_nonfinite_values_all_outputs"] == 1
            assert data["tiled_exact_mismatches"] == data["certified_exact_mismatches"] == 0
    record["checks_passed"] = len(record["executions"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script = Path(__file__).resolve()
    repo = next(p for p in script.parents if (p / "tools/mxfp4_training/csrc/check_mxfp4.cu").is_file())
    src = repo / "tools/mxfp4_training/csrc"
    record = {
        "schema_version": 1, "evidence_type": "host_cpp_harness_tests",
        "gpu_executed": False, "cuda_device_code_compiled": False, "cuda_linked": False,
        "memory_snapshots_are_test_stubs": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "compiler_version": run(["g++", "--version"]).stdout.splitlines()[0],
        "script_sha256": sha256(script),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in [
            src / "check_mxfp4.cu", src / "check_mxfp4_certified.cu",
            src / "bench_mxfp4.cu", src / "bench_mxfp4_fused.cu",
            src / "mxfp4_gemm_certified.cu"]},
    }
    try:
        with tempfile.TemporaryDirectory(prefix="mxfp4-harness-host-") as name:
            temp = Path(name)
            check_profile(src, temp, record)
            check_packed(src, temp, record)
        record["passed"] = True
    except Exception:
        record["passed"] = False
        record["error"] = traceback.format_exc()
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["behavior_checks_passed"] = (
        sum(r.get("checks_passed", 0) for r in record.get("correctness_profile", []))
        + record.get("packed_parser_and_guard", {}).get("checks_passed", 0)
        + record.get("finite_baseline_fault_injection", {}).get("checks_passed", 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "gpu_executed": False,
                      "behavior_checks_passed": record["behavior_checks_passed"],
                      "output": str(args.output), "output_sha256": sha256(args.output)}))
    if not record["passed"]:
        print(record["error"])
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
