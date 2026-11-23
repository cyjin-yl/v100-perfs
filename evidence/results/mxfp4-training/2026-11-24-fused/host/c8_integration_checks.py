#!/usr/bin/env python3
"""Check the real C8 build CLI and compile-time gate without CUDA or a GPU.

The build entry point runs with only its subprocess.run replaced by an explicit
host-only compiler stub. Its temporary artifacts and compilation report are
discarded; this report records command planning, never CUDA compilation success.
The certified variant gate is separately passed through the actual C++ host
preprocessor. The unchanged check/benchmark sources are bound by SHA256.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from unittest.mock import patch
from harness_host_checks import fragment


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_fixtures_and_dispatch(src: Path, temp: Path) -> dict:
    base = (src / "check_mxfp4.cu").read_text()
    wrapper = (src / "check_mxfp4_certified.cu").read_text()
    core = (src / "mxfp4_gemm_certified.cu").read_text()
    fused = (src / "mxfp4_gemm_fused.cu").read_text()
    small_start = fused.index("  const bool small =")
    small = fused[small_start:fused.index(";", small_start) + 1]
    eligible_start = fused.index("sizeof(Index) == sizeof(int)", fused.index("void dispatch_variant("))
    eligible = fused[eligible_start:fused.index(")\n    fused_certified_prefetch32_halfrow", eligible_start)]
    prelude = r'''
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
constexpr int cudaSuccess=0,cudaErrorInvalidValue=1;
constexpr bool kFused=true; constexpr size_t kBlasWorkspace=4u*1024u*1024u;
constexpr unsigned MXFP4_CERT_REJECT_NAN_SCALE=1u;
'''
    workspace = (fragment(core, "struct RowData {") + ";\n"
                 + fragment(core, "struct DeviceMetadata {") + ";\n"
                 + fragment(core, "int validate_shape(") + "\n"
                 + fragment(core, 'extern "C" int mxfp4_certified_workspace_bytes(') + "\n")
    helpers = base[base.index("constexpr int qtable[16]"):base.index("uint64_t required_bytes(")]
    helpers += fragment(base, "std::string json_string(") + "\n"
    helpers += '''uint64_t extra(int m,int n,int k){uint64_t b=0;
if(mxfp4_certified_workspace_bytes(m,n,k,&b))throw std::runtime_error("workspace");return b;}
#define MXFP4_CHECK_EXTRA_BYTES(m,n,k) extra(m,n,k)
'''
    helpers += fragment(base, "uint64_t required_bytes(") + "\n"
    helpers += fragment(wrapper, "Case multitile_row_bases(") + "\n"
    helpers += fragment(wrapper, "struct C8Fixture {") + ";\n"
    helpers += fragment(wrapper, "Case c8_max_delta(") + "\n"
    helpers += fragment(wrapper, "std::vector<C8Fixture> c8_loader_fixtures(") + "\n"
    route = "template<class Index>bool eligible(const uint8_t* a,const uint8_t* b,int64_t m,int64_t n){return " + eligible + ";}\n"
    route += '''int route(int64_t m,int64_t n,int64_t padded_k,uintptr_t ap,uintptr_t bp){
if(!m||!n)return 0; const int64_t int_limit=std::numeric_limits<int>::max();
''' + small + '''
auto a=reinterpret_cast<const uint8_t*>(ap);auto b=reinterpret_cast<const uint8_t*>(bp);
return small?(eligible<int>(a,b,m,n)?1:2):(eligible<int64_t>(a,b,m,n)?1:2);}
'''
    main = r'''
struct Rows { uint64_t energy=0; int minbase=255,maxbase=0,span=0; bool nan=false; };
Rows rows(const Case& c,const std::vector<uint8_t>& codes,const std::vector<uint8_t>& scales,int count){
  Rows result;
  for(int row=0;row<count;++row){
    int lo=255,hi=0;std::vector<unsigned> e(c.blocks);
    for(int block=0;block<c.blocks;++block){
      const int scale=scales[size_t(row)*c.blocks+block];result.nan|=scale==255;
      for(int kk=block*32;kk<(block+1)*32;++kk){const int q=get_q(codes,row,kk,c.padded_k);e[block]+=q*q;}
      if(e[block]){lo=std::min(lo,scale);hi=std::max(hi,scale);}
    }
    if(lo==255)lo=hi=127;
    uint64_t energy=0;
    for(int block=0;block<c.blocks;++block)if(e[block]){
      const int delta=int(scales[size_t(row)*c.blocks+block])-lo;
      if(delta<0||delta>12)throw std::runtime_error("fixture span");
      energy+=uint64_t(e[block])<<(2*delta);
    }
    result.energy=std::max(result.energy,energy);result.span=std::max(result.span,hi-lo);
    result.minbase=std::min(result.minbase,lo);result.maxbase=std::max(result.maxbase,lo);
  }return result;
}
int main(){try{
  uint64_t peak=0,values=0,zeros=0;int cases=0,accepted_count=0;
  uint32_t bits=0x7f7f7f7f;float poison;std::memcpy(&poison,&bits,4);
  for(const auto& f:c8_loader_fixtures()){
    const auto& c=f.data;const auto a=rows(c,c.a,c.as,c.m),b=rows(c,c.b,c.bs,c.n);
    const __uint128_t product=__uint128_t(a.energy)*b.energy;
    const int lambda_min=a.minbase+b.minbase-256,lambda_max=a.maxbase+b.maxbase-256;
    const bool accepted=!a.nan&&!b.nan&&a.span<=12&&b.span<=12&&product<=(uint64_t(1)<<48)
        &&lambda_min>=-126&&lambda_max<=103;
    if(accepted!=f.accepted)throw std::runtime_error(c.name+": independent certificate mismatch");
    if(c.name.find("delta12")!=std::string::npos&&a.span!=12)throw std::runtime_error("delta12 absent");
    const auto expected=reference(c);uint64_t casezeros=0;
    for(float v:expected){if(v==poison)throw std::runtime_error("poison is valid output");if(v==0){
      if(std::signbit(v))throw std::runtime_error("negative expected zero");++casezeros;}}
    const uint64_t bytes=required_bytes(c);if(bytes>(8ull<<20))throw std::runtime_error("cap");
    const int dispatch=route(c.m,c.n,c.padded_k,c.input_byte_offset,c.input_byte_offset);
    if((dispatch==1)!=f.vector_eligible)throw std::runtime_error("fixture route");
    peak=std::max(peak,bytes);values+=expected.size();zeros+=casezeros;++cases;accepted_count+=accepted;
    std::cout<<"{\"type\":\"host_fixture\",\"case\":"<<json_string(c.name)
      <<",\"values\":"<<expected.size()<<",\"required_bytes\":"<<bytes
      <<",\"accepted\":"<<(accepted?"true":"false")<<",\"energy_a\":"<<a.energy
      <<",\"energy_b\":"<<b.energy<<",\"energy_product\":"<<uint64_t(product)
      <<",\"max_span_a\":"<<a.span<<",\"max_span_b\":"<<b.span
      <<",\"lambda_min\":"<<lambda_min<<",\"lambda_max\":"<<lambda_max
      <<",\"reference_zero_values\":"<<casezeros<<"}\n";
  }
  if(cases!=9||values!=180224||peak!=170312||accepted_count!=8)throw std::runtime_error("fixture totals");
  int dispatch_cases=0;
  const std::array<std::array<int64_t,2>,9> shapes={{{0,128},{128,0},{1,1},{127,128},{128,128},
      {129,128},{128,129},{128,256},{256,128}}};
  for(auto shape:shapes)for(int k:{0,32,96,160})for(uintptr_t a=0;a<8;++a)for(uintptr_t b=0;b<8;++b){
    const int expected=(!shape[0]||!shape[1])?0:
      (shape[0]%128==0&&shape[1]%128==0&&a==0&&b==0)?1:2;
    if(route(shape[0],shape[1],k,a,b)!=expected)throw std::runtime_error("dispatch alignment/tail");
    ++dispatch_cases;
  }
  const std::array<std::array<int64_t,4>,6> boundaries={{{128,16777088,0,1},{128,16777216,0,2},
      {128,128,33554400,1},{128,128,33554432,2},{128,128,2147483616,2},{128,128,2147483648ll,2}}};
  for(auto t:boundaries){if(route(t[0],t[1],t[2],0,0)!=t[3])throw std::runtime_error("index boundary");++dispatch_cases;}
  std::cout<<"{\"type\":\"host_fixture_summary\",\"cases\":"<<cases<<",\"values\":"<<values
    <<",\"accepted_cases\":"<<accepted_count<<",\"peak_bytes\":"<<peak
    <<",\"reference_zero_values\":"<<zeros<<",\"actual_source_dispatch_cases\":"<<dispatch_cases
    <<",\"gpu_executed\":false,\"passed\":true}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what();return 1;}}
'''
    cpp, binary = temp / "actual_c8_fixtures.cpp", temp / "actual_c8_fixtures"
    cpp.write_text(prelude + workspace + helpers + route + main)
    command = ["g++", "-std=c++17", "-O2", str(cpp), "-o", str(binary)]
    compiled = subprocess.run(command, capture_output=True, text=True)
    if compiled.returncode:
        raise AssertionError(compiled.stderr)
    observed = subprocess.run([str(binary)], capture_output=True, text=True)
    if observed.returncode:
        raise AssertionError(observed.stderr)
    return {"host_compile_argv": command, "host_compile_stderr": compiled.stderr,
            "generated_host_source_sha256": sha256(cpp), "records": [json.loads(line) for line in observed.stdout.splitlines()],
            "stderr": observed.stderr, "passed": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve earlier host evidence")
    script = Path(__file__).resolve()
    repo = next(p for p in script.parents if (p / "tools/mxfp4_training/build_cuda.py").is_file())
    build_path = repo / "tools/mxfp4_training/build_cuda.py"
    src = build_path.parent / "csrc"
    spec = importlib.util.spec_from_file_location("mxfp4_build_host_check", build_path)
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    record = {
        "schema_version": 1, "evidence_type": "c8_host_integration_checks",
        "gpu_executed": False, "cuda_device_code_compiled": False,
        "cuda_linked": False, "build_compiler_is_test_stub": True,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256(script),
        "source_sha256": {str(p.relative_to(repo)): sha256(p) for p in [
            build_path, src / "mxfp4_gemm_certified.cu", src / "mxfp4_gemm_certified.h",
            src / "mxfp4_gemm_fused.cu",
            src / "check_mxfp4.cu", src / "check_mxfp4_certified.cu",
            src / "bench_mxfp4.cu", src / "bench_mxfp4_fused.cu",
            build_path.parent / "crater_remote_run.py",
        ]},
        "build_cli_checks": [], "preprocessor_gate_checks": [],
    }
    try:
        ast.parse(build_path.read_text(), filename=str(build_path))
        ast.parse(script.read_text(), filename=str(script))
        record["python_syntax_passed"] = True
        cases = [
            ("default_c2", ["--with-certified", "--library-only"], 0, 2,
             ["shared_library"]),
            ("c8_full", ["--with-certified", "--certified-variant", "8",
                         "--with-benchmark", "--with-fused-benchmark"], 0, 8,
             ["shared_library", "standalone_harness", "tiled_harness", "certified_harness",
              "performance_benchmark", "fused_packed_benchmark"]),
            ("c8_library_and_packed", ["--with-certified", "--certified-variant", "8",
                                       "--library-only", "--with-fused-benchmark"], 0, 8,
             ["shared_library", "fused_packed_benchmark"]),
            ("c3_legacy", ["--with-certified", "--certified-variant", "3",
                           "--library-only"], 0, 3, ["shared_library"]),
            ("c8_without_certified", ["--certified-variant", "8", "--with-fused-benchmark"],
             2, None, []),
            ("c2_packed_refused", ["--with-certified", "--certified-variant", "2",
                                   "--with-fused-benchmark"], 2, None, []),
            ("variant9_refused", ["--with-certified", "--certified-variant", "9"], 2, None, []),
            ("negative_variant_refused", ["--with-certified", "--certified-variant", "-1"],
             2, None, []),
        ]
        with tempfile.TemporaryDirectory(prefix="mxfp4-c8-host-integration-") as name:
            temp = Path(name)
            for label, options, expected_status, variant, expected_jobs in cases:
                calls = []

                def compiler_stub(command, **kwargs):
                    calls.append(command)
                    if command[-1] == "--version":
                        return subprocess.CompletedProcess(command, 0,
                            "HOST-ONLY COMPILER STUB; NO CUDA CODE COMPILED\n", "")
                    Path(command[-1]).write_bytes(b"HOST-ONLY COMPILER STUB ARTIFACT\n")
                    return subprocess.CompletedProcess(command, 0, "", "")

                output = temp / label
                argv = [str(build_path), "--output-dir", str(output),
                        "--nvcc", "/host-only-test-stub/nvcc", *options]
                stdout, stderr = io.StringIO(), io.StringIO()
                with patch.object(sys, "argv", argv), patch.object(build.subprocess, "run", compiler_stub), \
                        redirect_stdout(stdout), redirect_stderr(stderr):
                    try:
                        status = build.main()
                    except SystemExit as exc:
                        status = exc.code
                item = {"case": label, "options": options, "returncode": status,
                        "stderr": stderr.getvalue(), "planned_jobs": []}
                record["build_cli_checks"].append(item)
                assert status == expected_status, item
                if status:
                    assert not calls, calls
                    continue
                planned = json.loads((output / "compile-evidence.json").read_text())
                item["planned_jobs"] = planned["jobs"]
                item["certified_variant"] = planned["certified_variant"]
                item["tiled_variant"] = planned["tiled_variant"]
                assert planned["certified_variant"] == variant
                assert planned["tiled_variant"] == 1
                assert [job["name"] for job in planned["jobs"]] == expected_jobs
                for job in planned["jobs"]:
                    command = job["argv"]
                    assert f"-DMXFP4_CERTIFIED_VARIANT={variant}" in command
                    assert "--ftz=false" in command
                    assert "-gencode=arch=compute_70,code=[sm_70,compute_70]" in command
                    is_certified_job = job["name"] not in ("standalone_harness", "tiled_harness")
                    assert (str(src / "mxfp4_gemm_fused.cu") in command) == (variant >= 3 and is_certified_job)
                    if job["name"] == "certified_harness":
                        assert str(src / "check_mxfp4_certified.cu") in command
                    if job["name"] == "fused_packed_benchmark":
                        assert "-DMXFP4_BENCH_CERTIFIED" in command
                        assert "-DMXFP4_BENCH_TILED" in command
                        assert str(src / "bench_mxfp4_fused.cu") in command

            core = (src / "mxfp4_gemm_certified.cu").read_text()
            gate = core[core.index("#ifndef MXFP4_CERTIFIED_VARIANT"):core.index("namespace {")]
            gate_source = temp / "actual_certified_variant_gate.cpp"
            gate_source.write_text(gate + "\nMXFP4_CERTIFIED_VARIANT\n")
            for variant in [None, -1, 0, 2, 3, 7, 8, 9]:
                command = ["g++", "-std=c++17", "-E", "-P"]
                if variant is not None:
                    command.append(f"-DMXFP4_CERTIFIED_VARIANT={variant}")
                command.append(str(gate_source))
                observed = subprocess.run(command, capture_output=True, text=True)
                item = {"variant_define": variant, "returncode": observed.returncode,
                        "stdout": observed.stdout, "stderr": observed.stderr}
                record["preprocessor_gate_checks"].append(item)
                expected = variant is None or 0 <= variant <= 8
                assert (observed.returncode == 0) == expected, item
                if expected:
                    assert int(observed.stdout.strip()) == (0 if variant is None else variant)
                else:
                    assert "MXFP4_CERTIFIED_VARIANT must be" in observed.stderr
            record["c8_fixture_and_dispatch_checks"] = check_fixtures_and_dispatch(src, temp)

        unchanged = [src / "check_mxfp4.cu", src / "bench_mxfp4.cu", src / "bench_mxfp4_fused.cu",
                     build_path.parent / "crater_remote_run.py"]
        record["unchanged_vs_head"] = {}
        for path in unchanged:
            old = subprocess.run(["git", "show", f"HEAD:{path.relative_to(repo)}"],
                                 cwd=repo, check=True, capture_output=True).stdout
            matched = old == path.read_bytes()
            record["unchanged_vs_head"][str(path.relative_to(repo))] = matched
            assert matched, path
        record["passed"] = True
    except Exception:
        record["passed"] = False
        record["error"] = traceback.format_exc()
    record["finished_utc"] = datetime.now(timezone.utc).isoformat()
    record["cases_checked"] = len(record["build_cli_checks"]) + len(record["preprocessor_gate_checks"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"passed": record["passed"], "gpu_executed": False,
                      "cuda_device_code_compiled": False, "cases_checked": record["cases_checked"],
                      "output": str(args.output), "output_sha256": sha256(args.output)}))
    if not record["passed"]:
        print(record["error"])
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
