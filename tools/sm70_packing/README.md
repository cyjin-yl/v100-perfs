# SM70 packing evidence replay

The [research report](../../docs/research/performance/sm70-packing/index.md) defines the supported domains and separates CPU numerical evidence, recorded compiler evidence, cycle simulation, and pending hardware work.

From the repository root, with Python 3.10+, NumPy, and a C++17 compiler:

```bash
python3 tools/sm70_packing/run_cpu_validation.py \
  --require-round3 --random-cases 4096 \
  --output-dir /tmp/v100-packing-validation
```

The output directory must not already exist. The runner copies declared inputs into an isolated study, executes the checks there, and verifies that the repository evidence was not modified. It writes a `summary.json` with commands, input hashes, log hashes, timings of the CPU checks, and explicit evidence scopes. These CPU wall times are not GPU performance numbers.

No GPU, CUDA, network, Octave, or architecture-simulator binary is used by default. Parsing the recorded SASS and cycle log checks consistency; it does not repeat CUDA compilation or the original cycle simulation. The full numerical replay always checks all 5,000 published V100 fixtures; `--random-cases` controls the additional random cases for each of the six historical packing schemes. The round-three manifest declares its own deterministic sample sizes.

The seven explicit round-three checks are declared in [ci-checks.json](../../evidence/benchmarks/sm70-packing/study/simulation/round3/review/ci-checks.json). `--require-round3` fails if those checks cannot be located. The input manifest is repository-owned code and should be reviewed like any other test command.

Optional existing tools can be requested explicitly:

```bash
python3 tools/sm70_packing/run_cpu_validation.py \
  --require-round3 --octave-bin /path/to/octave-cli \
  --simulate --simulator-bin /path/to/accel-sim.out \
  --output-dir /tmp/v100-packing-full-replay
```

These options do not install software or access a GPU. The bundled simulator was built for the recorded Linux host; an ABI-compatible binary is required for replay. Licenses, pinned versions, traces, original logs, and build instructions are included under the study directory.

For actual GPU execution, use the separate [V100 harness](../../evidence/benchmarks/sm70-packing/study/simulation/round3/cuda/README.md). Its reported correctness and performance fields are generated only after a real run. The middle-coefficient decoder is a separate compiled probe, not an integrated matrix kernel.
