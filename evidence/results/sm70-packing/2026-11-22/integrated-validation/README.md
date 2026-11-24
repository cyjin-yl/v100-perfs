# Validation after repository integration

The unified runner was executed from this repository checkout after all frozen study inputs were copied into their final paths:

```bash
python3 tools/sm70_packing/run_cpu_validation.py \
  --require-round3 --random-cases 4096 \
  --output-dir /workspace/scratch/f8d9571c451b/verification/sm70-repo-final
```

All 16 checks passed, including seven round-three checks. The runner verified that 68 protected source inputs and historical result files remained unchanged. It rechecked 5,000 published hardware fixtures and 30,940 historical packing cases, then ran the new mathematical, fragment, decoder, preparation, and layout checks. The approximately 10.75-second wall time is CPU test execution time, not a V100 benchmark.

`summary.json` is the actual runner output. Its `logs/` paths refer to the files copied here unchanged; every log was checked against its recorded SHA-256. Absolute command paths record the execution environment. To reproduce elsewhere, use the command in [tools/sm70_packing](../../../../../tools/sm70_packing/README.md) with a new output directory.

The original Octave model and Accel-Sim were executed in the earlier study. This final default replay did not rerun those optional programs. It regenerated artificial traces and parsed recorded SASS/cycle logs, with each step's scope stated in the summary. No GPU or cuBLAS was executed. The isolated copy and its mechanically regenerated 10 MB search JSON are intentionally omitted from this evidence directory.
