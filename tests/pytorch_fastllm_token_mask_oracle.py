#!/usr/bin/env python3
"""Compare FastLLM's CUDA token mask against PyTorch CUDA bit-for-bit."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
EXECUTABLE = ROOT / "fastllm/build-rw/regressionOps"
SHAPES = [
    (1, 1),
    (2, 8),
    (3, 257),
    (4, 4097),
    (2, 248320),
    (7, 248320),
]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is required")
    if not EXECUTABLE.exists():
        raise RuntimeError(f"missing FastLLM regression binary: {EXECUTABLE}")

    torch.cuda.set_device(0)
    torch.manual_seed(20260821)
    rng = np.random.default_rng(20260821)
    checked = 0
    cases = 0

    with tempfile.TemporaryDirectory(prefix="fastllm-mask-oracle-") as directory:
        directory = Path(directory)
        for batch, vocab in SHAPES:
            for iteration in range(3):
                original = torch.randn(
                    (batch, vocab), device="cuda", dtype=torch.float32
                )
                mask_numpy = rng.random((batch, vocab)) > (0.15 + 0.2 * iteration)
                mask_numpy[:, 0] = False
                mask_numpy[:, -1] = True
                if vocab > 7:
                    # The disallowed token has the largest logit; the allowed
                    # intersection must still select token 3.
                    mask_numpy[0, 3] = True
                    mask_numpy[0, 5] = False
                    original[0, 3] = 123.0
                    original[0, 5] = 124.0

                mask = torch.from_numpy(mask_numpy).to("cuda")
                expected = torch.where(
                    mask, original, torch.full_like(original, -1.0e30)
                )
                logits_path = directory / "logits.bin"
                mask_path = directory / "mask.bin"
                output_path = directory / "output.bin"
                original.cpu().numpy().tofile(logits_path)
                np.ascontiguousarray(mask_numpy, dtype=np.uint8).tofile(mask_path)

                environment = os.environ.copy()
                environment.update(
                    {
                        "FASTLLM_REGRESSION_ONLY": "cuda_token_mask_oracle",
                        "FASTLLM_TOKEN_MASK_LOGITS": str(logits_path),
                        "FASTLLM_TOKEN_MASK_MASK": str(mask_path),
                        "FASTLLM_TOKEN_MASK_OUTPUT": str(output_path),
                        "FASTLLM_TOKEN_MASK_BATCH": str(batch),
                        "FASTLLM_TOKEN_MASK_VOCAB": str(vocab),
                        "CUDA_VISIBLE_DEVICES": "0",
                    }
                )
                result = subprocess.run(
                    [str(EXECUTABLE)],
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=120,
                    check=False,
                )
                if result.returncode != 0:
                    detail = (result.stderr + result.stdout)[-2000:]
                    raise AssertionError(f"FastLLM oracle process failed: {detail}")

                actual_numpy = np.fromfile(output_path, dtype=np.float32).reshape(
                    batch, vocab
                )
                actual = torch.from_numpy(actual_numpy).to("cuda")
                if not torch.equal(actual, expected):
                    bit_difference = (
                        actual.view(torch.int32) != expected.view(torch.int32)
                    )
                    bad = torch.nonzero(bit_difference)[0].tolist()
                    raise AssertionError(
                        f"bitwise mismatch batch={batch} vocab={vocab} at {bad}"
                    )
                if not torch.equal(actual.argmax(-1), expected.argmax(-1)):
                    raise AssertionError(
                        f"argmax mismatch batch={batch} vocab={vocab}"
                    )
                top_k = min(20, vocab)
                if not torch.equal(
                    actual.topk(top_k, dim=-1).indices,
                    expected.topk(top_k, dim=-1).indices,
                ):
                    raise AssertionError(
                        f"top-k mismatch batch={batch} vocab={vocab}"
                    )
                checked += batch * vocab
                cases += 1

    print(
        {
            "oracle": "PyTorch CUDA torch.where",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cases": cases,
            "elements_bitwise_checked": checked,
            "argmax": "exact",
            "topk": "exact",
            "result": "PASS",
        }
    )


if __name__ == "__main__":
    main()
