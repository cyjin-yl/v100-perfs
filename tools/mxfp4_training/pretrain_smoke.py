"""Bounded causal-decoder training smoke; synthetic data, not LLM convergence.

Run from the repository root, for example:
python -m tools.mxfp4_training.pretrain_smoke --output /tmp/mxfp4-smoke.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .torch_ops import MXLinear, MXRecipe


class TinyDecoder(nn.Module):
    """One pre-norm, causal self-attention block and an untied output head."""
    def __init__(self, recipe: MXRecipe, *, vocab: int = 32, width: int = 32,
                 length: int = 16, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.token = nn.Embedding(vocab, width)
        self.position = nn.Parameter(torch.zeros(length, width))
        self.norm1, self.norm2, self.norm3 = [nn.LayerNorm(width) for _ in range(3)]
        self.qkv = MXLinear(width, 3 * width, recipe=recipe)
        self.proj = MXLinear(width, width, recipe=recipe)
        self.fc1 = MXLinear(width, 2 * width, recipe=recipe)
        self.fc2 = MXLinear(2 * width, width, recipe=recipe)
        self.head = MXLinear(width, vocab, bias=False, recipe=recipe)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.token(ids) + self.position[:ids.shape[1]]
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = [qkv[:, :, i].transpose(1, 2) for i in range(3)]
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(attention.transpose(1, 2).reshape(batch, length, width))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return self.head(self.norm3(x))


def batch_for(step: int, device: str, *, batch: int = 4, length: int = 16):
    # An intentionally simple next-token task. Starts vary deterministically;
    # the vocabulary and rule are public and there is no external dataset.
    start = (torch.arange(batch, device=device) * 7 + step * 3) % 32
    seq = (start[:, None] + torch.arange(length + 1, device=device)) % 32
    return seq[:, :-1], seq[:, 1:]


def loss_for(model: nn.Module, ids: torch.Tensor, targets: torch.Tensor):
    logits = model(ids)
    return F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))


def run_arm(name: str, recipe: MXRecipe, state: dict, args) -> dict:
    torch.manual_seed(args.seed)
    model = TinyDecoder(recipe).to(args.device)
    model.load_state_dict(state)
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    validation = batch_for(101, args.device, batch=8)
    with torch.no_grad():
        before = float(loss_for(model, *validation))
    losses = []
    max_gradient = 0.0
    start = time.monotonic()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_for(model, *batch_for(step, args.device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{name}: non-finite loss at step {step}")
        loss.backward()
        for param_name, param in model.named_parameters():
            if param.grad is None or not bool(torch.isfinite(param.grad).all()):
                raise RuntimeError(f"{name}: missing/non-finite gradient {param_name}")
            max_gradient = max(max_gradient, float(param.grad.abs().max()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    wall_seconds = time.monotonic() - start
    with torch.no_grad():
        after = float(loss_for(model, *validation))
    changed = [key for key, param in model.named_parameters()
               if not torch.equal(initial[key], param.detach())]
    all_finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    optimizer_fp32 = all(value.dtype == torch.float32
                         for values in optimizer.state.values() for key, value in values.items()
                         if isinstance(value, torch.Tensor) and key != "step")
    passed = (math.isfinite(after) and after < before and all_finite
              and bool(changed) and optimizer_fp32)
    return {
        "name": name, "recipe": asdict(recipe), "steps": args.steps,
        "validation_loss_before": before, "validation_loss_after": after,
        "training_losses": losses, "max_abs_gradient": max_gradient,
        "changed_parameter_tensors": len(changed), "parameters_finite": all_finite,
        "master_parameters_fp32": all(p.dtype == torch.float32 for p in model.parameters()),
        "optimizer_states_fp32": optimizer_fp32, "wall_seconds": wall_seconds,
        "timing_scope": "instrumented synthetic smoke only; not a performance benchmark",
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--backend", choices=["reference", "cuda", "cuda_tiled", "cuda_certified"], default="reference")
    parser.add_argument("--quantizer-backend", choices=["torch", "cuda_rne"], default="torch")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence with a new path")
    if args.steps <= 0 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps and learning rate must be positive")
    if (args.backend == "reference") != (args.device == "cpu"):
        parser.error("reference requires CPU; CUDA requires an explicit CUDA device")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.backend != "reference":
        free, _ = torch.cuda.mem_get_info(args.device)
        if free < 512 * 1024 * 1024:
            raise RuntimeError("CUDA smoke needs at least 512 MiB free after context creation")
    baseline = MXRecipe(False, False, False, backend=args.backend, quantizer_backend=args.quantizer_backend)
    prototype = TinyDecoder(baseline)
    state = {k: v.detach().clone() for k, v in prototype.state_dict().items()}
    arms = [
        ("fp32", baseline),
        ("mx_forward", MXRecipe(True, False, False, backend=args.backend, quantizer_backend=args.quantizer_backend)),
        ("mx_forward_dgrad", MXRecipe(True, True, False, backend=args.backend, quantizer_backend=args.quantizer_backend)),
        ("mx_forward_dgrad_wgrad", MXRecipe(True, True, True, backend=args.backend, quantizer_backend=args.quantizer_backend)),
        ("mx_all_stochastic_grad", MXRecipe(True, True, True,
                                           gradient_rounding="stochastic", backend=args.backend, quantizer_backend=args.quantizer_backend)),
    ]
    report = {
        "schema_version": 1, "purpose": "causal next-token integration smoke",
        "dataset": "synthetic length-16 modular token sequences, vocabulary 32",
        "model": {"layers": 1, "width": 32, "heads": 4,
                  "parameter_count": sum(p.numel() for p in prototype.parameters())},
        "seed": args.seed, "learning_rate": args.lr, "optimizer": "AdamW FP32",
        "backend": args.backend, "quantizer_backend": args.quantizer_backend, "device": args.device,
        "certified_geometry_note": "Width-32 smoke does not establish 128x128 halfrow fast-loader performance; paths are not instrumented.",
        "gpu_executed": args.backend != "reference", "llm_convergence_validated": False,
        "performance_claim": None, "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob("*.py"))},
        "arms": [],
    }
    for name, recipe in arms:
        result = run_arm(name, recipe, state, args)
        report["arms"].append(result)
        print(f"{name}: {result['validation_loss_before']:.6f} -> "
              f"{result['validation_loss_after']:.6f}, passed={result['passed']}", flush=True)
    report["passed"] = all(r["passed"] for r in report["arms"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
