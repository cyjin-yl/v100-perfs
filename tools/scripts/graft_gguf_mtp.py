#!/usr/bin/env python3
"""Graft a Qwen3.5 MTP draft block from a source GGUF into a target GGUF.

The output is a composite Qwen3.5 GGUF:
  * every target tensor is copied byte-for-byte, unrequantized;
  * the source's nextn trunk block (source block index block_count-1) is
    appended as the final trunk block, keeping its original names;
  * qwen35.block_count is overridden to target_blocks + source_nextn and
    qwen35.nextn_predict_layers to source_nextn.

FastLLM's existing Qwen3.5 GGUF loader then treats the appended block as the
MTP draft layer (model.cpp reroutes layers.{block_count} into mtp.layers.0.*),
so no runtime change is required.  Inputs are opened read-only and never
modified; the output is written to a fresh path and re-verified against both
origins by tensor name/type/shape and raw bytes.
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter, GGUFValueType

SKIP_KEYS = ("general.name", "qwen35.block_count", "qwen35.nextn_predict_layers")
CORE_DIM_KEYS = (
    "qwen35.embedding_length",
    "qwen35.attention.head_count",
    "qwen35.attention.head_count_kv",
    "qwen35.attention.key_length",
    "qwen35.feed_forward_length",
    "qwen35.context_length",
)


def field_value(reader, key, default=None):
    field = reader.fields.get(key)
    if field is None:
        return default
    return field.contents()


def fail(msg):
    raise SystemExit(f"graft_gguf_mtp: {msg}")


def layer_ids(reader):
    ids = set()
    for tensor in reader.tensors:
        name = tensor.name
        if name.startswith("blk."):
            rest = name[4:]
            digits = ""
            for ch in rest:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits and rest[len(digits)] == ".":
                ids.add(int(digits))
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="GGUF whose tensors become layers 0..N-1")
    parser.add_argument("--mtp-source", required=True, help="GGUF providing the nextn draft block")
    parser.add_argument("--output", required=True, help="composite GGUF output path")
    parser.add_argument("--name", default=None, help="general.name override for the composite")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the manifest only")
    parser.add_argument("--no-verify", action="store_true", help="skip byte-level verification")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output")
    args = parser.parse_args()

    target = GGUFReader(args.target, "r")
    source = GGUFReader(args.mtp_source, "r")

    for reader, label in ((target, "target"), (source, "mtp-source")):
        arch = field_value(reader, "general.architecture")
        if arch != "qwen35":
            fail(f"{label} architecture is {arch!r}, expected 'qwen35'")

    target_blocks = field_value(target, "qwen35.block_count")
    source_blocks = field_value(source, "qwen35.block_count")
    target_nextn = field_value(target, "qwen35.nextn_predict_layers", 0)
    source_nextn = field_value(source, "qwen35.nextn_predict_layers", 0)
    if target_nextn not in (0, None):
        fail(f"target already carries MTP (nextn={target_nextn}); refuse to graft")
    if source_nextn != 1:
        fail(f"mtp-source nextn={source_nextn}, expected exactly 1")

    for key in CORE_DIM_KEYS:
        tv, sv = field_value(target, key), field_value(source, key)
        if tv != sv:
            fail(f"{key} mismatch: target={tv} source={sv}")

    mtp_layer = source_blocks - 1
    target_ids = layer_ids(target)
    if target_ids and max(target_ids) != target_blocks - 1:
        fail(f"target layer ids {sorted(target_ids)[-3:]}... do not end at block_count-1={target_blocks - 1}")
    if target_blocks in target_ids:
        fail(f"target already contains layer {target_blocks}; composite block numbering would collide")

    mtp_tensors = [t for t in source.tensors if t.name.startswith(f"blk.{mtp_layer}.")]
    if not mtp_tensors:
        fail(f"mtp-source has no blk.{mtp_layer}.* tensors")
    if not any("nextn" in t.name for t in mtp_tensors):
        fail(f"mtp-source blk.{mtp_layer} lacks nextn tensors")
    for t in target.tensors:
        if "nextn" in t.name or t.name.startswith("mtp."):
            fail(f"target contains MTP tensor {t.name}; refuse to graft")

    manifest = []
    for t in target.tensors:
        manifest.append({
            "name": t.name, "origin": "target", "type": t.tensor_type.name,
            "shape": [int(x) for x in t.shape], "nbytes": t.n_bytes,
            "offset": t.data_offset,
        })
    for t in mtp_tensors:
        manifest.append({
            "name": t.name, "origin": "mtp-source", "type": t.tensor_type.name,
            "shape": [int(x) for x in t.shape], "nbytes": t.n_bytes,
            "offset": t.data_offset,
        })
    total_bytes = sum(m["nbytes"] for m in manifest)

    print(json.dumps({
        "target": args.target, "mtp_source": args.mtp_source,
        "target_blocks": target_blocks, "source_blocks": source_blocks,
        "composite_blocks": target_blocks + source_nextn,
        "tensor_count": len(manifest), "total_bytes": total_bytes,
        "mtp_tensor_count": len(mtp_tensors),
        "mtp_bytes": sum(m["nbytes"] for m in manifest if m["origin"] == "mtp-source"),
    }, indent=2))

    if args.dry_run:
        return

    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        fail(f"output {out_path} exists; pass --overwrite to replace")

    writer = GGUFWriter(None, "qwen35", use_temp_file=True)
    for name, field in target.fields.items():
        if name in SKIP_KEYS or name.startswith("GGUF.") or name == "general.architecture":
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        writer.add_key_value(name, field.contents(), vtype, sub_type=sub_type)

    name_override = args.name or f"{field_value(target, 'general.name', 'graft')}-plus-mtp"
    writer.add_string("general.name", name_override)
    writer.add_uint32("qwen35.block_count", target_blocks + source_nextn)
    writer.add_uint32("qwen35.nextn_predict_layers", source_nextn)
    writer.add_string("mtp_graft.schema", "1")
    writer.add_string("mtp_graft.target", str(Path(args.target).name))
    writer.add_string("mtp_graft.mtp_source", str(Path(args.mtp_source).name))
    writer.add_string("mtp_graft.created_utc",
                      datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    for t in target.tensors:
        writer.add_tensor(t.name, t.data, raw_shape=tuple(int(x) for x in t.data.shape),
                          raw_dtype=t.tensor_type)
    for t in mtp_tensors:
        writer.add_tensor(t.name, t.data, raw_shape=tuple(int(x) for x in t.data.shape),
                          raw_dtype=t.tensor_type)

    writer.open_output_file(out_path)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    if args.no_verify:
        print("graft_gguf_mtp: written without verification")
        return

    verify = GGUFReader(out_path, "r")
    if field_value(verify, "qwen35.block_count") != target_blocks + source_nextn:
        fail("verification: composite block_count wrong")
    if field_value(verify, "qwen35.nextn_predict_layers") != source_nextn:
        fail("verification: composite nextn_predict_layers wrong")
    if len(verify.tensors) != len(manifest):
        fail(f"verification: tensor count {len(verify.tensors)} != {len(manifest)}")

    by_name = {m["name"]: m for m in manifest}
    mismatches = []
    for t in verify.tensors:
        meta = by_name.get(t.name)
        if meta is None:
            mismatches.append(f"unexpected tensor {t.name}")
            continue
        if t.tensor_type.name != meta["type"] or [int(x) for x in t.shape] != meta["shape"]:
            mismatches.append(f"metadata mismatch for {t.name}")
            continue
        origin = target if meta["origin"] == "target" else source
        origin_tensor = next(o for o in origin.tensors if o.name == t.name)
        if t.n_bytes != origin_tensor.n_bytes or not np.array_equal(t.data, origin_tensor.data):
            mismatches.append(f"byte mismatch for {t.name}")
    if mismatches:
        fail("verification failed:\n" + "\n".join(mismatches))
    print(f"graft_gguf_mtp: verified {len(verify.tensors)} tensors, {total_bytes} bytes -> {out_path}")


if __name__ == "__main__":
    main()
