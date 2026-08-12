#!/usr/bin/env python3
"""将 MTP 头（blk.64.* + nextn.*）从源 GGUF 移植到目标 GGUF。

用法: merge_mtp_head.py <src_mtp.gguf> <dst_no_mtp.gguf> <output.gguf>
MTP 头 tensor 从 src 复制，其余权重/元数据来自 dst；输出写入 output。
"""
import struct, sys, os

GGUF_MAGIC = b'GGUF'
TYPE_SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1}  # u8..bool

def read_str(f):
    ln = struct.unpack('<Q', f.read(8))[0]
    return f.read(ln)

def write_str(f, s):
    f.write(struct.pack('<Q', len(s)))
    f.write(s)

def read_gguf(path):
    """读 GGUF 的 metadata kv + tensor 信息（不读 tensor 数据）。
    返回 (kv_list, tensors, tensor_data_offsets)"""
    f = open(path, 'rb')
    magic = f.read(4)
    assert magic == GGUF_MAGIC, f'not gguf: {path}'
    ver = struct.unpack('<I', f.read(4))[0]
    n_tensors = struct.unpack('<Q', f.read(8))[0]
    n_kv = struct.unpack('<Q', f.read(8))[0]
    kv = []
    for _ in range(n_kv):
        key = read_str(f)
        t = struct.unpack('<I', f.read(4))[0]
        if t == 0: val = struct.unpack('<B', f.read(1))[0]
        elif t == 1: val = struct.unpack('<b', f.read(1))[0]
        elif t == 2: val = struct.unpack('<H', f.read(2))[0]
        elif t == 3: val = struct.unpack('<h', f.read(2))[0]
        elif t == 4: val = struct.unpack('<I', f.read(4))[0]
        elif t == 5: val = struct.unpack('<i', f.read(4))[0]
        elif t == 6: val = struct.unpack('<f', f.read(4))[0]
        elif t == 7: val = struct.unpack('<B', f.read(1))[0] != 0
        elif t == 8: val = read_str(f)
        elif t == 9:
            at = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            arr = []
            for _ in range(n):
                if at == 8:
                    arr.append(read_str(f))
                elif at in TYPE_SIZES:
                    arr.append(struct.unpack('<I', f.read(TYPE_SIZES[at]))[0])
                else:
                    raise ValueError(f'unsupported array elem type {at}')
            val = (at, arr)
        else:
            raise ValueError(f'unsupported kv type {t} for {key}')
        kv.append((key, t, val))
    tensors = []
    for _ in range(n_tensors):
        name = read_str(f)
        nd = struct.unpack('<I', f.read(4))[0]
        dims = struct.unpack('<' + 'Q' * nd, f.read(8 * nd))
        ttype = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]
        tensors.append((name, dims, ttype, offset))
    data_start = f.tell()
    f.close()
    return kv, tensors, data_start

def gguf_type_size(ttype, n_elements):
    """Q6_K 等块的字节数（llama.cpp 的 ggml_blck_size/type_size）。"""
    # GGML_TYPE_Q6_K: block_size 256, type_size 210 + 2 (scales?) => 210? llama: Q6_K type_size=210, blck=256
    # Q8_0: blck 32, type 34; Q5_K: blck 256, type 176; Q4_K: 144; Q6_K: 210; Q3_K: 112; Q2_K: 80
    blocks = {
        10: (32, 34),   # Q8_0
        12: (256, 176), # Q5_K
        13: (256, 144), # Q4_K
        14: (256, 210), # Q6_K
        15: (256, 112), # Q3_K
        16: (256, 80),  # Q2_K
        7: (1, 2),      # F16
        0: (1, 4),      # F32
    }
    if ttype in blocks:
        bs, ts = blocks[ttype]
        return (n_elements // bs) * ts
    if ttype == 9:  # BF16
        return n_elements * 2
    raise ValueError(f'unsupported tensor type {ttype}')

def main():
    src_path, dst_path, out_path = sys.argv[1:4]
    src_kv, src_tensors, src_data = read_gguf(src_path)
    dst_kv, dst_tensors, dst_data = read_gguf(dst_path)

    # 1) 目标模型里已有的 blk.64/nextn（应不存在，若存在则跳过该名）
    have = {t[0].decode() for t in dst_tensors}
    mtp_tensors = [t for t in src_tensors if t[0].startswith(b'blk.64.')]
    mtp_tensors = [t for t in mtp_tensors if t[0].decode() not in have]
    print(f'MTP tensors to graft: {len(mtp_tensors)}')
    for t in mtp_tensors:
        print('  ', t[0].decode(), t[1], 'type', t[2])

    if not mtp_tensors:
        print('ERROR: no MTP tensors found in src (already present in dst?)')
        sys.exit(1)

    # 2) 计算 MTP 数据大小
    mtp_bytes = {}
    with open(src_path, 'rb') as f:
        for name, dims, ttype, offset in mtp_tensors:
            n = 1
            for d in dims:
                n *= d
            sz = gguf_type_size(ttype, n)
            f.seek(src_data + offset)
            mtp_bytes[name] = f.read(sz)

    # 3) 写输出：header + kv（dst 的 + nextn 元数据）
    arch = None
    for key, t, val in dst_kv:
        if key == b'general.architecture':
            arch = val.decode() if isinstance(val, bytes) else str(val)
    with open(out_path, 'wb') as out:
        out.write(GGUF_MAGIC)
        out.write(struct.pack('<I', 3))
        out.write(struct.pack('<Q', len(dst_tensors) + len(mtp_tensors)))
        # metadata: dst 的全部 + nextn
        extra_kv = []
        if arch:
            extra_kv.append((f'{arch}.nextn_predict_layers'.encode(), 4, 1))
        out.write(struct.pack('<Q', len(dst_kv) + len(extra_kv)))
        for key, t, val in dst_kv:
            write_str(out, key)
            out.write(struct.pack('<I', t))
            if t == 0: out.write(struct.pack('<B', val))
            elif t == 1: out.write(struct.pack('<b', val))
            elif t == 2: out.write(struct.pack('<H', val))
            elif t == 3: out.write(struct.pack('<h', val))
            elif t == 4: out.write(struct.pack('<I', val))
            elif t == 5: out.write(struct.pack('<i', val))
            elif t == 6: out.write(struct.pack('<f', val))
            elif t == 7: out.write(struct.pack('<B', 1 if val else 0))
            elif t == 8: write_str(out, val)
            elif t == 9:
                at, arr = val
                out.write(struct.pack('<I', at))
                out.write(struct.pack('<Q', len(arr)))
                for item in arr:
                    if at == 8:
                        write_str(out, item)
                    else:
                        out.write(struct.pack('<I', item))
        for key, t, val in extra_kv:
            write_str(out, key)
            out.write(struct.pack('<I', t))
            if t == 4: out.write(struct.pack('<I', val))

        # tensor 信息：dst 的（offset 不变，数据区顺序不变）+ MTP（追加）
        for name, dims, ttype, offset in dst_tensors:
            write_str(out, name)
            out.write(struct.pack('<I', len(dims)))
            out.write(struct.pack('<' + 'Q' * len(dims), *dims))
            out.write(struct.pack('<I', ttype))
            out.write(struct.pack('<Q', offset))
        cur = 0
        for name, dims, ttype, offset in mtp_tensors:
            write_str(out, name)
            out.write(struct.pack('<I', len(dims)))
            out.write(struct.pack('<' + 'Q' * len(dims), *dims))
            out.write(struct.pack('<I', ttype))
            out.write(struct.pack('<Q', cur))
            cur += len(mtp_bytes[name])

        # 4) 数据区：dst 的（流式复制）+ MTP 的
        with open(dst_path, 'rb') as df:
            df.seek(dst_data)
            remaining = os.path.getsize(dst_path) - dst_data
            chunk = 64 * 1024 * 1024
            while remaining > 0:
                c = df.read(min(chunk, remaining))
                if not c:
                    break
                out.write(c)
                remaining -= len(c)
        for name, dims, ttype, offset in mtp_tensors:
            out.write(mtp_bytes[name])
    print(f'written {out_path} ({os.path.getsize(out_path)} bytes)')

if __name__ == '__main__':
    main()
