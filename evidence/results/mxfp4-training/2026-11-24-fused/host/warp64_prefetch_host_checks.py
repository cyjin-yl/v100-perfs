#!/usr/bin/env python3
"""Host geometry/staging checks for C7; run from the repository root."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path('.')
CUDA = ROOT / 'tools/mxfp4_training/csrc/mxfp4_gemm_fused.cu'
HEADER = CUDA.with_suffix('.h')
OLD_CUDA_SHA = 'da23760c6fbc2141dd77060c3cbac4ad7d37fcafb09b604d2de96017fbe82923'
OLD_HEADER_SHA = '71787df5949fd16d5de3dcc02e014119489f685ca7620286a4cd9d84f8041229'


def verify_old_sources_unchanged(source, header):
    begin = source.index('// Each BK32 copy maps eight')
    end = source.index('bool product_fits')
    restored = source[:begin] + source[end:]
    restored = restored.replace('else if (variant == 3 && m % 128 == 0 && n % 128 == 0)',
                                'else if (m % 128 == 0 && n % 128 == 0)')
    restored = restored.replace('else if (variant == 3)\n    fused_certified_prefetch32<Index, true>',
                                'else\n    fused_certified_prefetch32<Index, true>')
    new_dispatch = '''  else if (m % 128 == 0 && n % 128 == 0)
    fused_certified_prefetch32_warp64<Index, false><<<grid, 128, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
  else
    fused_certified_prefetch32_warp64<Index, true><<<grid, 128, 0, stream>>>(
        a, as, b, bs, out, Index(m), Index(n), Index(k), bases, accepted, Index(tiles_n));
'''
    assert new_dispatch in restored
    restored = restored.replace(new_dispatch, '')
    restored = restored.replace('variant > 4)', 'variant > 3)')
    restored = restored.replace('    case 4: return "sm70_certified_fused128_bk32_prefetch_warp64x64_v1";\n', '')
    assert hashlib.sha256(restored.encode()).hexdigest() == OLD_CUDA_SHA
    old_header = header.replace('//   variant4: CTA128x128, BK32, 128 threads, warp64x64, raw register prefetch.\n'
                                '//             128 FP32 accumulators/thread, same shared stage and barriers.\n', '')
    assert hashlib.sha256(old_header.encode()).hexdigest() == OLD_HEADER_SHA
    new_kernel = source[source.index('__global__ __launch_bounds__(128) void fused_certified_prefetch32_warp64'):end]
    assert new_kernel.count('__syncthreads();') == 3  # Row-base prologue plus two per stage.
    assert 'float accum[4][4][8]' in new_kernel and 'uint2 raw_a[8], raw_b[8]' in new_kernel
    assert new_kernel.index('if (*accepted == 0) return;') < new_kernel.index('base_a[tid] =')
    old_epilogue = source[source.index('  const bool vector_output =', source.index('void fused_certified_prefetch32(')):begin]
    new_epilogue = new_kernel[new_kernel.index('  const bool vector_output ='):]
    assert old_epilogue.replace('fr < 2', 'fr < 4') == new_epilogue


def output_pairs():
    result = []
    writes = Counter()
    for tid in range(128):
        lane, warp = tid & 31, tid >> 5
        warp_row, warp_col = (warp // 2) * 64, (warp % 2) * 64
        group = (lane >> 2) & 3
        for fr in range(4):
            for fc in range(4):
                for pair in range(4):
                    i = pair * 2
                    row = warp_row + fr * 16 + (group >> 1) * 8 + (lane & 1) + (i & 2) + (lane >> 4) * 4
                    col = warp_col + fc * 16 + (group & 1) * 8 + (i & 4) + (lane & 2) + (i & 1)
                    assert 0 <= row < 128 and 0 <= col < 127 and col % 2 == 0
                    writes[row, col] += 1
                    writes[row, col + 1] += 1
                    result.append((row, col))
    assert len(writes) == 16384 and set(writes.values()) == {1}
    return result


def stage(rows, tile, k, block):
    loaded = {}
    for tid in range(128):
        for copy in range(8):
            index = tid + copy * 128
            local_row, vector = index // 8, index % 8
            assert (local_row, vector) not in loaded
            row = tile + local_row
            value = None
            if row < rows and block < k // 32:
                address = row * (k // 2) + block * 16 + vector * 2
                scale_address = row * (k // 32) + block
                assert 0 <= address and address + 1 < rows * (k // 2)
                assert 0 <= scale_address < rows * (k // 32)
                value = (address, scale_address)
            loaded[local_row, vector] = value
    assert len(loaded) == 1024
    return loaded


def direct_stage(rows, tile, k, block):
    return {(local_row, vector): ((tile + local_row) * (k // 2) + block * 16 + vector * 2,
                                  (tile + local_row) * (k // 32) + block)
            if tile + local_row < rows else None
            for local_row in range(128) for vector in range(8)}


def main():
    source, header = CUDA.read_text(), HEADER.read_text()
    verify_old_sources_unchanged(source, header)
    pairs = output_pairs()
    cases = []
    for m, n in [(0, 128), (128, 0), (1, 1), (17, 19), (128, 128), (129, 131), (131, 129)]:
        for k in (0, 32, 64, 96, 192, 288):
            writes, comparisons, valid_loads = Counter(), 0, 0
            for tm in range((m + 127) // 128):
                for tn in range((n + 127) // 128):
                    for rows, tile in ((m, tm * 128), (n, tn * 128)):
                        raw = stage(rows, tile, k, 0) if k else None
                        valid_loads += sum(value is not None for value in raw.values()) if raw else 0
                        for block in range(k // 32):
                            assert raw == direct_stage(rows, tile, k, block)
                            comparisons += len(raw)
                            if block + 1 < k // 32:
                                raw = stage(rows, tile, k, block + 1)
                                valid_loads += sum(value is not None for value in raw.values())
                    for local_row, local_col in pairs:
                        row, col = tm * 128 + local_row, tn * 128 + local_col
                        if row < m and col < n:
                            writes[row, col] += 1
                            if col + 1 < n:
                                writes[row, col + 1] += 1
                                for output_offset in (0, 4):
                                    vector_output = output_offset % 8 == 0 and n % 2 == 0
                                    if vector_output:
                                        assert (output_offset + 4 * (row * n + col)) % 8 == 0
            assert len(writes) == m * n and (not writes or set(writes.values()) == {1})
            if k == 0 or m == 0 or n == 0:
                assert valid_loads == 0
            cases.append({'shape': [m, n, k], 'stage_group_comparisons': comparisons,
                          'valid_raw_groups_loaded': valid_loads, 'output_elements_covered_once': len(writes)})
    result = {'schema_version': 1, 'evidence_type': 'host_only_warp64_prefetch_mapping_and_staging',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'gpu_executed': False, 'cuda_compilation_executed': False,
              'limitations': 'Host index/state model and source preservation checks; no GPU arithmetic or performance claim.',
              'source_sha256': hashlib.sha256(CUDA.read_bytes()).hexdigest(),
              'header_sha256': hashlib.sha256(HEADER.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'prior_cuda_sha256': OLD_CUDA_SHA, 'prior_header_sha256': OLD_HEADER_SHA,
              'old_kernels_helpers_and_behavior_reconstructed_byte_exact': True,
              'new_epilogue_matches_c6_with_four_row_fragments': True,
              'tile_output_elements_covered_once': 16384, 'output_pairs_checked': len(pairs),
              'static_shared_bytes_from_source': (128 + 128) * 9 * 8 + 256,
              'shape_cases': len(cases),
              'stage_group_comparisons': sum(item['stage_group_comparisons'] for item in cases),
              'valid_raw_groups_loaded': sum(item['valid_raw_groups_loaded'] for item in cases), 'cases': cases}
    output = Path(__file__).with_name('warp64-prefetch-host-checks.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('passed', 'gpu_executed', 'shape_cases',
          'stage_group_comparisons', 'valid_raw_groups_loaded', 'source_sha256', 'header_sha256')}))


if __name__ == '__main__':
    main()
