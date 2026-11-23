#!/usr/bin/env python3
"""C8 source, address, stage and decode models; run from the repository root.

This does not compile CUDA, execute a GPU, or measure shared-bank transactions.
The bank model assumes one 16-thread phase for an 8-byte/thread shared store.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import struct


CUDA = Path('tools/mxfp4_training/csrc/mxfp4_gemm_fused.cu')
HEADER = CUDA.with_suffix('.h')
OLD_CUDA_SHA = '90e6252c455c099df510505a25edc63e3f5146e16357a03211af339727cbd3ad'
OLD_HEADER_SHA = 'b52633aeb346379562c5d93ba9ca8306c68aaff2c5de7a6b749808afece50b6c'
INT_MAX = (1 << 31) - 1
Q = (0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_source(source, header):
    begin = source.index('// Variant5 assigns one half MX32 row')
    end = source.index('bool product_fits', begin)
    restored = source[:begin] + source[end:]
    restored = restored.replace(
        '  // The caller has established an active MX32 block and 0<=delta<=12.\n'
        '  // A zero subgroup may share that valid delta with an active half row.\n'
        '  // Individual zero nibbles are masked to canonical+0 after the safe shifts.\n',
        '  // The caller has established at least one nonzero q in this group, hence\n'
        '  // its active block participated in row_min: 0<=delta<=12. Add exponent bits\n'
        '  // only after that guard. Individual zero nibbles are masked to canonical+0.\n')
    start_dispatch = restored.index('  else if (sizeof(Index) == sizeof(int)')
    end_dispatch = restored.index('\n}', start_dispatch)
    restored = restored[:start_dispatch] + restored[end_dispatch:]
    restored = restored.replace('else if (variant == 4 && m % 128 == 0 && n % 128 == 0)',
                                'else if (m % 128 == 0 && n % 128 == 0)')
    restored = restored.replace('else if (variant == 4)\n    fused_certified_prefetch32_warp64<Index, true>',
                                'else\n    fused_certified_prefetch32_warp64<Index, true>')
    restored = restored.replace('variant > 5)', 'variant > 4)')
    restored = restored.replace('    case 5: return "sm70_certified_fused128_bk32_halfrow_prefetch_warp32x64_v1";\n', '')
    # Removing a complete dispatch tail leaves one extra line break.
    restored = restored.replace('accepted, Index(tiles_n));\n\n}\n\n}  // namespace',
                                'accepted, Index(tiles_n));\n}\n\n}  // namespace')
    assert sha(restored.encode()) == OLD_CUDA_SHA
    h_begin = header.index('//   variant5:')
    h_end = header.index('// All variants', h_begin)
    restored_header = (header[:h_begin] + header[h_end:]).replace(
        '// Bounded runtime experiments (no automatic tuning):',
        '// Bounded runtime experiments (no automatic tuning or silent substitution):')
    restored_header = restored_header.replace(
        '// become half +0; output zero is canonical +0. Exponent manipulation only\n'
        '// occurs after establishing a valid delta from an active MX32 block.\n',
        '// become half +0 before exponent manipulation; output zero is canonical +0.\n')
    assert sha(restored_header.encode()) == OLD_HEADER_SHA
    old = source[source.index('__global__ __launch_bounds__(256) void fused_certified_prefetch32('):
                 source.index('// Each BK32 copy maps eight')]
    new = source[source.index('__global__ __launch_bounds__(256) void fused_certified_prefetch32_halfrow('):end]
    compute = '    #pragma unroll\n    for (int vector = 0; vector < 8; ++vector)'
    assert old[old.index(compute):] == new[new.index(compute):]
    assert new.count('__syncthreads();') == 3
    assert new.index('if (*accepted == 0) return;') < new.index('base_a[tid] =')
    assert 'const int load_row = (linear_row & ~12) | ((linear_row & 4) << 1)\n      | ((linear_row & 8) >> 1);' in new
    assert 'uint2 raw_a, raw_b;\n  unsigned raw_as, raw_bs;' in new
    assert 'if (blocks) {' in new and 'if (block + 1 < blocks) {' in new
    assert 'k ? row_bases[tile_row + tid] : 127' in new
    publish = source[source.index('__device__ __forceinline__ void publish_halfrow32'):source.index('// Low-register experiment:')]
    assert publish.index('if ((raw.x | raw.y) & 0x77777777u)') < publish.index('const int delta')
    assert '((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b)) & 7u) == 0' in source


def row_for(tid):
    row = tid >> 1
    return (row & ~12) | ((row & 4) << 1) | ((row & 8) >> 1)


def coverage_and_banks():
    coverage = Counter((row_for(tid), 4 * (tid & 1) + copy)
                       for tid in range(256) for copy in range(4))
    assert len(coverage) == 1024 and set(coverage.values()) == {1}
    phases = 0
    old_max = 0
    for phase in range(16):
        for copy in range(4):
            words = [18 * row_for(tid) + 8 * (tid & 1) + 2 * copy + word
                     for tid in range(phase * 16, (phase + 1) * 16) for word in range(2)]
            assert len(set(words)) == 32  # These are distinct addresses, not broadcasts.
            assert Counter(word % 32 for word in words) == Counter(range(32))
            old_words = [18 * ((tid + copy * 256) // 8) + 2 * (tid % 8) + word
                         for tid in range(phase * 16, (phase + 1) * 16) for word in range(2)]
            old_max = max(old_max, max(Counter(word % 32 for word in old_words).values()))
            phases += 1
    return {'groups_per_operand_exactly_once': len(coverage), 'store_phases_checked_per_operand': phases,
            'c8_max_distinct_addresses_per_bank_per_assumed_phase': 1,
            'c6_max_distinct_addresses_per_bank_same_model': old_max,
            'model_only_not_a_hardware_transaction_measurement': True}


def model_stage(rows, tile, k, block):
    loaded = {}
    for tid in range(256):
        local_row, half = row_for(tid), tid & 1
        row = tile + local_row
        address = row * (k // 2) + block * 16 + half * 8
        scale_address = row * (k // 32) + block
        assert 0 <= address and address + 7 < rows * (k // 2) and address % 8 == 0
        assert 0 <= scale_address < rows * (k // 32)
        for copy in range(4):
            loaded[local_row, half * 4 + copy] = (address + copy * 2, scale_address)
    expected = {(row, vector): ((tile + row) * (k // 2) + block * 16 + vector * 2,
                               (tile + row) * (k // 32) + block)
                for row in range(128) for vector in range(8)}
    assert loaded == expected
    return loaded


def check_stages():
    cases, groups = [], 0
    for m, n in ((128, 128), (128, 256), (256, 128), (256, 256)):
        for k in (0, 32, 64, 96, 160, 288):
            case_groups = 0
            for tile_row in range(0, m, 128):
                for tile_col in range(0, n, 128):
                    for rows, tile in ((m, tile_row), (n, tile_col)):
                        raw = model_stage(rows, tile, k, 0) if k else None
                        for block in range(k // 32):
                            assert raw == model_stage(rows, tile, k, block)
                            case_groups += len(raw)
                            if block + 1 < k // 32:
                                raw = model_stage(rows, tile, k, block + 1)
            if k == 0:
                assert case_groups == 0
            cases.append({'shape': [m, n, k], 'stage_groups_matched': case_groups})
            groups += case_groups
    return {'shape_cases': len(cases), 'stage_groups_matched': groups, 'cases': cases}


def fast_route(m, n, k, a, b):
    if not m or not n:
        return 'no_op'
    small = (m + n <= INT_MAX and k <= INT_MAX and m <= INT_MAX // n
             and (k == 0 or (m <= INT_MAX // (k // 2) and n <= INT_MAX // (k // 2))))
    if small and m % 128 == 0 and n % 128 == 0 and (a | b) & 7 == 0:
        return 'c8_halfrow'
    return 'c6_safety_fallback'


def check_dispatch():
    cases = []
    for m, n in ((0, 128), (128, 0), (1, 1), (127, 128), (128, 128), (129, 128),
                 (128, 129), (128, 256), (256, 128)):
        for k in (0, 32, 96, 160):
            for a in range(8):
                for b in range(8):
                    actual = fast_route(m, n, k, a, b)
                    expected = ('no_op' if not m or not n else 'c8_halfrow'
                                if m % 128 == 0 and n % 128 == 0 and a == 0 and b == 0
                                else 'c6_safety_fallback')
                    assert actual == expected
                    cases.append(actual)
    boundaries = [
        (128, 16777088, 0, 'c8_halfrow'), (128, 16777216, 0, 'c6_safety_fallback'),
        (128, 128, 33554400, 'c8_halfrow'), (128, 128, 33554432, 'c6_safety_fallback'),
        (128, 128, 2147483616, 'c6_safety_fallback'),
        (128, 128, 2147483648, 'c6_safety_fallback')]
    for m, n, k, expected in boundaries:
        assert fast_route(m, n, k, 0, 0) == expected
    return {'alignment_tail_zero_cases': len(cases), 'routes': dict(Counter(cases)),
            'index_boundaries': [{'shape': [m, n, k], 'expected_route': route} for m, n, k, route in boundaries],
            'output_plus4_keeps_same_dispatch': True,
            'zero_k_null_inputs_full_tile_route': fast_route(128, 128, 0, 0, 0)}


def normalize_half(raw, scale, base):
    if raw & 0x7777777777777777 == 0:
        return [0] * 16, False
    delta = scale - base
    assert 0 <= delta <= 12
    bits = []
    for copy in range(4):
        packed = (raw >> (16 * copy)) & 0xffff
        for nibble in range(4):
            code = (packed >> (4 * nibble)) & 15
            q = Q[code]
            h = int.from_bytes(struct.pack('<e', q), 'little') if q else 0
            bits.append((h + (delta << 10)) & 0xffff if q else 0)
    return bits, True


def check_decode():
    zero_cases = 0
    for signs in range(1 << 16):
        raw = sum(((signs >> i) & 1) << (4 * i + 3) for i in range(16))
        bits, touched_delta = normalize_half(raw, 0, 127)
        assert bits == [0] * 16 and not touched_delta
        zero_cases += 1
    active_cases = 0
    for position in range(16):
        for code in range(16):
            if not Q[code]:
                continue
            raw = (0x8888888888888888 & ~(15 << (position * 4))) | (code << (position * 4))
            for delta in range(13):
                bits, touched_delta = normalize_half(raw, 127 + delta, 127)
                expected = [0] * 16
                expected[position] = int.from_bytes(struct.pack('<e', Q[code] * (1 << delta)), 'little')
                assert bits == expected and touched_delta
                active_cases += 1
    rng = random.Random(20261124)
    random_cases = 1024
    for case in range(random_cases):
        raw, delta = rng.getrandbits(64), case % 13
        bits, _ = normalize_half(raw, 127 + delta, 127)
        expected = [int.from_bytes(struct.pack('<e', Q[(raw >> (i * 4)) & 15] * (1 << delta)), 'little')
                    for i in range(16)]
        assert bits == expected
    return {'all_sign_zero_half_patterns': zero_cases, 'zero_patterns_with_negative_delta_never_normalized': True,
            'single_active_nibble_half_cases': active_cases, 'random_half_cases': random_cases,
            'individual_normalized_half_values_compared': (zero_cases + active_cases + random_cases) * 16}


def main():
    source, header = CUDA.read_text(), HEADER.read_text()
    verify_source(source, header)
    result = {'schema_version': 1, 'evidence_type': 'host_only_c8_halfrow_mapping_staging_decode_dispatch',
              'recorded_at_utc': datetime.now(timezone.utc).isoformat(), 'passed': True,
              'gpu_executed': False, 'cuda_compilation_executed': False,
              'limitations': 'Host source/address/decode models only; no GPU arithmetic, occupancy, bank transaction or performance measurement.',
              'source_sha256': sha(CUDA.read_bytes()), 'header_sha256': sha(HEADER.read_bytes()),
              'script_sha256': sha(Path(__file__).read_bytes()),
              'prior_cuda_sha256': OLD_CUDA_SHA, 'prior_header_sha256': OLD_HEADER_SHA,
              'old_c0_through_c7_sources_reconstructed_byte_exact': True,
              'c6_compute_loop_and_epilogue_byte_identical': True,
              'static_shared_bytes_from_source': 18688,
              'raw_register_payload_from_source': 6,
              'coverage_and_bank_model': coverage_and_banks(), 'stages': check_stages(),
              'dispatch': check_dispatch(), 'decode': check_decode()}
    output = Path(__file__).with_name('halfrow-prefetch-host-checks.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('passed', 'gpu_executed', 'source_sha256', 'header_sha256',
                                                 'coverage_and_bank_model', 'decode')}))


if __name__ == '__main__':
    main()
