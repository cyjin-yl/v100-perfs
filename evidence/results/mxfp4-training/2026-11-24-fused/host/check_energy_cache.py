import json, random, re
from pathlib import Path
src = Path('tools/mxfp4_training/csrc/mxfp4_gemm_certified.cu').read_text()
header = Path('tools/mxfp4_training/csrc/mxfp4_gemm_certified.h').read_text()
CAP = int(re.search(r'MXFP4_CERTIFIED_ENERGY_CAP UINT64_C\((\d+)\)', header).group(1))
BOUND = CAP - 1
helper = re.search(r'unsigned packed_word_q_energy\(unsigned packed\) \{(.*?)\n\}', src, re.S).group(1)
constants = re.findall(r'__byte_perm\((0x[0-9a-f]+)u, (0x[0-9a-f]+)u,', helper)
assert constants == [('0x03020100','0x0c080604')]*2
lut = bytes.fromhex('000102030406080c')
q = (0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12)
energy4 = []
for packed in range(65536):
    selector = packed & 0x7777
    magnitudes = [lut[(selector >> (4*i)) & 7] for i in range(4)]
    device_model = sum(v*v for v in magnitudes)
    reference = sum(q[(packed >> (4*i)) & 15]**2 for i in range(4))
    assert device_model == reference
    energy4.append(device_model)

def result_metadata(energy, nan_count, minimum, maximum, active):
    base = minimum if minimum != 255 else 127
    span = maximum - minimum if minimum != 255 else 0
    return (energy, nan_count, base, span, not active, energy == CAP)

def original(packed, scales):
    minimum, maximum, active = 255, 0, False
    for b,s in enumerate(scales):
        nz = any(v & 0x77 for v in packed[b*16:b*16+16])
        active |= nz
        if nz and s != 255:
            minimum, maximum = min(minimum,s), max(maximum,s)
    base = minimum if minimum != 255 else 127
    # Original kernel gives every four-code group to group%128, then reduces.
    lanes = [0]*128
    for g in range(len(packed)//2):
        v = packed[2*g] | (packed[2*g+1] << 8)
        e, s = energy4[v], scales[g//8]
        if e and s != 255:
            d = s-base
            assert d >= 0
            term = CAP if d > 24 else e << (2*d)
            lanes[g%128] = min(CAP,lanes[g%128]+term)
    return result_metadata(min(CAP,sum(lanes)),scales.count(255),minimum,maximum,active)

def cached(packed, scales):
    minimum, maximum, active = 255, 0, False
    cache = []
    for b,s in enumerate(scales):
        block = packed[b*16:b*16+16]
        words = [int.from_bytes(block[i:i+4], 'little') for i in range(0,16,4)]
        partials = [energy4[w&65535]+energy4[w>>16] for w in words]
        e = (partials[0]+partials[1])+(partials[2]+partials[3])
        assert 0 <= e <= 4608
        cache.append((s<<16)|e)
        active |= e != 0
        if e and s != 255:
            minimum, maximum = min(minimum,s), max(maximum,s)
    base = minimum if minimum != 255 else 127
    lanes = [0]*128
    for b,v in enumerate(cache):
        e, s = v & 65535, v >> 16
        if e and s != 255:
            d = s-base
            assert d >= 0
            term = CAP if d > 24 else e << (2*d)
            assert term < (1<<61)
            lanes[b%128] = min(CAP,lanes[b%128]+term)
    return result_metadata(min(CAP,sum(lanes)),scales.count(255),minimum,maximum,active)

def oracle(packed, scales):
    raw = [sum(q[v&15]**2+q[v>>4]**2 for v in packed[b*16:b*16+16]) for b in range(len(scales))]
    active_scales = [s for s,e in zip(scales,raw) if e and s != 255]
    minimum, maximum = (min(active_scales),max(active_scales)) if active_scales else (255,0)
    base = minimum if minimum != 255 else 127
    # Arbitrary-precision total: no per-group shifts/clamps or lane reduction.
    exact = sum(e*(4**(s-base)) for s,e in zip(scales,raw) if e and s != 255)
    return result_metadata(min(CAP,exact),scales.count(255),minimum,maximum,any(raw))

def block(codes):
    codes = list(codes)+[0]*(32-len(codes))
    return bytes(codes[i] | codes[i+1]<<4 for i in range(0,32,2))

cases = []
def add(name, blocks, scales):
    cases.append((name,b''.join(blocks),list(scales)))
add('empty',[],[])
for s in range(256):
    add(f'zero_scale_{s}',[bytes([0x88]*16)],[s])
    add(f'nonzero_scale_{s}',[block([1,15,6,0,8])],[s])
for d in range(255):
    add(f'delta_{d}',[block([1]),block([1,15,7]),bytes([0x88]*16)],[0,d,0])
for power in (12,24):
    for adjustment in (-1,0,1):
        blocks = [block([1]*(4+adjustment))] + [block([1]*3) for _ in range(1,power)]
        add(f'power_{2*power}_{adjustment:+}',blocks,range(power))
        expected = min(CAP,(1<<(2*power))+adjustment)
        assert oracle(cases[-1][1],cases[-1][2])[0] == expected
for count in (1,63,64,127,128,129,1023,1024,1025,8192,8193):
    add(f'all_twelve_{count}',[bytes([0xff]*16)]*count,[128]*count)
    add(f'zero_low_scale_{count}',[bytes([0x88]*16)]*count+[block([1])],[0]*count+[128])
# Exact U=2^24 and U=2^24+1 fixtures also exercise the long-K fallback shape.
add('long_energy_bound',[bytes([0x66]*16)]*8192,[128]*8192)
add('long_energy_above',[bytes([0x66]*16)]*8192+[block([1])],[128]*8193)
assert oracle(cases[-2][1],cases[-2][2])[0] == (1<<24)
assert oracle(cases[-1][1],cases[-1][2])[0] == (1<<24)+1
rng = random.Random(0x7a11cace)
for i in range(6000):
    count = rng.choice((0,1,2,3,7,31,63,64,65,127,128,129,257))
    packed = bytearray(rng.randbytes(16*count))
    mode = i%6
    if mode == 0:
        base = rng.randrange(243)
        scales = [base+rng.randrange(13) for _ in range(count)]
    elif mode == 1:
        scales = [rng.randrange(256) for _ in range(count)]
    elif mode == 2:
        scales = [128]*count
    elif mode == 3:
        scales = [rng.choice((0,1,24,25,127,128,254,255)) for _ in range(count)]
    elif mode == 4:
        packed[:] = bytes([0x88])*len(packed)
        scales = [rng.randrange(256) for _ in range(count)]
    else:
        scales = [255]*count
    for b in range(count):
        if rng.randrange(8)==0:
            packed[b*16:b*16+16] = bytes([0x88])*16
            scales[b] = rng.choice((0,1,255))
    cases.append((f'random_{i}',bytes(packed),scales))
for name,packed,scales in cases:
    a,b,c = original(packed,scales),cached(packed,scales),oracle(packed,scales)
    assert a == b == c, (name,a,b,c)
# Byte offsets cover the loader's aligned and byte-safe branches' address sets.
for offset in range(16):
    raw = rng.randbytes(16)
    storage = b'\xa5'*offset+raw+b'\x5a'*16
    words = [sum(storage[offset+i+j]<<(8*j) for j in range(4)) for i in range(0,16,4)]
    assert words == [int.from_bytes(raw[i:i+4], 'little') for i in range(0,16,4)]
print(json.dumps({'host_only':True,'four_nibble_patterns':65536,'row_cases':len(cases),'metadata_fields_compared':['energy','nan_scales','min_scale','span','zero','saturated'],'byte_offsets':16,'max_blocks':max(len(s) for _,_,s in cases),'energy_cap':CAP,'passed':True}))
