"""Validate GPU-assigned logical waves and two dependent votes, including each saved lane."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "scripts"))
import run_gpu_tests as gpu

MASK = (1 << 32) - 1
CAPACITY = 512


def verify(heap, count, seed):
    data = heap.read_bytes()
    words = struct.unpack("<16384I", data)
    halves, errors, moved = words[:3]
    if not (0 < halves <= CAPACITY and errors == 0):
        raise ValueError(f"allocation or width error: halves={halves}, errors={errors}")
    record_offset = (4 + 3 * CAPACITY) * 4
    records = list(struct.iter_unpack("<4I", data[record_offset:record_offset + count * 16]))
    occupied, first, second = ([0] * CAPACITY for _ in range(3))
    values = []
    for key, (slot, _, _, visits) in enumerate(records):
        half, lane = divmod(slot, 32)
        if half >= halves or occupied[half] & (1 << lane) or visits != 0x111:
            raise ValueError(f"invalid assignment or missing pass for key {key}, slot {slot}, visits {visits:#x}")
        x = (key * 747796405 + seed) & MASK
        value = ((x ^ (x >> 16)) * 2246822519) & MASK
        values.append(value)
        occupied[half] |= 1 << lane
        if value & 8:
            first[half] |= 1 << lane
    if not all(occupied[:halves]):
        raise ValueError("allocated empty half")
    for key, (slot, saved, _, _) in enumerate(records):
        pair = (slot // 32) & ~1
        pop = first[pair].bit_count() + first[pair + 1].bit_count()
        value = values[key] ^ (((pop + 1) * 0x9e3779b9 + slot % 64) & MASK)
        if saved != value:
            raise ValueError(f"live register mismatch for key {key}")
        if value & 16:
            second[slot // 32] |= 1 << (slot % 32)
    for i, expected in enumerate((occupied, first, second)):
        start = 4 + i * CAPACITY
        if list(words[start:start + CAPACITY]) != expected:
            raise ValueError(f"GPU mask table {i} differs from CPU oracle")
    for key, (slot, saved, final, _) in enumerate(records):
        pair = (slot // 32) & ~1
        pop = second[pair].bit_count() + second[pair + 1].bit_count()
        if final != (saved + pop * 17) & MASK:
            raise ValueError(f"second vote result mismatch for key {key}")
    print(f"{count} lanes, {halves} physical halves, {(halves+1)//2} logical waves, "
          f"{sum(word != MASK for word in occupied[:halves])} partial halves, "
          f"{moved} keys resumed on a different lane", flush=True)
    return moved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--glslang", default="glslangValidator")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    gpu.spirv_as(gpu.VS_TRI_SPVASM, str(work / "triangle.spv"))
    flat = gpu.make_fs_passthru(1).replace("OpDecorate %in_v Location 1",
                                          "OpDecorate %in_v Location 1\nOpDecorate %in_v Flat")
    gpu.spirv_as(flat, str(work / "flat.spv"))
    for stage, shader_type in (("vs", "vert"), ("ps", "frag")):
        module = work / f"{stage}.spv"
        defines = ["-DVERTEX_STAGE=1"] if stage == "vs" else []
        subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.2", "-S", shader_type,
                        *defines, "-o", module, HERE / "replay.glsl"], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", module], check=True)
        vertex, fragment = (module, work / "flat.spv") if stage == "vs" else (work / "triangle.spv", module)
        moved_total = 0
        for width, height in ([(32,32)] if stage == "vs" else [(32,32), (29,23), (5,3)]):
            count = 192 if stage == "vs" else width * height
            for seed in (7, 12345, 0xfeed):
                prefix = work / f"{stage}-{width}x{height}-{seed}"
                heap = prefix.with_suffix(".heap")
                heap.unlink(missing_ok=True)
                env = {k:v for k,v in os.environ.items() if not k.startswith("VKH_")}
                env.update(VKH_BALLOT_REPLAY_STAGE=stage, VKH_HEAP="0=0", VKH_HEAP_DUMP=str(heap),
                           VKH_SGPRS=f"0,0,{count},{width},{seed}")
                rc, output = gpu.run_cmd([str(args.harness.resolve()), str(vertex), str(fragment),
                                          str(prefix.with_suffix(".raw")), str(width), str(height)], env=env)
                if rc:
                    raise RuntimeError(f"{stage} seed {seed}: exit {rc}\n{output}")
                print(f"{stage} seed={seed}:", end=" ")
                moved_total += verify(heap, count, seed)
        if not moved_total:
            raise ValueError(f"{stage}: replay never changed a physical lane")


if __name__ == "__main__":
    main()
