"""Assemble RDNA2 inputs, compile through the replay CLI, and check register results on the GPU."""
import argparse
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests as gpu

MASK = 2**32 - 1


def verify(path, count, seed, killed):
    words = struct.unpack("<16384I", path.read_bytes())
    halves, errors = words[:2]
    if errors or not 0 < halves <= 512:
        raise ValueError(f"invalid capture: {halves} halves, {errors} errors")
    # This fixture needs four persistent words (three after the overwrite variant).
    stride = 9 if killed else 10
    records = [words[1026+i*stride:1026+i*stride+6] for i in range(count)]
    slots = [r[0] for r in records]
    if max(slots) >= halves*32 or len(set(slots)) != count or set(s//32 for s in slots) != set(range(halves)):
        raise ValueError("invalid GPU lane assignment")
    masks = [[0]*512 for _ in range(2)]
    for key, slot in enumerate(slots):
        if (((key + seed) & MASK) ^ 37) & 8:
            masks[0][slot//32] |= 1 << (slot % 32)

    def population(table, slot):
        pair = (slot//32) & ~1
        return table[pair].bit_count() + table[pair+1].bit_count()

    for key, slot in enumerate(slots):
        start = (key + seed) & MASK
        first = population(masks[0], slot)
        if ((start ^ 37) + (start ^ first)) & 16:
            masks[1][slot//32] |= 1 << (slot % 32)
    if list(words[2:1026]) != masks[0] + masks[1]:
        raise ValueError("ballot bit mismatch")
    for key, (slot, visits, *result) in enumerate(records):
        start = (key + seed) & MASK
        first, second = (population(table, slot) for table in masks)
        expected = [((start ^ first) + second) & MASK, ((61 if killed else start ^ 37) + seed) & MASK,
                    first, second]
        if visits != 0x111 or result != expected:
            raise ValueError(f"key {key}: visits={visits:#x}, got {result}, expected {expected}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("harness", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--glslang", default="glslangValidator")
    args = ap.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    gpu.spirv_as(gpu.VS_TRI_SPVASM, str(work / "triangle.spv"))
    flat = gpu.make_fs_passthru(1).replace("OpDecorate %in_v Location 1",
                                          "OpDecorate %in_v Location 1\nOpDecorate %in_v Flat")
    gpu.spirv_as(flat, str(work / "flat.spv"))
    for stage, extension, export in (("vs", "vert", "param1"), ("ps", "frag", "mrt0")):
        for killed in (False, True):
            name = f"{stage}_replay_{int(killed)}"
            asm = (HERE / "program.asm").read_text(encoding="ascii")
            asm += f"\nexp {export} v4, v5, v6, v7 done" + (" vm" if stage == "ps" else "") + "\ns_endpgm\n"
            if killed:
                asm = asm.replace("v_add_nc_u32_e32 v5, s3, v5", "v_mov_b32_e32 v5, 61\nv_add_nc_u32_e32 v5, s3, v5")
                for old, new in (("s3", "s17"), ("s10", "s21"), ("v4", "v19")):
                    asm = re.sub(rf"\b{old}\b", new, asm)
            rawasm, disasm = work / (name + ".asm"), work / (name + ".s")
            rawasm.write_text(asm, encoding="ascii")
            gpu.assemble_to_s(os.environ.get("LLVM_MC", "llvm-mc"), str(rawasm), str(disasm), name)
            glsl, module = work / (name + ".glsl"), work / (name + ".spv")
            subprocess.run([sys.executable, "-I", ROOT / "tools/rdna2spirv.py", disasm,
                            "--graphics-replay-glsl", "--out", glsl], check=True)
            subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.2", "-S", extension,
                            "-o", module, glsl], check=True)
            subprocess.run(["spirv-val", "--target-env", "vulkan1.2", module], check=True)
            vertex, fragment = (module, work / "flat.spv") if stage == "vs" else (work / "triangle.spv", module)
            for w,h in ([(32,32)] if stage == "vs" else [(32,32),(29,23),(5,3)]):
                count = 192 if stage == "vs" else w*h
                for seed in (7, 12345, 0xfffffff1):
                    prefix = work / f"{name}-{w}x{h}-{seed}"
                    heap = prefix.with_suffix(".heap")
                    heap.unlink(missing_ok=True)
                    env = {k:v for k,v in os.environ.items() if not k.startswith("VKH_")}
                    env.update(VKH_BALLOT_REPLAY_STAGE=stage, VKH_HEAP="0=0", VKH_HEAP_DUMP=str(heap),
                               VKH_SGPRS=f"0,0,{count},{w},{seed}")
                    rc, output = gpu.run_cmd([str(args.harness.resolve()), str(vertex), str(fragment),
                                              str(prefix.with_suffix(".raw")), str(w), str(h)], env=env)
                    if rc:
                        raise RuntimeError(output)
                    verify(heap, count, seed, killed)
                    print(f"PASS {name} {w}x{h} seed={seed} lanes={count}", flush=True)


if __name__ == "__main__":
    main()
