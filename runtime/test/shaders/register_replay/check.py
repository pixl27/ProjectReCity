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


def verify(path, count, seed, killed, shape, nested_stride=None):
    words = struct.unpack("<16384I", path.read_bytes())
    halves, errors = words[:2]
    if errors or not 0 < halves <= 512:
        raise ValueError(f"invalid capture: {halves} halves, {errors} errors")
    # This fixture needs four persistent words (three after the overwrite variant).
    stride = nested_stride or (9 if killed else 10)
    masked_vote = shape.startswith("ballot_")
    tables = 3 if masked_vote else 2
    base = 2 + 512 * tables
    records = [words[base+i*stride:base+i*stride+6] for i in range(count)]
    slots = [r[0] for r in records]
    if max(slots) >= halves*32 or len(set(slots)) != count or set(s//32 for s in slots) != set(range(halves)):
        raise ValueError("invalid GPU lane assignment")
    masks = [[0]*512 for _ in range(tables)]
    for key, slot in enumerate(slots):
        if (((key + seed) & MASK) ^ 37) & 8:
            masks[0][slot//32] |= 1 << (slot % 32)

    def population(table, slot):
        pair = (slot//32) & ~1
        return table[pair].bit_count() + table[pair+1].bit_count()

    if masked_vote:
        for key, slot in enumerate(slots):
            x = (key + seed) & MASK
            target = 3 if shape == "ballot_then" else 0
            if shape != "ballot_empty" and x & 3 == target and x & 4:
                masks[1][slot // 32] |= 1 << (slot % 32)

    outcomes = set()
    wave_paths = {}
    merged = []
    checkpoints = []
    for key, slot in enumerate(slots):
        v4 = (key + seed) & MASK
        v5, s3 = v4 ^ 37, seed
        first = population(masks[0], slot)
        if masked_vote:
            outer, inner = bool(v4 & 1), bool(v4 & 2)
            if shape == "ballot_empty":
                outer = True
            then = shape == "ballot_then"
            active = outer == then and inner == then
            inner_count = population(masks[1], slot)
            if active:
                v4 = (v4 + inner_count) & MASK
            else:
                increment = inner_count if then else (3 if outer else 7)
                v5 = (v5 + increment) & MASK
            outcomes.add((active, bool((key + seed) & 4)))
        elif shape.startswith("nested_exec"):
            outer, inner = bool(v4 & 1), bool(v4 & 2)
            outcomes.add((outer, inner))
            wave_paths.setdefault(slot // 64, set()).add((outer, inner))
            if outer and inner:
                v4 ^= first
                leaf = 1
            elif outer:
                v5 = (v5 + 11) & MASK
                leaf = 2
            elif inner:
                v4 = (v4 + 5) & MASK
                leaf = 4
            else:
                v5 ^= 17
                leaf = 8
            parent_vote = bool((v4 + v5) & 16)
            marker = leaf | (16 if outer else 32) | 64 | (128 if parent_vote else 0)
            checkpoints.append((v4, v5, marker))
            v4 = (v4 + 3) & MASK
        elif shape in ("divergent", "masked_skip"):
            predicate = bool(v4 & 1)
            wave_paths.setdefault(slot // 64, set()).add(predicate)
            outcomes.add(predicate)
            if predicate:
                v4 = ((v4 ^ first) + 1) & MASK
            elif shape == "divergent":
                v5 = (v5 + 17) & MASK
        elif shape == "nested":
            outer, inner = first >= 30, seed == 7
            outcomes.add((outer, inner))
            if outer and inner:
                s3 = 9
            elif outer:
                v4 ^= first
            elif inner:
                s3 = 17
            else:
                v5 = (v5 + 5) & MASK
        elif shape:
            condition = first >= 30
            outcomes.add(condition)
            if (shape == "diamond" and condition) or (shape == "skip" and not condition):
                v4 ^= first
                s3 = 9
            elif shape == "diamond":
                v5 = (v5 + 5) & MASK
        else:
            v4 ^= first
        merged.append((v4, v5, s3))
        if (v4 + v5) & 16:
            masks[-1][slot//32] |= 1 << (slot % 32)
    if list(words[2:base]) != sum(masks, []):
        raise ValueError("ballot bit mismatch")
    for key, (slot, visits, *result) in enumerate(records):
        v4, v5, s3 = merged[key]
        first, second = population(masks[0], slot), population(masks[-1], slot)
        expected = [(v4 + second) & MASK, ((61 if killed else v5) + s3) & MASK,
                    first, second]
        if shape.startswith("nested_exec"):
            inner_v4, inner_v5, marker = checkpoints[key]
            expected[2] = marker
            if shape == "nested_exec_snapshot":
                expected[:2] = [inner_v4, inner_v5]
        if masked_vote:
            expected[2] = population(masks[1], slot)
        if visits != (0x1111 if masked_vote else 0x111) or result != expected:
            raise ValueError(f"key {key}: visits={visits:#x}, got {result}, expected {expected}")
    required_paths = {(a,b) for a in (False, True) for b in (False, True)} if shape.startswith("nested_exec") else {False, True}
    if wave_paths and required_paths not in wave_paths.values():
        raise ValueError("no logical wave took both paths")
    return outcomes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("harness", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--glslang", default="glslangValidator")
    ap.add_argument("--branches", action="store_true", help="exercise forward SCC branches between votes")
    ap.add_argument("--divergent", action="store_true", help="exercise different EXEC paths within the same wave")
    ap.add_argument("--nested-exec", action="store_true", help="check two saved EXEC levels and restore checkpoints")
    ap.add_argument("--masked-ballot", action="store_true", help="check a vote under two EXEC masks")
    args = ap.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    gpu.spirv_as(gpu.VS_TRI_SPVASM, str(work / "triangle.spv"))
    flat = gpu.make_fs_passthru(1).replace("OpDecorate %in_v Location 1",
                                          "OpDecorate %in_v Location 1\nOpDecorate %in_v Flat")
    gpu.spirv_as(flat, str(work / "flat.spv"))
    for stage, extension, export in (("vs", "vert", "param1"), ("ps", "frag", "mrt0")):
        variants = [(False, "diamond"), (False, "skip"), (False, "nested")] if args.branches else [(False, ""), (True, "")]
        if args.divergent:
            variants = [(False, "divergent"), (False, "masked_skip")]
        if args.nested_exec:
            variants = [(False, "nested_exec"), (False, "nested_exec_snapshot")]
        if args.masked_ballot:
            variants = [(False, name) for name in ("ballot_then", "ballot_else", "ballot_empty")]
        for killed, shape in variants:
            reached = set()
            name = f"{stage}_replay_{shape or int(killed)}"
            asm = (HERE / "program.asm").read_text(encoding="ascii")
            asm += f"\nexp {export} v4, v5, v6, v7 done" + (" vm" if stage == "ps" else "") + "\ns_endpgm\n"
            if shape:
                fixture = "divergent" if shape == "masked_skip" else shape
                if shape.startswith("nested_exec"):
                    fixture = "nested_exec"
                    exports = "v20, v21" if shape.endswith("_snapshot") else "v4, v5"
                    asm = asm.replace(f"exp {export} v4, v5, v6, v7", f"exp {export} {exports}, v18, v7")
                if shape.startswith("ballot_"):
                    fixture = "ballot_else" if shape == "ballot_empty" else shape
                    asm = asm.replace("v_mov_b32_e32 v6, s10", "v_mov_b32_e32 v6, s23")
                body = (HERE / (fixture + ".asm")).read_text(encoding="ascii")
                if shape == "ballot_empty":
                    body = body.replace("v_and_b32_e32 v13, 1, v4", "v_mov_b32_e32 v13, 1")
                if shape == "masked_skip":
                    body = body[:body.index("s_andn2_b64")] + body[body.index("s_mov_b64 exec"):]
                asm = asm.replace("v_xor_b32_e32 v4, s10, v4", body)
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
            nested_stride = None
            if shape.startswith(("nested_exec", "ballot_")):
                nested_stride = int(re.search(r"uint base = key \* (\d+)u;", glsl.read_text()).group(1))
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
                    if shape.startswith("ballot_"):
                        env["VKH_BALLOT_PASSES"] = "3"
                    rc, output = gpu.run_cmd([str(args.harness.resolve()), str(vertex), str(fragment),
                                              str(prefix.with_suffix(".raw")), str(w), str(h)], env=env)
                    if rc:
                        raise RuntimeError(output)
                    reached.update(verify(heap, count, seed, killed, shape, nested_stride))
                    print(f"PASS {name} {w}x{h} seed={seed} lanes={count}", flush=True)
            if shape:
                required = {(a,b) for a in (False, True) for b in (False, True)} if shape.startswith("nested") else {False, True}
                if shape.startswith("ballot_"):
                    activity = (False,) if shape == "ballot_empty" else (False, True)
                    required = {(active, pred) for active in activity for pred in (False, True)}
                if reached != required:
                    raise ValueError(f"{name}: missing paths {required - reached}")
                print(f"{name}: {len(reached)} paths verified", flush=True)


if __name__ == "__main__":
    main()
