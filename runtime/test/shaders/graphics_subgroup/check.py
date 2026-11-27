"""Measure VS/PS/GS subgroup width and compare a synthetic translated wave64 ballot to its expected counts."""

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import struct
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests as gpu  # noqa: E402
import rdna2spirv  # noqa: E402


OUTPUT = {"vs": "%out_param1", "ps": "%out_mrt0", "gs": "%gs_out1"}


def with_width(asm, stage):
    """Replace the synthetic marker channel with the physical width, leaving the ballot untouched."""
    out = OUTPUT[stage]
    asm = re.sub(r'(?m)^(\s*OpEntryPoint .*"main".*)$', r"\1 %probe_size", asm, count=1)
    decorations = "OpDecorate %probe_size BuiltIn SubgroupSize\n"
    decorations += f"OpDecorate {out} Flat\n" if stage != "ps" else "OpDecorate %probe_size Flat\n"
    asm = asm.replace("OpDecorate %sg_lane BuiltIn", decorations + "OpDecorate %sg_lane BuiltIn", 1)
    asm = asm.replace("%sg_lane = OpVariable", "%probe_size = OpVariable %_ptr_Input_uint Input\n%sg_lane = OpVariable", 1)
    count = 0

    def export(match):
        nonlocal count
        count += 1
        suffix = str(count)
        value = match[1]
        return (f"%size{suffix} = OpLoad %uint %probe_size\n"
                f"%size_f{suffix} = OpBitcast %float %size{suffix}\n"
                f"%result{suffix} = OpCompositeInsert %v4float %size_f{suffix} {value} 3\n"
                f"OpStore {out} %result{suffix}")

    asm = re.sub(rf"(?m)^[ \t]*OpStore {out} (%\w+)$", export, asm)
    if count != (3 if stage == "gs" else 1):
        raise RuntimeError(f"{stage}: expected synthetic output stores, found {count}")
    return asm


def build(work, llvm_mc):
    disasm = work / "disasm"
    disasm.mkdir(parents=True, exist_ok=True)
    for stage in OUTPUT:
        name = stage + "_wave64_ballot"
        path = disasm / (name + ".s")
        signature = gpu.make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])
        chunks = [(b"ISG1", signature), (b"OSG1", signature)] if stage == "gs" else []
        (work / (name + ".dxbc")).write_bytes(gpu.make_dxbc(chunks))
        gpu.assemble_to_s(llvm_mc, str(HERE / (name + ".asm")), str(path), name)
        gpu.spirv_as(with_width(rdna2spirv.translate(str(path)), stage), str(work / (stage + ".spv")))
    gpu.spirv_as(gpu.VS_TRI_SPVASM, str(work / "triangle.spv"))
    flat = gpu.make_fs_passthru(1).replace("OpDecorate %in_v Location 1", "OpDecorate %in_v Location 1\nOpDecorate %in_v Flat")
    gpu.spirv_as(flat, str(work / "readback.spv"))
    gpu.spirv_as((HERE / "gs_reference.spv.txt").read_text(encoding="ascii"), str(work / "gs-reference.spv"))
    for name in (*OUTPUT, "triangle", "readback", "gs-reference"):
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", work / (name + ".spv")], check=True)


def draw(harness, work, stage, varying, *, reference=False):
    env = dict(os.environ)
    for key in ("VKH_SUBGROUP_SIZE", "VKH_VARYING_SUBGROUP_STAGE", "VKH_DRAW_COUNTS", "VKH_GS", "VKH_MESH"):
        env.pop(key, None)
    if varying:
        env["VKH_VARYING_SUBGROUP_STAGE"] = stage
    if stage != "ps":
        env["VKH_DRAW_COUNTS"] = "6,64"
    if stage == "gs":
        env["VKH_GS"] = str(work / ("gs-reference.spv" if reference else "gs.spv"))
    vertex = "triangle.spv" if stage == "ps" else "vs.spv"
    fragment = "ps.spv" if stage == "ps" else "readback.spv"
    label = f"{stage}{'-reference' if reference else ''}-{'varying' if varying else 'default'}"
    raw = work / (label + ".raw")
    raw.unlink(missing_ok=True)
    rc, output = gpu.run_cmd([str(harness), str(work / vertex), str(work / fragment), str(raw)], env=env)
    print(f"{label}\n{output}", flush=True)
    if rc == gpu.UNSUPPORTED:
        print("  SKIP (Vulkan features unavailable)")
        return False
    if rc:
        raise RuntimeError(f"{stage}: harness exited {rc}")
    pixels = list(struct.iter_unpack("<4I", raw.read_bytes()))
    if len(pixels) != 4096:
        raise RuntimeError(f"{stage}: read {len(pixels)} of 4096 pixels")
    widths = {width for _, _, _, width in pixels}
    if 0 in widths:
        raise RuntimeError(f"{stage}: the draw left pixels without a subgroup result")
    if widths - {32, 64}:
        print(f"  SKIP (observed widths {sorted(widths)} outside the 32/64 test)")
        return False
    for lane, upper, total, size in pixels:
        if not (0 <= lane < 64 and 1 <= total <= 64 and max(0, total - 32) <= upper <= min(32, total)):
            raise RuntimeError(f"{stage}: invalid ballot (lane, upper, total, width) = {(lane, upper, total, size)}")
    histogram = Counter((size, upper, total) for lane, upper, total, size in pixels)
    lanes = [pixel[0] for pixel in pixels]
    print(f"  Observed lanes {min(lanes)}..{max(lanes)}")
    print(f"  (reported width, upper votes, total votes) across 4096 pixels: {dict(sorted(histogram.items()))}")
    inconsistent = sum(lane >= size or total > size or (size == 32 and upper != 0)
                       for lane, upper, total, size in pixels)
    exact = sum(upper == 32 and total == 64 for _, upper, total, _ in pixels)
    if exact:
        print(f"  {exact} pixels match the full wave64 ballot result (32 upper, 64 total).")
    if inconsistent:
        print(f"  INCONCLUSIVE: {inconsistent} pixels have lane IDs or votes inconsistent with SubgroupSize.")
    elif histogram[(32, 0, 32)]:
        print(f"  {histogram[(32, 0, 32)]} pixels demonstrate the missing upper half on a full physical wave32.")
    return bool(inconsistent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--llvm-mc", default=os.environ.get("LLVM_MC", "llvm-mc"))
    parser.add_argument("--strict", action="store_true", help="exit nonzero for contradictory subgroup measurements")
    args = parser.parse_args()
    work, harness = args.work.resolve(), args.harness.resolve()
    build(work, args.llvm_mc)
    contradictions = False
    for stage in OUTPUT:
        for varying in (False, True):
            inconsistent = draw(harness, work, stage, varying)
            contradictions |= inconsistent
            if inconsistent and stage == "gs" and draw(harness, work, stage, varying, reference=True):
                print("  The direct Vulkan geometry control also reports contradictory subgroup data.")
    return int(args.strict and contradictions)


if __name__ == "__main__":
    sys.exit(main())
