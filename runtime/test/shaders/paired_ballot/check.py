"""Compare a pair of captured graphics wave32 ballots with a native wave64 and a CPU bitmask."""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "scripts"))
import run_gpu_tests as gpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("harness", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--glslang", default="glslangValidator")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    gpu.spirv_as(gpu.VS_TRI_SPVASM, str(work / "triangle.spv"))
    fragment = gpu.make_fs_passthru(1).replace("OpDecorate %in_v Location 1",
                                              "OpDecorate %in_v Location 1\nOpDecorate %in_v Flat")
    gpu.spirv_as(fragment, str(work / "flat.spv"))
    cases = {
        "none": 0, "all": (1 << 64) - 1, "lower": (1 << 32) - 1,
        "upper": 0xffffffff00000000, "top16": 0xffff000000000000,
        "boundary": (1 << 31) | (1 << 32), "sparse": 0x80a0400c01041009,
    }
    for stage, glsl_stage in (("vs", "vert"), ("ps", "frag")):
        shader = work / (stage + ".spv")
        defines = ["-DVERTEX_STAGE=1"] if stage == "vs" else []
        subprocess.run([args.glslang, "-V", "--target-env", "vulkan1.2", "-S", glsl_stage,
                        *defines, "-o", shader, HERE / "paired.glsl"], check=True)
        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", shader], check=True)
        vertex, fragment = (shader, work / "flat.spv") if stage == "vs" else (work / "triangle.spv", shader)
        for name, mask in cases.items():
            expected = (mask & 65535, (mask >> 16) & 65535, (mask >> 32) & 65535,
                        (mask >> 48) + mask.bit_count() * 65536)
            reference = None
            for width in (64, 32):
                env = {k: v for k, v in os.environ.items() if not k.startswith("VKH_")}
                env["VKH_BALLOT_REPLAY_STAGE"] = stage
                env["VKH_SGPRS"] = "0,1" if width == 64 else "0,0"
                env["VKH_HEAP"] = ",".join(f"{lane*4}={(mask >> lane) & 1}" for lane in range(64))
                raw = work / f"{stage}-{name}-{width}.raw"
                raw.unlink(missing_ok=True)
                rc, output = gpu.run_cmd([str(args.harness.resolve()), str(vertex), str(fragment), str(raw)], env=env)
                if rc:
                    raise RuntimeError(f"{stage} {name} width={width}: exit {rc}\n{output}")
                data = raw.read_bytes()
                if len(data) != 4096 * 16 or data != struct.pack("<4f", *expected) * 4096:
                    raise RuntimeError(f"{stage} {name} width={width}: expected {expected}, first pixel "
                                       f"{struct.unpack_from('<4f', data)}\n{output}")
                if reference is not None and data != reference:
                    raise RuntimeError("native and paired images differ")
                reference = data
            print(f"{stage} {name}: wave64 and paired wave32 match all 4096 pixels; votes={mask.bit_count()}", flush=True)


if __name__ == "__main__":
    main()
