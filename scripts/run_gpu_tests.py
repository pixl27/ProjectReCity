"""Assemble, translate, and run the synthetic RDNA2->SPIR-V GPU test suite on Vulkan.

Usage:
    python -I scripts/run_gpu_tests.py [--llvm-mc CMD] [--build DIR] [--work DIR] [--filter SUBSTR]

--llvm-mc (or LLVM_MC) is a path or a whole command, e.g. "wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc".
"""
import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADER_DIR = os.path.join(ROOT, "runtime", "test", "shaders")

TRANSLATOR = os.path.join(ROOT, "tools", "rdna2spirv.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import mc_command  # noqa: E402
sys.path.insert(0, SHADER_DIR)
from build import CLIP_CASES, clip_container, shader_source
from check import Skip, clip_test, fmt_1010102_test
from compute_fixture import compute_header
from check import (CMPX_CLASS_WORDS, CVT_F32_SAT_CASES, GRADIENT_TESTS, TEST_PROGRAMS, TEXTURE_MIP_DEPENDENCIES,
                   TEXTURE_MIP_TEST, GATHER_WRAP, USER_SGPRS_ENV, USER_SGPRS_WORDS, Skip, mesh_v5_test, not_found_message,
                   pick_build_dir, start_failure, texture_gradient_a16_test, texture_mip_test)
from check import TEXOFF_ARGS, texoff_problem   # ps_texoff's expected words and arguments live in check.py
from check import gpu_test_timeout, timed_out   # per-program time limit (RECITY_GPU_TEST_TIMEOUT)
from check import WAVE32_LANE_DUMPS, narrow_subgroup_skip   # wave32 lane-mask dumps skip below 32 lanes, as in check.py


def make_dxbc(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Build a minimal DXBC container from (fourcc, payload) pairs."""
    n = len(chunks)
    hdr_size = 32 + 4 * n
    offsets = []
    body = bytearray()
    for tag, payload in chunks:
        offsets.append(hdr_size + len(body))
        body += tag + struct.pack("<I", len(payload)) + payload
    total = hdr_size + len(body)
    hdr = bytearray(b"DXBC") + bytes(16) + struct.pack("<III", 1, total, n)
    for off in offsets:
        hdr += struct.pack("<I", off)
    return bytes(hdr) + bytes(body)


def make_sig_chunk(elements: list[tuple[int, int, int, int, int]]) -> bytes:
    """Build ISG1/OSG1 payload: elements of (sysv, ctype, reg, mask, rw_mask)."""
    out = bytearray(struct.pack("<II", len(elements), 8))
    for sysv, ctype, reg, mask, rw in elements:
        e = bytearray(32)
        struct.pack_into("<6I", e, 0, 0, 0, 0, sysv, ctype, reg)
        e[24] = mask
        e[25] = rw
        out += e
    return bytes(out)


def make_ms_psv0(threads_x: int, maxv: int, maxp: int, topology: int = 2, group_shared: int = 0) -> bytes:
    """Build PSV0 payload for a mesh shader (stage 13)."""
    r = bytearray(48)
    struct.pack_into("<I", r, 0, group_shared)
    struct.pack_into("<HH", r, 12, maxv, maxp)
    r[24] = 13  # mesh stage
    r[27] = topology  # 2 = triangles
    struct.pack_into("<III", r, 36, threads_x, 1, 1)
    return struct.pack("<I", len(r)) + bytes(r)


def assemble_to_s(llvm_mc: str, asm_path: str, s_path: str, name: str) -> None:
    """Assemble raw RDNA2 .asm with llvm-mc -show-encoding into .s format for rdna2spirv.py."""
    raw = open(asm_path, encoding="utf-8").read()
    cmd = mc_command.mc_command(llvm_mc) + ["-triple=amdgcn--amdpal", "-mcpu=gfx1013"]
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if not s.startswith(";"):
            break
        m = re.match(r";\s*mattr:\s*(\S+)", s)
        if m:
            cmd.append(f"-mattr={m.group(1)}")
            break
    if name in CLIP_CASES:
        cmd.append(f"-defsym=PARAM_SLOT={CLIP_CASES[name]}")
    cmd.append("-show-encoding")
    p = subprocess.run(
        cmd,
        input=raw,
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise RuntimeError(f"llvm-mc failed on {name}:\n{p.stderr}")
    lines = [f"; {name}  code 0x0..0x0 (synthetic), gfx1013, 0 invalid words"]
    for ln in p.stdout.splitlines():
        if ln.startswith("\t.") or ln.strip().startswith("."):
            continue
        if ln.startswith("\t"):
            lines.append(ln)
    with open(s_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def spirv_as(asm_text: str, out_spv: str) -> None:
    p = subprocess.run(["spirv-as", "--target-env", "vulkan1.2", "-o", out_spv, "-"],
                       input=asm_text, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"spirv-as failed for {out_spv}:\n{p.stderr}")


VS_TRI_SPVASM = """\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %gl_VertexIndex %gl_Position %out0 %out1
               OpDecorate %gl_VertexIndex BuiltIn VertexIndex
               OpDecorate %gl_Position BuiltIn Position
               OpDecorate %out0 Location 0
               OpDecorate %out1 Location 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %ptr_in_i = OpTypePointer Input %int
  %ptr_out_v = OpTypePointer Output %v4float
     %uint_3 = OpConstant %uint 3
     %arr_v4 = OpTypeArray %v4float %uint_3
 %ptr_fn_arr = OpTypePointer Function %arr_v4
 %ptr_fn_v4  = OpTypePointer Function %v4float
        %fm1 = OpConstant %float -1
         %f0 = OpConstant %float 0
         %f1 = OpConstant %float 1
         %f2 = OpConstant %float 2
         %f3 = OpConstant %float 3
      %pos_0 = OpConstantComposite %v4float %fm1 %fm1 %f0 %f1
      %pos_1 = OpConstantComposite %v4float %f3 %fm1 %f0 %f1
      %pos_2 = OpConstantComposite %v4float %fm1 %f3 %f0 %f1
    %pos_arr = OpConstantComposite %arr_v4 %pos_0 %pos_1 %pos_2
      %uv0_0 = OpConstantComposite %v4float %f0 %f0 %f0 %f1
      %uv0_1 = OpConstantComposite %v4float %f2 %f0 %f0 %f1
      %uv0_2 = OpConstantComposite %v4float %f0 %f2 %f0 %f1
    %uv0_arr = OpConstantComposite %arr_v4 %uv0_0 %uv0_1 %uv0_2
      %uv1_0 = OpConstantComposite %v4float %f0 %f0 %f0 %f0
      %uv1_1 = OpConstantComposite %v4float %f2 %f0 %f0 %f0
      %uv1_2 = OpConstantComposite %v4float %f0 %f2 %f0 %f0
    %uv1_arr = OpConstantComposite %arr_v4 %uv1_0 %uv1_1 %uv1_2
%gl_VertexIndex = OpVariable %ptr_in_i Input
%gl_Position = OpVariable %ptr_out_v Output
       %out0 = OpVariable %ptr_out_v Output
       %out1 = OpVariable %ptr_out_v Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %p_var = OpVariable %ptr_fn_arr Function
      %0_var = OpVariable %ptr_fn_arr Function
      %1_var = OpVariable %ptr_fn_arr Function
               OpStore %p_var %pos_arr
               OpStore %0_var %uv0_arr
               OpStore %1_var %uv1_arr
        %idx = OpLoad %int %gl_VertexIndex
       %p_pt = OpAccessChain %ptr_fn_v4 %p_var %idx
       %p_vl = OpLoad %v4float %p_pt
               OpStore %gl_Position %p_vl
       %0_pt = OpAccessChain %ptr_fn_v4 %0_var %idx
       %0_vl = OpLoad %v4float %0_pt
               OpStore %out0 %0_vl
       %1_pt = OpAccessChain %ptr_fn_v4 %1_var %idx
       %1_vl = OpLoad %v4float %1_pt
               OpStore %out1 %1_vl
               OpReturn
               OpFunctionEnd
"""


def make_fs_passthru(loc: int) -> str:
    return f"""\
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %in_v %out_v
               OpExecutionMode %main OriginUpperLeft
               OpDecorate %in_v Location {loc}
               OpDecorate %out_v Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
   %ptr_in_v = OpTypePointer Input %v4float
  %ptr_out_v = OpTypePointer Output %v4float
       %in_v = OpVariable %ptr_in_v Input
      %out_v = OpVariable %ptr_out_v Output
       %main = OpFunction %void None %fn
      %entry = OpLabel
          %v = OpLoad %v4float %in_v
               OpStore %out_v %v
               OpReturn
               OpFunctionEnd
"""


def read_raw_rgba32f(path: str) -> list[tuple[float, float, float, float]]:
    data = open(path, "rb").read()
    n = len(data) // 16
    return [struct.unpack_from("<4f", data, 16 * i) for i in range(n)]


def read_raw_u32(path: str) -> list[tuple[int, int, int, int]]:
    data = open(path, "rb").read()
    n = len(data) // 16
    return [struct.unpack_from("<4I", data, 16 * i) for i in range(n)]


UNSUPPORTED = 77   # vk_harness: the device lacks a stage the test draws with (geometry or mesh shaders)


def unsupported_skip(msg: str) -> str:
    """The SKIP detail for a harness that exited UNSUPPORTED, quoting its "unsupported: ..." line."""
    m = re.search(r"unsupported: .*", msg)
    return f"SKIP ({m.group(0) if m else 'unsupported device'})"


def run_cmd(cmd: list[str], env: dict | None = None, gpu: bool = True) -> tuple[int, str]:
    # A hung GPU test program fails its test instead of stalling the suite (check.py, RECITY_GPU_TEST_TIMEOUT).
    # gpu=False (the translator over the whole suite) runs without a limit.
    limit = gpu_test_timeout() if gpu else None
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=limit)
    except subprocess.TimeoutExpired:
        return 1, timed_out(cmd[0], limit)
    except OSError as e:   # e.g. Smart App Control blocked a fresh build: fail this test, keep running the others
        return 1, start_failure(cmd[0], e)
    return p.returncode, (p.stdout + "\n" + p.stderr).strip()


def w64_emulated() -> bool:
    # Wave64 emulation on: the suite runs every wave64 check without asking for a 64-wide
    # subgroup, matching check.py (W64_EMULATED). Accepts "1"/"true"/"yes"/"on" and tolerates
    # the trailing space cmd.exe's `set VAR=1 &` leaves in the value.
    return (os.environ.get("RECITY_W64_EMULATE") or "").strip().lower() in ("1", "true", "yes", "on")
def wavemask_check(vk_compute: str, spv: str, env: dict | None, emulated: bool) -> tuple[bool | None, str]:
    """cs_wavemask, as in check.py: (passed, detail), passed None for a skip. The native build needs a 64-wide
    subgroup and skips where the device has none; with RECITY_W64_EMULATE=1 it was translated with
    --wave64-emulate and runs on any subgroup size, so it asks for none."""
    cmd = [vk_compute, spv, "1", "wavemask"] + ([] if emulated else ["--require-wave64"])
    rc, msg = run_cmd(cmd, env=env)
    if not emulated and rc == 77:
        return None, "SKIP (no 64-wide compute subgroup on this device)"
    if not emulated and rc == 0 and "compute subgroup size: 64" not in msg.splitlines():
        return False, "64-wide subgroup was not confirmed (rebuild vk_compute_test)"
    return rc == 0, msg.splitlines()[-1] if msg else ""
def dump_words(msg: str) -> list[int] | None:
    """The eight words of vk_compute_test's "out[0..7] = ..." line, or None unless it is eight decimal words (as in
    check.py): trailing text, a missing word or a non-number is a malformed dump, not a result."""
    m = re.search(r"^out\[0\.\.7\] = ((?:[0-9]+ ){7}[0-9]+)$", msg, re.MULTILINE)
    return [int(x) for x in m.group(1).split()] if m else None


def missing_result(expected: str, msg: str) -> str:
    return f"helper exited 0 without {expected}\n{msg}".rstrip()


def judge_cs_dump(rc: int, msg: str, want: list[int], *, wave64: bool, partial: bool, emulated: bool,
                  hexwords: bool = False) -> tuple[bool | None, str]:
    """Judge one vk_compute_test "dump" run: (True, words), (False, why), or (None, why) when the device can't run it.
    wave64: it was run with --require-wave64, so exit 77 is a skip and the 64-wide confirmation must be printed.
    partial: a group smaller than a wave, run at subgroup size 64 without full subgroups; skipped when the device
    reports no subgroup size of 64. emulated: translated with --wave64-emulate, runs on any subgroup size."""
    no64 = partial and not emulated and not any(int(x) >= 64 for x in re.findall(r"subgroup \d+\.\.(\d+)", msg))
    if (wave64 and rc == 77) or (no64 and rc == 0):
        return None, "SKIP (no 64-wide compute subgroup on this device)"
    if rc != 0:
        return False, msg
    words = dump_words(msg)
    if words is None:
        return False, missing_result("a valid out[0..7] line (eight decimal words)", msg)
    if wave64 and "compute subgroup size: 64" not in msg.splitlines():
        return False, "64-wide subgroup was not confirmed (rebuild vk_compute_test)"
    vals = words[:len(want)]
    return vals == want, " ".join((hex if hexwords else str)(x) for x in vals)


def check_pixel_const(raw_path: str, want: tuple[int, int, int, int]) -> tuple[bool, str]:
    px = read_raw_u32(raw_path)
    if len(px) != 64 * 64:
        return False, f"expected 4096 pixels, got {len(px)}"
    bad = [(i, p) for i, p in enumerate(px) if p != want]
    if bad:
        i, p = bad[0]
        return False, f"{len(bad)}/4096 mismatches; first at ({i%64},{i//64}): got {[hex(x) for x in p]}, want {[hex(x) for x in want]}"
    return True, "4096/4096 pixels exact"


def check_bary(raw_path: str) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    bad = 0
    first = ""
    for i, (ax, ay, bi, bj) in enumerate(px):
        if abs(ax * 0.5 - bi) > 1e-4 or abs(ay * 0.5 - bj) > 1e-4:
            if not bad:
                first = f"at ({i%64},{i//64}): attr/2=({ax*0.5:.5f},{ay*0.5:.5f}) raw=({bi:.5f},{bj:.5f})"
            bad += 1
    return (bad == 0), (first if bad else "4096/4096 pixels exact")


def check_bitmask_perm(raw_path: str) -> tuple[bool, str]:
    px = read_raw_u32(raw_path)
    bad = 0
    first = ""
    judged = 0
    for i, (lane, v1, v2, v3) in enumerate(px):
        if lane >= 32:      # wave32 test: in a 64-wide subgroup lanes 32-63 hold values it never wrote
            continue
        judged += 1
        w1 = lane ^ 9
        w2 = ((lane & 5) | 16) ^ 1
        if v1 != w1 or v2 != w2 or v3 != 0:
            if not bad:
                first = f"at ({i%64},{i//64}) lane={lane}: got ({v1},{v2},{v3}), want ({w1},{w2},0)"
            bad += 1
    return (bad == 0 and judged > 0), (first if bad else f"{judged}/{judged} pixels in lanes 0-31 exact")


# Pixel shaders that read memory: vk_harness puts this V# in s[0:3] and seeds the heap it points at, as
# tests/semantic/test_value_semantics.py does for the interpreter (same table as check.py).
PIXEL_ENV = {"ps_vcc_x2": {"VKH_SGPRS": "0x1000,0,16,0", "VKH_HEAP": "0x1000=0x11223344,0x1004=0x55667788"}}


def check_flat(raw_path: str) -> tuple[bool, str]:
    """ps_flat with vs_ids: the integer IDs come from the first (provoking) vertex, the UVs still interpolate."""
    data = open(raw_path, "rb").read()
    px = list(struct.iter_unpack("<2I2f", data))
    w = int(len(px) ** 0.5)
    wrong = sum((a, b) != (0x101, 0x102) for a, b, _, _ in px)
    err = max(max(abs(u - (i % w + 0.5) / w), abs(v - (i // w + 0.5) / w)) for i, (_, _, u, v) in enumerate(px))
    ok = wrong == 0 and err < 1e-6
    return ok, (f"{len(px)} pixels: flat IDs exact, uv within {err:.1g}" if ok else
                f"{wrong} pixels without the first vertex's value, uv off by {err:.3g}")


def check_wqm(raw_path: str) -> tuple[bool, str]:
    """ps_wqm.asm: per pixel (wqm, own, wqm, own); wqm iff x%4 < 2 and y%4 >= 2, own iff x%4 == 1 and y%4 == 2."""
    px = read_raw_u32(raw_path)
    w = int(len(px) ** 0.5)
    wrong = sum(tuple(p) != (a, b, a, b)
                for i, p in enumerate(px)
                for a, b in [(int(i % w % 4 < 2 and i // w % 4 >= 2), int(i % w % 4 == 1 and i // w % 4 == 2))])
    return wrong == 0, "whole-quad mask exact on every pixel" if not wrong else f"{wrong} of {len(px)} pixels differ"


def check_uniform_branch(raw_path: str) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    u32 = read_raw_u32(raw_path)
    """ps_uniform_branch / ps_scc_mask32: s5 = 8 wherever the lane's own bit is set; a lane with bit 0 gets 8 when
    another lane of its wave has the bit (the wave-wide vote), else 7. All three kinds must occur, as in check.py:
    without a (8, bit 0) pixel the vote is not shown, and a translator that decides per lane would pass."""
    bad = 0
    n8 = 0
    n7 = 0
    n80 = 0
    first = ""
    for i, ((_, _, bary_i, _), (s5, bit, _, _)) in enumerate(zip(px, u32)):
        n80 += s5 == 8 and bit == 0
        if bit == 1:
            if bary_i <= 0.27 or s5 != 8:
                if not bad:
                    first = f"bit=1 at ({i%64},{i//64}): s5={s5}, i={bary_i:.4f}"
                bad += 1
        elif bit == 0:
            if bary_i > 0.2701 or s5 not in (7, 8):
                if not bad:
                    first = f"bit=0 at ({i%64},{i//64}): s5={s5}, i={bary_i:.4f}"
                bad += 1
        else:
            if not bad:
                first = f"unexpected bit={bit}"
            bad += 1
        if s5 == 8:
            n8 += 1
        elif s5 == 7:
            n7 += 1
    if bad:
        return False, first
    detail = f"s5==8: {n8} ({n80} of them with bit 0), s5==7: {n7} ({len(u32)} px)"
    if not n80:
        return False, "no bit=0 pixel got s5=8: the branch was decided per lane, not by the wave; " + detail
    return (n8 - n80 > 0 and n7 > 0), detail


def check_grad(raw_path: str, want_b: float, want_a: float) -> tuple[bool, str]:
    px = read_raw_rgba32f(raw_path)
    bad = 0
    first = ""
    for i, (r, g, b, a) in enumerate(px):
        wx = ((i % 64) + 0.5) / 64.0
        wy = ((i // 64) + 0.5) / 64.0
        if abs(r - wx) > 1e-3 or abs(g - wy) > 1e-3 or abs(b - want_b) > 1e-3 or abs(a - want_a) > 1e-3:
            if not bad:
                first = f"at ({i%64},{i//64}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f}), want ({wx:.4f},{wy:.4f},{want_b},{want_a})"
            bad += 1
    return (bad == 0), (first if bad else "4096/4096 pixels exact")


def check_clip_half(raw_path: str, want_b: float, want_a: float) -> tuple[bool, str]:
    """Check 64x64 output with ClipDistance[0] = NDC x: left half (x < 32) clipped to clear (-1),
    right half (x >= 32) covered (2048/4096 pixels) with the gradient."""
    px = read_raw_rgba32f(raw_path)
    if len(px) != 64 * 64:
        return False, f"expected 4096 pixels, got {len(px)}"
    covered = 0
    bad = 0
    first = ""
    for i, (r, g, b, a) in enumerate(px):
        x, y = i % 64, i // 64
        is_clear = (r == -1.0 and g == -1.0 and b == -1.0 and a == -1.0)
        if not is_clear:
            covered += 1
        if x < 32:
            if not is_clear:
                if not bad:
                    first = f"unclipped pixel at ({x},{y}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f})"
                bad += 1
        else:
            wx = (x + 0.5) / 64.0
            wy = (y + 0.5) / 64.0
            if is_clear or abs(r - wx) > 1e-3 or abs(g - wy) > 1e-3 or abs(b - want_b) > 1e-3 or abs(a - want_a) > 1e-3:
                if not bad:
                    first = f"at ({x},{y}): got ({r:.4f},{g:.4f},{b:.4f},{a:.4f}), want ({wx:.4f},{wy:.4f},{want_b},{want_a})"
                bad += 1
    ok = (bad == 0 and covered == 2048)
    detail = f"{covered}/4096 pixels covered (expected 2048)" if ok else f"covered={covered}/2048; {first}"
    return ok, detail


# Shaders a selected test draws with, built even when the filter leaves them out (#192).
DRAWN_WITH = {"ps_flat": "vs_ids.asm", "ps_texture_sample": "vs_ids.asm", "ps_cube_sample": "vs_ids.asm",
              "ps_texture_sample_d16": "vs_ids.asm", "ps_sample_c_lz": "vs_ids.asm", "ps_sample_c_lz_a16": "vs_ids.asm",
              **{name: "vs_ids.asm" for name in GRADIENT_TESTS}, "vs_clip": "ps_ops.asm"}


def with_draw_shaders(asm_files: list[str]) -> list[str]:
    build = list(asm_files)
    for f in asm_files:
        extra = DRAWN_WITH.get(shader_source(os.path.splitext(f)[0]))
        if extra and extra not in build:
            build.append(extra)
    if TEXTURE_MIP_TEST + ".asm" in asm_files:
        for name in TEXTURE_MIP_DEPENDENCIES:
            if name + ".asm" not in build:
                build.append(name + ".asm")
    return build


REQUIRED_TEST_PROGRAMS = ("vk_compute_test", "vk_store_test", "vk_harness")
BUILD_CONFIGS = ("RelWithDebInfo", "Release")


def find_test_programs(build: str, ext: str) -> str | None:
    """Return the directory containing all required GPU test programs.

    Single-configuration generators (Ninja, Makefiles) put executables directly
    in the build directory. Visual Studio and other multi-configuration
    generators put them in a configuration subdirectory.
    """
    candidates = [build, *(os.path.join(build, config) for config in BUILD_CONFIGS)]
    for directory in candidates:
        if all(os.path.isfile(os.path.join(directory, name + ext)) for name in REQUIRED_TEST_PROGRAMS):
            return directory
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llvm-mc", default=os.environ.get("LLVM_MC", "llvm-mc"),
                    help="llvm-mc path, or a whole command such as 'wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc'")
    ap.add_argument("--build", default=os.path.join(ROOT, "runtime", "build"),
                    help="CMake build folder; with the Visual Studio generator the RelWithDebInfo/Release/... "
                         "subfolder that has the test programs is found by itself")
    ap.add_argument("--work", default=None)
    ap.add_argument("--filter", default="")
    a = ap.parse_args()
    # Some test programs run with their working directory in <work>/spirv (vk_texture_test), so a relative path
    # given here would point somewhere else for them.
    a.build = os.path.abspath(a.build)
    if a.work:
        a.work = os.path.abspath(a.work)

    ext = ".exe" if os.name == "nt" else ""
    required = tuple(t + ext for t in TEST_PROGRAMS)
    a.build = pick_build_dir(a.build, required)
    missing = [t for t in required if not os.path.isfile(os.path.join(a.build, t))]
    if missing:
        sys.exit(not_found_message(a.build, missing))
    requested_build = a.build
    resolved_build = find_test_programs(requested_build, ext)
    if resolved_build is None:
        searched = [requested_build, *(os.path.join(requested_build, config) for config in BUILD_CONFIGS)]
        details = []
        for directory in searched:
            missing = [name + ext for name in REQUIRED_TEST_PROGRAMS
                       if not os.path.isfile(os.path.join(directory, name + ext))]
            details.append(f"  {directory}: {', '.join(missing)}")
        sys.exit("Required GPU test programs were not found together in one build directory.\n"
                 "Missing files by searched directory:\n" + "\n".join(details) + "\n"
                 "Build them with cmake -S runtime -B runtime/build, then cmake --build runtime/build.\n"
                 "For Visual Studio, use: cmake --build <build-dir> --config RelWithDebInfo "
                 "--target vk_compute_test vk_store_test vk_harness.\n"
                 "CMake only builds them when it finds Vulkan. If it printed \"Vulkan SDK not found\", install the "
                 "Vulkan headers (vulkan-headers on Arch) and configure again.")
    a.build = resolved_build

    work = a.work or tempfile.mkdtemp(prefix="recity_gpu_tests_")
    try:
        return run_suite(a, work)
    finally:
        if a.work is None:
            shutil.rmtree(work, ignore_errors=True)


def run_suite(a, work: str) -> int:
    asm_files = sorted({f for f in os.listdir(SHADER_DIR) if f.endswith(".asm")} | {n + ".asm" for n in CLIP_CASES})
    if a.filter:
        asm_files = [f for f in asm_files if a.filter in f]
        if not asm_files:
            print(f"error: --filter {a.filter!r} matched no GPU test cases", file=sys.stderr)
            return 2
    build_files = with_draw_shaders(asm_files)

    fssm_dir = os.path.join(work, "fssm")
    disasm_dir = os.path.join(work, "disasm")
    spirv_dir = os.path.join(work, "spirv")
    raw_dir = os.path.join(work, "raw")
    for d in (fssm_dir, disasm_dir, spirv_dir, raw_dir):
        os.makedirs(d, exist_ok=True)

    # 1. Create synthetic DXBC / FSSM metadata + assemble .asm -> .s
    disasm_files = []
    for fname in build_files:
        name = os.path.splitext(fname)[0]
        if name in CLIP_CASES and name != "vs_clip_undeclared":
            dxbc = clip_container()
        elif name == "gs_tri":
            dxbc = make_dxbc([
                (b"ISG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])),
                (b"OSG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (0, 3, 1, 0xF, 0xF)])),
            ])
        elif name in ("ms_tri", "ms_w64x_ballot", "ms_wave_info_s3"):
            dxbc = make_dxbc([(b"PSV0", make_ms_psv0(128, 64, 64, 2))])
        elif name == "ms_v5_workgroup":
            dxbc = make_dxbc([(b"PSV0", make_ms_psv0(32, 3, 1, group_shared=128))])
        elif name == "ms_clip":
            dxbc = make_dxbc([
                (b"PSV0", make_ms_psv0(128, 64, 64, 2)),
                # OSG1: SV_Position (sysv=1) reg 0 xyzw, SV_ClipDistance (sysv=2) reg 1 x
                (b"OSG1", make_sig_chunk([(1, 3, 0, 0xF, 0xF), (2, 3, 1, 0x1, 0xE)])),
            ])
        else:
            dxbc = make_dxbc([])
        with open(os.path.join(work, f"{name}.dxbc"), "wb") as f:
            f.write(dxbc)
        if name.startswith("cs_"):
            with open(os.path.join(fssm_dir, f"{name}.fssm"), "wb") as f:
                f.write(compute_header(os.path.join(SHADER_DIR, fname)))
        fname = shader_source(name) + ".asm"
        disasm_path = os.path.join(disasm_dir, f"{name}.s")
        assemble_to_s(a.llvm_mc, os.path.join(SHADER_DIR, fname), disasm_path, name)
        disasm_files.append(disasm_path)
        # A failed translation must not run an old binary left in a reused --work directory.
        Path(spirv_dir, f"{name}.spv").unlink(missing_ok=True)

    # 2. Translate this run's inputs only. The translator looks for metadata one level above the input directory.
    with tempfile.TemporaryDirectory(prefix="disasm-", dir=work) as selected_disasm:
        for path in disasm_files:
            shutil.copyfile(path, os.path.join(selected_disasm, os.path.basename(path)))
        cmd = [sys.executable, "-I", TRANSLATOR, selected_disasm, "--out", spirv_dir]
        if w64_emulated():   # wave64 compute/mesh exact on any subgroup size
            cmd.append("--wave64-emulate")
        rc, out = run_cmd(cmd, gpu=False)
    print("=== rdna2spirv.py ===")
    print(out)
    if rc != 0:
        return 1

    # 3. Build helper SPIR-V shaders
    vs_tri = os.path.join(spirv_dir, "vs_tri.spv")
    fs_passthru = os.path.join(spirv_dir, "fs_passthru.spv")
    fs_loc1 = os.path.join(spirv_dir, "fs_loc1.spv")
    spirv_as(VS_TRI_SPVASM, vs_tri)
    spirv_as(make_fs_passthru(0), fs_passthru)
    spirv_as(make_fs_passthru(1), fs_loc1)

    index = {}
    idx_path = os.path.join(spirv_dir, "index.json")
    if os.path.isfile(idx_path):
        index = json.load(open(idx_path, encoding="utf-8"))

    ext = ".exe" if os.name == "nt" else ""
    vk_compute = os.path.join(a.build, "vk_compute_test" + ext)
    vk_store = os.path.join(a.build, "vk_store_test" + ext)
    vk_harness = os.path.join(a.build, "vk_harness" + ext)
    vk_bindless = os.path.join(a.build, "vk_bindless_test" + ext)

    results: list[tuple[str, bool, str]] = []
    for fname in asm_files:
        name = os.path.splitext(fname)[0]
        spv = os.path.join(spirv_dir, f"{name}.spv")
        if not os.path.isfile(spv):
            results.append((name, False, "missing .spv (translation failed)"))
            continue
        sg = index.get(name, {}).get("subgroup_size")
        cs_env = dict(os.environ, VKC_SUBGROUP_SIZE=str(sg)) if sg else None
        gfx_env = dict(os.environ, VKH_SUBGROUP_SIZE=str(sg)) if sg else None

        if name in CLIP_CASES:
            detail = clip_test(a.build, spirv_dir, name)
            results.append((name, None if isinstance(detail, Skip) else not detail, detail or "four pixel masks exact"))
        elif name == "cs_workgroup_yz":
            rc, msg = run_cmd([vk_compute, spv, "2x3x4", "workgroup_yz"], env=cs_env)
            results.append((name, rc == 0 and "2x3x4 workgroup Y/Z: 0 mismatches" in msg.splitlines(), msg))
        elif name == "cs_lds":
            rc, msg = run_cmd([vk_compute, spv, "4", "mirror64"], env=cs_env)
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_exec64":
            rc, msg = run_cmd([vk_compute, spv, "1", "exec64"], env=cs_env)
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_bvh":
            rc, msg = run_cmd([vk_compute, spv, "1", "bvh"], env=cs_env)
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name in ("cs_bounds", "cs_cmpx_class_f32"):
            # These per-lane results do not require a particular subgroup width.
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=cs_env)
            want = CMPX_CLASS_WORDS if name == "cs_cmpx_class_f32" else [0x12345678, 0x12345678, 0, 0xEEEEEEEE]
            results.append((name, *judge_cs_dump(rc, msg, want, wave64=False, partial=False, emulated=False,
                                                 hexwords=True)))
        elif name == "cs_fmt_1010102":
            problem = fmt_1010102_test(a.build, spirv_dir, cs_env)
            results.append((name, not problem, problem or "64/64 dumped words exact"))
        elif name == "cs_global_neg_offset":
            # Negative global_load/global_store offsets (cs_global_neg_offset.asm): 40 groups give a 0x2800-byte
            # heap holding both the right addresses and the 4096-lower ones. One thread, no wave64 requirement.
            rc, msg = run_cmd([vk_compute, spv, "40", "dump"], env=cs_env)
            want = [0x5EED1FFC, 0x8BAD1800, 0x5EED1FF0, 0x5EED1FF4, 0x9E, 0xFFFF8BAD, 0x57023FFC, 0xEEEEEEEE]
            ok, detail = judge_cs_dump(rc, msg, want, wave64=False, partial=False, emulated=False, hexwords=True)
            results.append((name, ok, detail))
        elif name == "cs_user_sgprs":
            # vk_compute_test seeds s4..s7 from VKC_SGPRS; the shader stores them back.
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"], env=dict(cs_env or os.environ, **USER_SGPRS_ENV))
            results.append((name, *judge_cs_dump(rc, msg, USER_SGPRS_WORDS, wave64=False, partial=False,
                                                 emulated=False, hexwords=True)))
        elif name == "cs_cvt_f32_sat":
            # #300's saturating float -> int conversions: one dispatch per input word in s4 (VKC_SGPRS).
            wrong = []
            for bits, unsigned, signed in CVT_F32_SAT_CASES:
                rc, msg = run_cmd([vk_compute, spv, "1", "dump"],
                                  env=dict(cs_env or os.environ, VKC_SGPRS=f"{bits:#010x}"))
                ok, detail = judge_cs_dump(rc, msg, [unsigned, signed], wave64=False, partial=False,
                                           emulated=False, hexwords=True)
                if not ok:
                    wrong.append(f"{bits:#010x} expected {unsigned:#x} {signed:#x}, got {detail or 'no output'}")
            n = len(CVT_F32_SAT_CASES)
            results.append((name, not wrong, f"{n}/{n} inputs exact" if not wrong
                            else f"{len(wrong)}/{n} inputs wrong, first {wrong[0]}"))
        elif name == "cs_wavemask":
            # Whole-wave mask test, as in check.py: needs a 64-wide subgroup, skipped where the device has none.
            # With RECITY_W64_EMULATE=1 it was translated with --wave64-emulate and runs on any subgroup size.
            emulated = os.environ.get("RECITY_W64_EMULATE") == "1"
            args = [] if emulated else ["--require-wave64"]
            rc, msg = run_cmd([vk_compute, spv, "1", "wavemask", *args], env=cs_env)
            # Whole-wave mask test, as in check.py (require_wave64=not W64_EMULATED): ask for a 64-wide
            # subgroup and skip where the device has none — unless wave64 is emulated, in which case it
            # runs on any subgroup size (translated with --wave64-emulate) and passes on "0 mismatches".
            emulated = w64_emulated()
            rc, msg = run_cmd([vk_compute, spv, "1", "wavemask"] + (["--require-wave64"] if not emulated else []),
                              env=cs_env)
            if not emulated and rc == 77:
                results.append((name, None, "SKIP (no 64-wide compute subgroup on this device)"))
            elif not emulated and rc == 0 and "compute subgroup size: 64" not in msg.splitlines():
                results.append((name, False, "64-wide subgroup was not confirmed (rebuild vk_compute_test)"))
            else:
                results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
            results.append((name,) + wavemask_check(vk_compute, spv, cs_env,
                                                    os.environ.get("RECITY_W64_EMULATE") == "1"))
        elif name in ("cs_store", "cs_smem_soffset_imm", "cs_w32_lane_wrap", "cs_movrelsd_2", "cs_w32_permute", "cs_w32_lds_u8_d16", "cs_wave64_ballot", "cs_wm_half_ops", "cs_wm_half_cmp", "cs_wm_bfm64", "cs_wm_scc_rfl", "cs_wqm_w64", "cs_wqm_w32", "cs_wqm_w64_half",

                      "cs_w64_mask_detect",
                      "cs_w64x_readlane", "cs_w64x_writelane", "cs_w64x_readfirstlane", "cs_w64x_bpermute", "cs_w64x_permute",
                      "cs_w64x_branch", "cs_w64x_ballot", "cs_w64x_wg128_branches", "cs_w64x_lds_raw",
                      "cs_w64x_lds_war", "cs_w64x_partial", "cs_w64x_sbarrier", "cs_w64x_load_ballot_load",
                      "cs_w64x_mask_reuse", "cs_w64x_mask_reuse2", "cs_w64x_chain_barrier", "cs_barrier_divergent",
                      "cs_bpermute32"):
            cs_dump_wants = {
                "cs_store": [0x400080FF, 0xE00003FF, 0x3C004000, 0xEEEEEEEE, 0x40003C00, 0xBC003800],
                # base + soffset + imm (cs_smem_soffset_imm.asm); out[4..7] are seeds the shader leaves alone
                "cs_smem_soffset_imm": [0x5EED0024, 0x5EED0028, 0x5EED002C, 0x5EED0030,
                                        0x5EED0010, 0x5EED0014, 0x5EED0018, 0x5EED001C],
                # SHADER-52: wave32 lane select & 31 (40 → 8)
                "cs_w32_lane_wrap": [25, 16, 777, 777],
                # v_movrelsd_2_b32: VGPR[D + M0[25:16]] = VGPR[S0 + M0[9:0]] (cs_movrelsd_2.asm)
                "cs_movrelsd_2": [11, 33, 33, 55, 1717, 77, 1515, 1717],
                # SHADER-90: ds_permute_b32 (cs_w32_permute.asm): scatter, unwritten lane = 0, EXEC gates the write
                "cs_w32_permute": [218, 29, 8, 211, 0, 15, 99, 15],
                # SHADER-85: ds_read_u8/i8_d16[_hi] (cs_w32_lds_u8_d16.asm header)
                "cs_w32_lds_u8_d16": [0xBEEF00FF, 0x00FF1234, 0xBEEFFF80, 0xFF801234, 0xBEEF007F, 0x007F1234, 0xBEEF0080, 0xFFFF1234],
                "cs_wave64_ballot": [32, 16, 24, 40, 8, 48],
                "cs_wm_half_ops": [32, 16, 16, 0, 16],
                "cs_wm_half_cmp": [1, 0, 1, 1],
                "cs_wm_bfm64": [16, 8, 8, 40],
                "cs_wm_scc_rfl": [55, 48],
                "cs_w64_mask_detect": [48],
                "cs_wqm_w64": [20, 4, 12, 8, 1, 0, 15728895, 20],       # cs_wqm_w64.asm / cs_wqm_w32.asm headers
                "cs_wqm_w32": [12, 1, 4, 0, 8, 15728880, 4, 32],
                "cs_wqm_w64_half": [11, 14, 0, 1, 6, 8, 12, 2],
                # cross-half tests (derivations in each .asm); need exact wave64 or its emulation
                "cs_w64x_readlane": [121, 121, 16, 97, 16, 100, 100, 121],
                "cs_w64x_writelane": [4660, 77, 99, 2748, 0, 0, 4660, 99],
                "cs_w64x_readfirstlane": [140, 140, 163, 132, 100, 105, 133, 48],
                "cs_w64x_bpermute": [36, 316, 253, 218, 435, 225, 0, 442],
                "cs_w64x_permute": [218, 442, 435, 239, 0, 99, 239, 232],      # SHADER-90, cs_w64x_permute.asm
                "cs_w64x_branch": [0, 0, 0, 0, 5, 5, 9, 9],
                "cs_w64x_mask_reuse": [8, 40, 8, 40, 2, 20, 10, 64],
                "cs_w64x_mask_reuse2": [10, 20, 8, 8, 5, 2, 132, 131],
                "cs_w64x_chain_barrier": [564, 563, 500, 627],
                "cs_w64x_ballot": [2053, 2061, 2069, 2077, 2085, 2093, 2101, 2109],
                "cs_w64x_wg128_branches": [121, 121, 343, 343, 14, 31, 19812, 22728],
                "cs_w64x_lds_raw": [132, 137, 100, 108, 131, 101, 149, 118],
                "cs_w64x_lds_war": [1032, 1063, 1000, 1031, 2032, 2063, 2000, 2031],
                "cs_w64x_partial": [40, 40, 528, 0, 0, 532, 539, 524820],
                "cs_w64x_sbarrier": [605, 605, 686, 686, 3032, 3013, 3032, 3031],
                "cs_w64x_load_ballot_load": [43, 74, 75, 42, 42, 18, 41, 32],
                "cs_bpermute32": [0, 50, 162, 99, 36, 218, 183, 15],     # wave32: needs 32 lanes
                # SHADER-91: s_barrier sites in different wave-uniform branches meet (cs_barrier_divergent.asm)
                "cs_barrier_divergent": [127, 126, 125, 124, 123, 122, 121, 120],
            }
            want = cs_dump_wants[name]
            # Every dump but cs_store / cs_smem_soffset_imm / cs_w32_lane_wrap / cs_movrelsd_2 / cs_w32_permute /
            # cs_w32_lds_u8_d16 and the wave32 cs_wqm_w32 / cs_bpermute32 is a whole-wave (wave64) mask test:
            # ask for a 64-wide subgroup and skip, as check.py does, where the device has none (vk_compute_test
            # exits 77) instead of reporting a failure. With RECITY_W64_EMULATE=1 they were translated with
            # --wave64-emulate and run on any subgroup size.
            # cs_w64x_partial (a 40-thread group) can't have full subgroups: it asks for size 64 without them.
            emulated = w64_emulated()
            partial = name == "cs_w64x_partial"
            wave64 = name not in ("cs_store", "cs_smem_soffset_imm", "cs_wqm_w32", "cs_w32_lane_wrap", "cs_movrelsd_2",
                                  "cs_w32_permute", "cs_barrier_divergent", "cs_w32_lds_u8_d16", "cs_bpermute32") and not emulated and not partial

            if partial and not emulated:
                cs_env = dict(os.environ, VKC_SUBGROUP_SIZE="64")
            rc, msg = run_cmd([vk_compute, spv, "1", "dump"] + (["--require-wave64"] if wave64 else []), env=cs_env)
            skip = narrow_subgroup_skip(msg) if name in WAVE32_LANE_DUMPS else None
            if skip:   # e.g. Mesa llvmpipe's 8-wide subgroups: these 32-lane masks can't come out right
                ok, detail = None, str(skip)
            else:
                ok, detail = judge_cs_dump(rc, msg, want, wave64=wave64, partial=partial, emulated=emulated,
                                           hexwords=name in ("cs_store", "cs_smem_soffset_imm", "cs_w32_lds_u8_d16"))
            results.append((name, ok, detail))
        elif name == "cs_imgstore":
            rc, msg = run_cmd([vk_store, spv])
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "cs_imgload":
            rc, msg = run_cmd([vk_store, spv, "load"])
            results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == TEXTURE_MIP_TEST:
            problem = texture_mip_test(a.build, spirv_dir)
            ok = None if isinstance(problem, Skip) else not problem
            results.append((name, ok, problem or "two passes, 4096 RGBA32F pixels each exact; wrong mip and UV rejected"))
        elif name in GRADIENT_TESTS:
            problem = texture_gradient_a16_test(a.build, spirv_dir, name)
            ok = None if isinstance(problem, Skip) else not problem
            results.append((name, ok, problem or "4096 RGBA pixels matched"))
        elif name in ("ps_texture_sample", "ps_cube_sample", "ps_texture_sample_d16", "ps_sample_c_lz",
                      "ps_sample_c_lz_a16"):
            # Bindless texture tests: vk_texture_test finds vs_ids.spv and the shader in its working directory.
            tool = os.path.join(a.build, "vk_texture_test" + ext)
            raw = os.path.join(raw_dir, f"{name}.raw")
            if name == "ps_texture_sample_d16":
                cmd = [tool, os.path.join(spirv_dir, "vs_ids.spv"), spv, raw, "0", "0", "0", "0", "0"]
            else:
                mode = {"ps_cube_sample": ["cube"], "ps_sample_c_lz": ["dref"],
                        "ps_sample_c_lz_a16": ["dref_a16"]}.get(name, [])
                cmd = [tool, *mode, raw]
            try:
                p = subprocess.run(cmd, capture_output=True, text=True, cwd=spirv_dir)
                rc, msg = p.returncode, (p.stdout + "\n" + p.stderr).strip()
            except FileNotFoundError:
                rc, msg = 1, f"missing {tool}"
            except OSError as e:
                rc, msg = 1, start_failure(tool, e)
            if name == "ps_texture_sample_d16" and rc == 0 and os.path.isfile(raw):
                with open(raw, "rb") as f:
                    f.seek((32 * 64 + 32) * 16)
                    r, g, b, alpha = struct.unpack("<4f", f.read(16))
                exp_b = struct.unpack("<e", struct.pack("<e", 128.0 / 255.0))[0]
                ok = abs(r - exp_b) <= 0.05 and abs(g - exp_b) <= 0.05 and abs(b - exp_b) == 0.0 and alpha == 1.0
                results.append((name, ok, f"center(32,32)=({r:.6f}, {g:.6f}, {b:.6f}, {alpha:.1f})"))
            elif rc == 0 and "PASSED" not in msg:
                results.append((name, False, missing_result("the PASSED marker", msg)))
            else:
                results.append((name, rc == 0, msg.splitlines()[-1] if msg else ""))
        elif name == "ps_gather_wrap":
            # SHADER-36: image_gather4_l_o under wrap/mirror/clamp/mirror-once S#s; every pixel holds GATHER_WRAP.
            tool = os.path.join(a.build, "vk_texture_test" + ext)
            raw = os.path.join(raw_dir, f"{name}.raw")
            try:
                p = subprocess.run([tool, os.path.join(spirv_dir, "vs_ids.spv"), spv, raw, "0", "92", "c00000",
                                    "14500000", "0"], capture_output=True, text=True, cwd=spirv_dir)
                rc, msg = p.returncode, (p.stdout + "\n" + p.stderr).strip()
            except FileNotFoundError:
                rc, msg = 1, f"missing {tool}"
            bad = 4096
            if rc == 0 and os.path.isfile(raw):
                data = Path(raw).read_bytes()
                bad = 4096 if len(data) != 64 * 64 * 16 else sum(q != GATHER_WRAP for q in struct.iter_unpack("<4f", data))
            results.append((name, rc == 0 and bad == 0, "4096/4096 pixels exact" if bad == 0 else
                            f"{bad} pixels differ from {GATHER_WRAP}: {msg.splitlines()[-1] if msg else ''}"))
        elif name == "ps_tex":
            # A texture through the descriptor hash table and the bindless set. vk_bindless_test compares the
            # pixels itself, and exits 77 where the device lacks a feature the set or the shader needs.
            rc, msg = run_cmd([vk_bindless, vs_tri, spv])
            lines = msg.splitlines()
            if rc == 77:
                reason = next((ln for ln in lines if ln.startswith("unsupported:")), "unsupported device")
                results.append((name, None, f"SKIP ({reason})"))
            else:
                results.append((name, rc == 0, lines[-1] if lines else ""))
        elif name == "ps_texoff":
            # Texel offsets: needs vk_texture_test's texture, sampler and descriptor table (vk_harness binds none),
            # and VK_KHR_maintenance8 for the non-constant Offset; the tool exits 77 without it.
            raw = os.path.join(raw_dir, f"{name}.raw")
            rc, msg = run_cmd([os.path.join(a.build, "vk_texture_test" + ext), vs_tri, spv, raw, *TEXOFF_ARGS],
                              env=dict(os.environ, VKT_NEED_MAINTENANCE8="1"))
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                with open(raw, "rb") as f:
                    problem, covered = texoff_problem(f.read(), msg)
                results.append((name, not problem, problem or f"4096/4096 pixels exact ({covered})"))
        elif name.startswith("ps_"):
            raw = os.path.join(raw_dir, f"{name}.raw")
            vs = os.path.join(spirv_dir, "vs_ids.spv") if name == "ps_flat" else vs_tri   # check.py draws ps_flat with vs_ids
            env = dict(gfx_env or os.environ, **PIXEL_ENV[name]) if name in PIXEL_ENV else gfx_env
            rc, msg = run_cmd([vk_harness, vs, spv, raw], env=env)
            if rc == 77:   # vk_harness: the device lacks a feature this shader needs (e.g. barycentrics)
                results.append((name, None, unsupported_skip(msg)))
                continue
            if rc != 0:
                results.append((name, False, msg))
                continue
            const_wants = {
                "ps_ops": (0x48804300, 5, 7, 0xFFFFFFFC),
                "ps_ops2": (0xABCD4400, 0x46001234, 0xFFFFFFFA, 7),
                "ps_ops3": (0xFFFFFFFA, 0x44660066, 0x44AABBCC, 16),
                "ps_ops4": (0x34000010, 0x00120800, 0xBEEF0000, 0x12343C00),
                "ps_scc": (1, 1, 3, 0x80000000),
                "ps_sdwa": (0x000000F1, 0xFFFEFFF2, 0x3C003344, 0xFFF70000),
                "ps_legacy_mul": (0, 0, 0, 0x40C00000),        # same expectation as check.py
                "ps_ops5": (0x12340010, 0xBEEF0800, 0xAAAAFFFF, 0xDEADBC00),   # from the .asm headers
                "ps_ops6": (0xF800FFFF, 0x01000010, 0x00800800, 0x8000FFF0),
                "ps_ops7": (0x00FF8077, 0xFF0000FF, 0x89ABCDEF, 0x00070003),
                "ps_ops8": (0x80000000, 0xFFFFFFFF, 0x000F100E, 0x0F0F0F1F),
                "ps_f32_mods": (0xC0256A01, 0x3F800000, 0x41E00000, 0x3FC00000),
                "ps_vcc_x2": (0x11223344, 0x55667788, 0, 0),
                "ps_med3_mods": (0x3F800000, 0x40400000, 0x40C00000, 0x3F400000),
                "ps_scc_minmax": (0x00030205, 0x00030207, 0x00030209, 0x00030203),
                "ps_fma_literal": (0x41B80000, 0x41800000, 0x40E00000, 0x41A40000),
                "ps_f64_mods": (0x40040000, 0x3FF00000, 0x3FE40000, 0x3FF00000),   # needs shaderFloat64
                # f16 op_sel / partial writes / inline constants: RDNA2 hardware words (see the .asm headers)
                "ps_f16_med3_opsel": (0xAAAA4100, 0xAAAA4200, 0xAAAA3E00, 0x40005555),
                "ps_f16_med3_mods": (0x41005555, 0x3C005555, 0x41003800, 0xAAAABA00),
                "ps_f16_vop2": (0xAAAA4000, 0xAAAAC000, 0xAAAA3A00, 0x42003F00),
                "ps_f16_vop2b": (0xAAAA3E00, 0xAAAAB800, 0xAAAA3C00, 0x42003400),
                "ps_f16_vop1": (0xAAAA4000, 0xAAAABC00, 0xAAAAC000, 0xAAAA8000),
                "ps_f16_vop1b": (0xAAAA3400, 0xAAAA4000, 0xAAAAB400, 0xAAAA3800),
                "ps_f16_const": (0xAAAA3D00, 0x38005555, 0xAAAA3800, 0x3F000000),
                "ps_f16_const2": (0x42003D00, 0x3D004200, 0xAAAABC00, 3),
                "ps_f16_pack": (0x34003C00, 0x38003400, 0x34000000, 0x44004200),
                "ps_sff_imm": (2, 1, 20, 16),                  # SHADER-46: s_ff0/s_ff1 of an immediate
            }
            if name in const_wants:
                ok, detail = check_pixel_const(raw, const_wants[name])
            elif name == "ps_bary":
                ok, detail = check_bary(raw)
            elif name == "ps_bitmask_perm":
                m = re.search(r"subgroup \d+\.\.(\d+)", msg)
                if m and int(m.group(1)) < 32:
                    ok, detail = None, f"SKIP (subgroup {m.group(1)} is narrower than 32 lanes)"
                else:
                    ok, detail = check_bitmask_perm(raw)
            elif name in ("ps_uniform_branch", "ps_scc_mask32"):   # same expected pattern (see the .asm headers)
                ok, detail = check_uniform_branch(raw)
            elif name == "ps_wqm":
                ok, detail = check_wqm(raw)
            elif name == "ps_flat":
                ok, detail = check_flat(raw)
            else:
                ok, detail = False, "no checker defined"
            results.append((name, ok, detail))
        elif name == "gs_tri":
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_GS=spv)
            rc, msg = run_cmd([vk_harness, vs_tri, fs_loc1, raw], env=env)
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_grad(raw, 0.0, 0.0)
                results.append((name, ok, detail))
        elif name == "ms_v5_workgroup":
            rc, msg = run_cmd(["spirv-val", "--target-env", "vulkan1.2", spv])
            if rc:
                results.append((name, False, msg))
                continue
            problem = mesh_v5_test(a.build, spirv_dir, Path(raw_dir) / f"{name}.raw")
            results.append((name, None if isinstance(problem, Skip) else not problem,
                            problem or "24/24 workgroup coordinates exact (2 x 3 x 4)"))
        elif name in ("ms_tri", "ms_w64x_ballot", "ms_wave_info_s3"):
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_MESH="1")
            rc, msg = run_cmd([vk_harness, spv, fs_passthru, raw], env=env)
            # ms_w64x_ballot needs exact wave64 (a 64-wide subgroup, or --wave64-emulate): skip it elsewhere, as
            # the wave64 compute dumps are.
            no64 = (name == "ms_w64x_ballot" and not w64_emulated() and
                    not any(int(x) >= 64 for x in re.findall(r"subgroup \d+\.\.(\d+)", msg)))
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            elif no64:
                results.append((name, None, "SKIP (no 64-wide subgroup on this device)"))
            else:
                ok, detail = check_grad(raw, 0.5, 1.0)
                results.append((name, ok, detail))
        elif name == "ms_clip":
            raw = os.path.join(raw_dir, f"{name}.raw")
            env = dict(gfx_env or os.environ, VKH_MESH="1")
            rc, msg = run_cmd([vk_harness, spv, fs_passthru, raw], env=env)
            if rc == UNSUPPORTED:
                results.append((name, None, unsupported_skip(msg)))
            elif rc != 0:
                results.append((name, False, msg))
            else:
                ok, detail = check_clip_half(raw, 0.5, 1.0)
                results.append((name, ok, detail))

    print("\n=== GPU TEST RESULTS ===")
    passed = skipped = 0
    for name, ok, detail in results:
        tag = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        passed += ok is True
        skipped += ok is None
        print(f"  {tag:4s}  {name:<20s}  {detail}")
    print(f"\n{passed}/{len(results)} passed" + (f", {skipped} skipped" if skipped else ""))
    return 0 if passed + skipped == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
