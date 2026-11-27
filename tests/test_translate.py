"""Regression tests for tools/rdna2spirv.py (RDNA2 -> SPIR-V).

Run from the repository root:  python -m pytest tests
Needs llvm-mc (AMDGPU target) and spirv-tools; the tests skip themselves when those are missing.
"""
import glob
import os
import re

import pytest

import rdna2spirv
from conftest import EMPTY_DXBC, KNOWN_VAL_ISSUES, ROOT, SAMPLE_DXBC, make_dxbc, mesh_psv0_chunk, requires_tools, signature_chunk

pytestmark = requires_tools

SAMPLES = sorted(glob.glob(os.path.join(ROOT, "runtime", "test", "shaders", "*.asm")))

# One minimal pixel shader per instruction family listed in the "Supported" section of rdna2spirv.py's docstring.
# Each body is wrapped in a program that exports its result so nothing is optimised away.
SUPPORTED = {
    "v_mov": "v_mov_b32 v0, 1.0",
    "v_add_f32": "v_mov_b32 v1, 1.0\nv_add_f32 v0, v1, v1",
    "v_sub_f32": "v_mov_b32 v1, 2.0\nv_sub_f32 v0, v1, 1.0",
    "v_mul_f32": "v_mov_b32 v1, 2.0\nv_mul_f32 v0, v1, v1",
    "v_fma_f32": "v_mov_b32 v1, 2.0\nv_fma_f32 v0, v1, v1, v1",
    "v_max_f32": "v_mov_b32 v1, 2.0\nv_max_f32 v0, v1, 1.0",
    "v_min_f32": "v_mov_b32 v1, 2.0\nv_min_f32 v0, v1, 1.0",
    "v_rcp_f32": "v_mov_b32 v1, 4.0\nv_rcp_f32 v0, v1",
    "v_sqrt_f32": "v_mov_b32 v1, 4.0\nv_sqrt_f32 v0, v1",
    "v_floor_f32": "v_mov_b32 v1, 1.5\nv_floor_f32 v0, v1",
    "v_fract_f32": "v_mov_b32 v1, 1.5\nv_fract_f32 v0, v1",
    "v_cvt_f32_u32": "v_mov_b32 v1, 7\nv_cvt_f32_u32 v0, v1",
    "v_cvt_u32_f32": "v_mov_b32 v1, 7.0\nv_cvt_u32_f32 v0, v1",
    "v_add_nc_u32": "v_mov_b32 v1, 7\nv_add_nc_u32 v0, v1, v1",
    "v_and_b32": "v_mov_b32 v1, 7\nv_and_b32 v0, 3, v1",
    "v_lshlrev_b32": "v_mov_b32 v1, 7\nv_lshlrev_b32 v0, 2, v1",
    "v_lshrrev_b32": "v_mov_b32 v1, 7\nv_lshrrev_b32 v0, 1, v1",
    "v_bfe_u32": "v_mov_b32 v1, 0xf0\nv_bfe_u32 v0, v1, 4, 4",
    "v_med3_f32": "v_mov_b32 v1, 2.0\nv_med3_f32 v0, v1, 1.0, 4.0",
    "s_mov_b32": "s_mov_b32 s40, 5\nv_mov_b32 v0, s40",
    "s_add_u32": "s_mov_b32 s40, 5\ns_add_u32 s41, s40, 3\nv_mov_b32 v0, s41",
}


def program(body):
    return f"{body}\nv_mov_b32 v8, v0\nexp mrt0 v8, v8, v8, v8 done vm\ns_endpgm\n"


@pytest.mark.parametrize("path", SAMPLES, ids=[os.path.basename(p) for p in SAMPLES])
def test_gpu_samples_translate_and_validate(path, make_shader):
    """The hand-written GPU-harness samples must translate to SPIR-V that passes spirv-val."""
    stem = os.path.splitext(os.path.basename(path))[0]
    name = stem if stem.startswith(("vs_", "ps_", "cs_", "gs_", "ms_")) else "ps_" + stem   # the prefix selects the stage
    if name.startswith(("gs_", "ms_")) and stem not in SAMPLE_DXBC:
        pytest.skip(f"no synthetic DXBC (ISG1/OSG1 for gs, PSV0 for ms) defined for {stem} in conftest.SAMPLE_DXBC")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), name, dxbc=SAMPLE_DXBC.get(stem, EMPTY_DXBC))
    shader.validate(shader.translate(), KNOWN_VAL_ISSUES.get(stem))


@pytest.mark.parametrize("case", sorted(SUPPORTED))
def test_supported_instruction(case, make_shader):
    shader = make_shader(program(SUPPORTED[case]))
    shader.validate(shader.translate())


def test_translation_is_deterministic(make_shader):
    shader = make_shader(program(SUPPORTED["v_fma_f32"]))
    assert shader.translate() == shader.translate()


def test_unsupported_instruction_is_reported_not_guessed(make_shader):
    """Anything outside the supported set must raise Unsupported rather than emit a wrong program."""
    shader = make_shader(program("v_mov_b32 v1, 1.0\nv_sat_pk_u8_i16 v0, v1"))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("d16", [False, True], ids=["f32", "d16"])
def test_depth_compare_sample_result_width(make_shader, d16):
    """image_sample_c_lz returns one value: f32 bits, or with d16 an f16 in the low half like every other d16 sample
    result. Written as f32 bits under d16, every compare result would read back as 0 (1.0 = 0x3f800000)."""
    shader = make_shader("v_mov_b32 v0, 0.5\nv_mov_b32 v1, 0.5\nv_mov_b32 v2, 0.25\n"
                         "image_sample_c_lz v4, v[0:2], s[0:7], s[8:11] dmask:0x1 dim:SQ_RSRC_IMG_2D"
                         + (" d16" if d16 else "") + "\nexp mrt0 v4, v4, v4, v4 done vm\ns_endpgm\n", "ps_dref")
    spvasm = shader.translate()
    shader.validate(spvasm)
    dref = re.search(r"^\s*(%\w+) = OpImageSampleDrefExplicitLod %float ", spvasm, re.M)
    assert dref, "the depth compare sample is missing"
    d = re.escape(dref.group(1))
    bitcast = re.search(rf"^\s*(%\w+) = OpBitcast %uint {d}$", spvasm, re.M)
    assert bitcast, "the depth compare bitcast is missing"
    bc = re.escape(bitcast.group(1))
    as_f16 = re.search(rf"^\s*%\w+ = OpShiftRightLogical %uint {bc} %c10$", spvasm, re.M)
    assert bool(as_f16) == d16



@pytest.mark.parametrize("export,depth", [
    ("exp mrtz v2, off, off, off vm", True),               # Z only (SPI_SHADER_32_R)
    ("exp mrtz v2, v3, off, off vm", True),                # Z and stencil (32_GR)
    ("exp mrtz off, v3, off, off vm", False),              # stencil only (32_GR)
    ("exp mrtz off, off, v3, off vm", False),              # sample mask only, 32-bit
    ("exp mrtz v2, v2, v3, v3 compr vm", False),           # stencil and sample mask, 16-bit (UINT16_ABGR)
], ids=["z", "z-stencil", "stencil", "mask", "compr"])
def test_mrtz_depth_only_from_an_enabled_x(make_shader, export, depth):
    """exp mrtz carries Z in x only when x is on in an uncompressed export; the 16-bit form holds stencil and sample
    mask, never Z (Mesa ac_nir_lower_ps_late.c emit_ps_mrtz_export). Without a depth the shader must not replace it."""
    shader = make_shader("v_mov_b32 v2, 0.5\nv_mov_b32 v3, 1\n" + export + "\nexp mrt0 v2, v2, v2, v2 done vm\n"
                         "s_endpgm\n", "ps_mrtz")
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert ("BuiltIn FragDepth" in spvasm) == depth
    assert ("DepthReplacing" in spvasm) == depth

def test_a16_sample_gradients_use_full_vgprs(make_shader):
    path = os.path.join(ROOT, "runtime", "test", "shaders", "ps_texture_grad_a16.asm")
    with open(path, encoding="utf-8") as fixture:
        shader = make_shader(fixture.read(), "ps_texture_grad_a16")
    spvasm = shader.translate()
    sample = re.search(r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Grad (%\w+) (%\w+)$", spvasm, re.M)
    assert sample, "the explicit-gradient sample is missing"
    definitions = dict(re.findall(r"^\s*(%\w+) = (Op[^\n]+)", spvasm, re.M))

    # Trace dPdx and dPdy back to the fixture's full 32-bit derivative words, in operand order.
    for vector, registers in zip(sample.groups(), ((8, 9), (10, 11))):
        assert definitions[vector].startswith("OpCompositeConstruct %v2float ")
        components = definitions[vector].split()[2:]
        assert len(components) == 2
        for component, register in zip(components, registers):
            cast, float_type, word = definitions[component].split()
            assert (cast, float_type) == ("OpBitcast", "%float")
            load, word_type, pointer = definitions[word].split()
            assert (load, word_type) == ("OpLoad", "%uint")
            *access, index = definitions[pointer].split()
            assert access == ["OpAccessChain", "%_ptr_Function_uint", "%vgpr"]
            assert definitions[index] == f"OpConstant %uint {register}"
    shader.validate(spvasm)


def test_unknown_stage_is_rejected(make_shader):
    shader = make_shader(program("v_mov_b32 v0, 1.0"), name="xx_test")
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_compute_translates_and_validates(make_shader):
    shader = make_shader("v_mov_b32 v1, 1\ns_endpgm\n", name="cs_test")
    shader.validate(shader.translate())


def test_cs_store_packed_atomic_pointers(make_shader):
    """Packed stores need element zero of the physical dword array for Metal atomics."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_store.asm"), encoding="utf-8") as f:
        spvasm = make_shader(f.read(), "cs_store").translate()
    defs = dict(re.findall(r"^\s*(%\w+) = (Op[^\n]+)", spvasm, re.M))
    atomics = re.findall(r"= (OpAtomicAnd|OpAtomicOr) %uint (%\w+) ", spvasm)
    assert {op for op, ptr in atomics} == {"OpAtomicAnd", "OpAtomicOr"}
    for _, ptr in atomics:
        access = defs[ptr].split()
        assert access[:2] == ["OpAccessChain", "%_ptr_PSB_uint"]
        _, _, base, index = access
        assert defs[base].startswith("OpConvertUToPtr %_ptr_PSB_arr ")
        assert defs[index] == "OpConstant %uint 0"


def test_compute_thread_group_size_sgpr_is_rejected(make_shader):
    """TG_SIZE_EN (RSRC2 bit 10) adds an SGPR the translator doesn't model."""
    shader = make_shader("s_endpgm\n", name="cs_test", cs_header=(0x8088 | 1 << 10, (64, 1, 1)))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("group", [(0, 1, 1), (2048, 1, 1)], ids=["empty", "over-1024"])
def test_compute_bad_group_size_is_rejected(make_shader, group):
    shader = make_shader("s_endpgm\n", name="cs_test", cs_header=(0x8088, group))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_geometry_without_signatures_is_rejected(make_shader):
    """A gs needs ISG1/OSG1 to know which registers LDS holds and which outputs to write."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "gs_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "gs_tri")                   # empty DXBC
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


@pytest.mark.parametrize("kw", [dict(topology=1), dict(threads=(2048, 1, 1)), dict(max_vertices=0)],
                         ids=["not-triangles", "over-1024-threads", "no-vertices"])
def test_mesh_bad_psv0_is_rejected(make_shader, kw):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_tri", dxbc=make_dxbc([mesh_psv0_chunk(**kw)]))
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_mesh_without_psv0_is_rejected(make_shader):
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_tri")                   # empty DXBC
    with pytest.raises(rdna2spirv.Unsupported):
        shader.translate()


def test_wave32_readlane_writelane_mask_31(make_shader):
    """SHADER-52: wave32 lane select is & 31; wave64 keeps & 63."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w32_lane_wrap.asm"), encoding="utf-8") as f:
        w32 = make_shader(f.read(), name="cs_w32_lane_wrap", cs_header=(0x8088, (32, 1, 1)))
    w32_spv = w32.translate()
    w32.validate(w32_spv)
    assert "OpBitwiseAnd %uint" in w32_spv and "%c1f" in w32_spv  # 31
    # Every readlane/writelane mask in this wave32 program is 31, never 63.
    assert not re.search(r"OpBitwiseAnd %uint %\w+ %c3f\b", w32_spv)
    assert rdna2spirv.translate.last_gen.wave64 is False

    with open(os.path.join(ROOT, "runtime", "test", "shaders", "cs_w64x_readlane.asm"), encoding="utf-8") as f:
        w64 = make_shader(f.read(), name="cs_w64x_readlane")
    w64_spv = w64.translate()
    assert rdna2spirv.translate.last_gen.wave64 is True
    assert re.search(r"OpBitwiseAnd %uint %\w+ %c3f\b", w64_spv)  # 63


@pytest.mark.parametrize("threads", [96, 128], ids=["96-threads", "128-threads"])
def test_mesh_v5_seed_and_threads_constant(make_shader, threads):
    """Mesh prologue computes v5 = %wlin * NumThreads + %lidx and registers NumThreads before assembly (7b82460)."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_tri.asm"), encoding="utf-8") as f:
        asm = f.read()
    dxbc = make_dxbc([mesh_psv0_chunk(threads=(threads, 1, 1), max_vertices=64, max_primitives=64)])
    shader = make_shader(asm, name=f"ms_v5_seed_{threads}", dxbc=dxbc)
    spvasm = shader.translate()
    cid = f"%c{threads:x}"
    assert f"{cid} = OpConstant %uint {threads}" in spvasm
    assert f"%wbase = OpIMul %uint %wlin {cid}" in spvasm
    assert "%v5v = OpIAdd %uint %wbase %lidx" in spvasm
    assert "%v5p = OpAccessChain %_ptr_Function_uint %vgpr %c5" in spvasm
    assert "OpStore %v5p %v5v" in spvasm
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
def test_mesh_cull_primitive_is_block_bool(make_shader):
    """CullPrimitiveEXT must be a Block bool member (VUID-07036); array-of-bool trips SPIRV-Tools ≤2025.1."""
    with open(os.path.join(ROOT, "runtime", "test", "shaders", "ms_v5_workgroup.asm"), encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_v5_workgroup", dxbc=SAMPLE_DXBC["ms_v5_workgroup"])
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpDecorate %ms_cull_blk Block" in spvasm
    assert "OpMemberDecorate %ms_cull_blk 0 BuiltIn CullPrimitiveEXT" in spvasm
    assert "OpMemberDecorate %ms_cull_blk 0 PerPrimitiveEXT" in spvasm
    assert "%ms_cull_blk = OpTypeStruct %bool" in spvasm
    assert re.search(r"%ms_arr_cull = OpTypeArray %ms_cull_blk %c\w+", spvasm)
    assert "OpDecorate %ms_cull BuiltIn CullPrimitiveEXT" not in spvasm
    assert re.search(r"OpAccessChain %_ptr_Out_bool_ms %ms_cull %\w+ %c0", spvasm)


# ---- pos1..pos3: ClipDistance / CullDistance / misc exports (VS & MS) ----

def _vgpr_to_index_map(spvasm, array_var, ptr_type):
    """Trace each OpStore to `array_var[idx]` back to the VGPR index loaded from `%vgpr`."""
    import re
    vgpr_of = {}
    val_of = {}
    ptr_idx = {}
    stores = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr %c([0-9a-f]+)", line)
        if m:
            vgpr_of[m.group(1)] = int(m.group(2), 16)
            continue
        m = re.match(r"\s*(%\w+) = OpLoad %uint (%\w+)", line)
        if m and m.group(2) in vgpr_of:
            val_of[m.group(1)] = vgpr_of[m.group(2)]
            continue
        m = re.match(r"\s*(%\w+) = OpBitcast %float (%\w+)", line)
        if m and m.group(2) in val_of:
            val_of[m.group(1)] = val_of[m.group(2)]
            continue
        m = re.match(rf"\s*(%\w+) = OpAccessChain {re.escape(ptr_type)} {re.escape(array_var)} %c(\d+)", line)
        if m:
            ptr_idx[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"\s*(%\w+) = OpSelect %float %\w+ (%\w+) %\w+", line)
        if m and m.group(2) in val_of:
            val_of[m.group(1)] = val_of[m.group(2)]
            continue
        m = re.match(r"\s*OpStore (%\w+) (%\w+)", line)
        if m and m.group(1) in ptr_idx and m.group(2) in val_of:
            stores[ptr_idx[m.group(1)]] = val_of[m.group(2)]
    return stores


def _ms_header_asm(pos_exports):
    return (
        "v_mbcnt_lo_u32_b32 v0, -1, 0\n"
        "v_mbcnt_hi_u32_b32 v0, -1, v0\n"
        "s_mov_b32 m0, 0x1003\n"
        "s_sendmsg sendmsg(MSG_GS_ALLOC_REQ)\n"
        "v_mov_b32 v1, 0.0\n"
        "v_mov_b32 v2, 1.0\n"
        "v_mov_b32 v10, 0.25\n"
        "v_mov_b32 v11, 0.5\n"
        "v_mov_b32 v12, 1.0\n"
        "v_mov_b32 v13, 2.0\n"
        "v_mov_b32 v14, 4.0\n"
        "exp pos0 v1, v1, v1, v2\n"
        f"{pos_exports}\n"
        "v_mov_b32 v9, 0x200400\n"
        "exp prim v9, off, off, off done\n"
        "s_endpgm\n"
    )


def _vs_header_asm(pos_exports):
    return (
        "v_mov_b32 v1, 0.0\n"
        "v_mov_b32 v2, 1.0\n"
        "v_mov_b32 v10, 0.25\n"
        "v_mov_b32 v11, 0.5\n"
        "v_mov_b32 v12, 1.0\n"
        "v_mov_b32 v13, 2.0\n"
        "v_mov_b32 v14, 4.0\n"
        "exp pos0 v1, v1, v1, v2\n"
        f"{pos_exports}\n"
        "s_endpgm\n"
    )


@pytest.mark.parametrize("n_comp,exp_line,mask", [
    (1, "exp pos1 v10, off, off, off done", 0x1),
    (4, "exp pos1 v10, v11, v12, v13 done", 0xF),
], ids=["1-comp", "4-comp"])
def test_vs_clip_distance_pos1(make_shader, n_comp, exp_line, mask):
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, mask)])])
    shader = make_shader(_vs_header_asm(exp_line), name="vs_clip_test", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" not in spvasm
    assert "OpDecorate %out_clip BuiltIn ClipDistance" in spvasm
    assert f"%arr_clip = OpTypeArray %float %c{n_comp}" in spvasm
    chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    assert [int(x) for x in chains] == list(range(n_comp))
    assert _vgpr_to_index_map(spvasm, "%out_clip", "%_ptr_Output_float") == {i: 10 + i for i in range(n_comp)}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" not in meta["features"]
    assert 1 not in meta.get("outputs", [])


@pytest.mark.parametrize("n_comp,exp_line,mask", [
    (1, "exp pos1 v10, off, off, off done", 0x1),
    (4, "exp pos1 v10, v11, v12, v13 done", 0xF),
], ids=["1-comp", "4-comp"])
def test_ms_clip_distance_pos1(make_shader, n_comp, exp_line, mask):
    import re
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, mask)])])
    shader = make_shader(_ms_header_asm(exp_line), name="ms_clip_test", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" in spvasm
    assert "OpDecorate %ms_vtx_blk Block" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 1 BuiltIn ClipDistance" in spvasm
    assert f"%ms_arr_clip = OpTypeArray %float %c{n_comp}" in spvasm
    chains = re.findall(r"OpAccessChain %_ptr_Function_float %ms_clip %c(\d+)", spvasm)
    assert [int(x) for x in chains] == list(range(n_comp))
    assert _vgpr_to_index_map(spvasm, "%ms_clip", "%_ptr_Function_float") == {i: 10 + i for i in range(n_comp)}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert 1 not in meta.get("outputs", [])
    assert 1 not in meta.get("params", [])


def test_vs_clip_and_cull_distances_across_pos1_pos2(make_shader):
    """2 clip distances + 3 cull distances pack contiguously: pos1.xy=clip[0..1], pos1.zw=cull[0..1], pos2.x=cull[2]."""
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x3), (2, 3, 0x7)])])
    asm = _vs_header_asm("exp pos1 v10, v11, v12, v13\nexp pos2 v14, off, off, off done")
    shader = make_shader(asm, name="vs_clip_cull", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" in spvasm
    assert "%arr_clip = OpTypeArray %float %c2" in spvasm
    assert "%arr_cull = OpTypeArray %float %c3" in spvasm
    clip_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    cull_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_cull %c(\d+)", spvasm)
    assert [int(x) for x in clip_chains] == [0, 1]
    assert [int(x) for x in cull_chains] == [0, 1, 2]
    assert _vgpr_to_index_map(spvasm, "%out_clip", "%_ptr_Output_float") == {0: 10, 1: 11}
    assert _vgpr_to_index_map(spvasm, "%out_cull", "%_ptr_Output_float") == {0: 12, 1: 13, 2: 14}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" in meta["features"]
    assert not ({1, 2} & set(meta.get("outputs", [])))
    assert not ({1, 2} & set(meta.get("params", [])))


def test_vs_misc_pos1_shifts_clip_to_pos2(make_shader):
    """When OSG1 declares SV_RenderTargetArrayIndex (sysv=4), pos1 is the misc vector and ClipDistance starts at pos2."""
    import re
    dxbc = make_dxbc([signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 4, 0x1), (2, 2, 0x3)])])
    asm = _vs_header_asm("exp pos1 off, off, v14, off\nexp pos2 v10, v11, off, off done")
    shader = make_shader(asm, name="vs_misc_clip", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm)
    assert "OpCapability ShaderLayer" in spvasm
    assert "OpCapability ClipDistance" in spvasm
    assert "OpDecorate %out_layer BuiltIn Layer" in spvasm
    assert "%arr_clip = OpTypeArray %float %c2" in spvasm
    clip_chains = re.findall(r"OpAccessChain %_ptr_Output_float %out_clip %c(\d+)", spvasm)
    assert [int(x) for x in clip_chains] == [0, 1]


def test_pos1_without_osg1_is_ignored(make_shader):
    """Without OSG1 declaring SV_ClipDistance/SV_CullDistance, pos1 exports are discarded in both VS and MS."""
    vs = make_shader(_vs_header_asm("exp pos1 v10, off, off, off done"), name="vs_no_osg1_pos1")
    vs_spv = vs.translate()
    vs.validate(vs_spv)
    assert "OpCapability ClipDistance" not in vs_spv
    assert "BuiltIn ClipDistance" not in vs_spv
    vs_meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, vs_spv)
    assert "shaderClipDistance" not in vs_meta["features"]
    assert 1 not in vs_meta.get("outputs", [])

    ms = make_shader(_ms_header_asm("exp pos1 v10, off, off, off done"), name="ms_no_osg1_pos1",
                     dxbc=make_dxbc([mesh_psv0_chunk()]))
    ms_spv = ms.translate()
    ms.validate(ms_spv, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" not in ms_spv
    assert "BuiltIn ClipDistance" not in ms_spv
    assert "%ms_out_pos" in ms_spv and "%ms_vtx_blk" not in ms_spv
    ms_meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, ms_spv)
    assert "shaderClipDistance" not in ms_meta["features"]
    assert 1 not in ms_meta.get("params", [])


def test_ms_dummy_all_off_pos1_does_not_emit_clip(make_shader):
    """An all-off dummy pos1 export in a mesh shader must not emit ClipDistance or change the output block."""
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x1)])])
    shader = make_shader(_ms_header_asm("exp pos1 off, off, off, off done"), name="ms_dummy_pos1", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" not in spvasm
    assert "%ms_out_pos" in spvasm
    assert "%ms_vtx_blk" not in spvasm


def test_ms_clip_and_cull_distances_across_pos1_pos2(make_shader):
    """Mesh shader with 2 clip + 3 cull distances emits both arrays in %ms_vtx_blk and maps v10..v14."""
    dxbc = make_dxbc([mesh_psv0_chunk(), signature_chunk(b"OSG1", [(0, 1, 0xF), (1, 2, 0x3), (2, 3, 0x7)])])
    asm = _ms_header_asm("exp pos1 v10, v11, v12, v13\nexp pos2 v14, off, off, off done")
    shader = make_shader(asm, name="ms_clip_cull", dxbc=dxbc)
    spvasm = shader.translate()
    shader.validate(spvasm, KNOWN_VAL_ISSUES.get("ms_tri"))
    assert "OpCapability ClipDistance" in spvasm
    assert "OpCapability CullDistance" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 1 BuiltIn ClipDistance" in spvasm
    assert "OpMemberDecorate %ms_vtx_blk 2 BuiltIn CullDistance" in spvasm
    assert "%ms_arr_clip = OpTypeArray %float %c2" in spvasm
    assert "%ms_arr_cull_d = OpTypeArray %float %c3" in spvasm
    assert _vgpr_to_index_map(spvasm, "%ms_clip", "%_ptr_Function_float") == {0: 10, 1: 11}
    assert _vgpr_to_index_map(spvasm, "%ms_cull_dist", "%_ptr_Function_float") == {0: 12, 1: 13, 2: 14}
    meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spvasm)
    assert "shaderClipDistance" in meta["features"]
    assert "shaderCullDistance" in meta["features"]
    assert not ({1, 2} & set(meta.get("outputs", [])))
    assert not ({1, 2} & set(meta.get("params", [])))


def test_wave64_detected_from_mask_ops_without_exec_markers(make_shader):
    """#210 part 2 / SHADER-28: s_bcnt1_i32_b64 on ordinary SGPRs marks the shader wave64 (no exec_hi / saveexec)."""
    shader = make_shader(
        "; mattr: +wavefrontsize64,-wavefrontsize32\n"
        "v_and_b32_e32 v5, 5, v0\nv_cmp_ne_u32_e64 s[20:21], 0, v5\n"
        "s_bcnt1_i32_b64 s30, s[20:21]\nv_mov_b32 v1, s30\ns_endpgm\n",
        name="cs_mask_detect_ci",
    )
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64


def test_wave64_bfm64_and_half_cmp_emit_wave_ops(make_shader):
    """Wave64 s_bfm_b64 must use wave_lane() and reconstruct the 64-bit mask; 32-bit half ops must guard their half."""
    bfm = make_shader("s_mov_b64 exec, -1\ns_bfm_b64 s[6:7], 16, 8\ns_bcnt1_i32_b64 s8, s[6:7]\ns_endpgm\n",
                      name="cs_bfm64_ci")
    asm_bfm = bfm.translate()
    bfm.validate(asm_bfm)
    assert "OpGroupNonUniformBitwiseOr" in asm_bfm
    assert "OpLoad %v3uint %local_id" in asm_bfm and "OpLoad %uint %sg_lane" not in asm_bfm

    hcmp = make_shader("s_mov_b32 exec_lo, 0\ns_mov_b32 exec_hi, 0xffff\ns_mov_b64 s[6:7], exec\n"
                       "s_cmp_eq_u32 s6, 0\ns_cmp_lg_u32 s7, 0\ns_endpgm\n", name="cs_halfcmp_ci")
    asm_hcmp = hcmp.translate()
    hcmp.validate(asm_hcmp)
    # Low-half s_cmp_eq_u32: lane < 32 (%c20) guards OpSelect(..., %true) feeding OpGroupNonUniformAll.
    assert re.search(
        r"(%t\d+) = OpULessThan %bool %t\d+ %c20\n\s*"
        r"(%t\d+) = OpSelect %bool \1 %t\d+ %true\n\s*"
        r"%t\d+ = OpGroupNonUniformAll %bool %c3 \2",
        asm_hcmp,
    ), "low-half s_cmp_eq_u32 did not guard OpGroupNonUniformAll with lane < 32"
    # High-half s_cmp_lg_u32: lane >= 32 (%c20) guards OpLogicalAnd feeding OpGroupNonUniformAny.
    assert re.search(
        r"(%t\d+) = OpUGreaterThanEqual %bool %t\d+ %c20\n\s*"
        r"(%t\d+) = OpLogicalAnd %bool \1 %t\d+\n\s*"
        r"%t\d+ = OpGroupNonUniformAny %bool %c3 \2",
        asm_hcmp,
    ), "high-half s_cmp_lg_u32 did not guard OpGroupNonUniformAny with lane >= 32"

    # Copying vcc_lo / vcc_hi to a 32-bit SGPR preserves half identity for wave64 s_bcnt1_i32_b32.
    vcc_half = make_shader(
        "s_mov_b64 exec, -1\nv_cmp_gt_u32_e32 vcc_lo, 16, v0\n"
        "s_mov_b32 s6, vcc_lo\ns_bcnt1_i32_b32 s8, s6\n"
        "s_mov_b32 s7, vcc_hi\ns_bcnt1_i32_b32 s9, s7\ns_endpgm\n",
        name="cs_vcchalf_ci",
    )
    asm_vcc = vcc_half.translate()
    vcc_half.validate(asm_vcc)
    assert asm_vcc.count("OpBitCount %uint %c0") == 2


def test_wave64_bpermute_reads_zero_from_lanes_that_are_not_running(make_shader):
    """Native wave64 ds_bpermute: a source lane missing from a partial wave must read 0, not an undefined shuffle."""
    shader = make_shader("s_mov_b64 exec, -1\nv_lshlrev_b32_e32 v1, 2, v0\nds_bpermute_b32 v2, v1, v0\n"
                         "s_waitcnt lgkmcnt(0)\ns_endpgm\n", name="cs_bperm64_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64
    assert re.search(
        r"(%t\d+) = OpGroupNonUniformBallot %v4uint_bal %c3 %true\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallotBitExtract %bool %c3 \1 (%t\d+)\n\s*"
        r"(%t\d+) = OpGroupNonUniformShuffle %uint %c3 %t\d+ \3\n\s*"
        r"%t\d+ = OpSelect %uint \2 \4 %c0\n",
        asm,
    ), "wave64 ds_bpermute did not keep the shuffle only where the source lane is running"



def test_wave32_bpermute_zeroes_inactive_and_missing_source_lanes(make_shader):
    """Wave32 ds_bpermute (RDNA ISA 10.4.4): the source value is 0 where the source lane is off in EXEC, and the
    shuffle result is kept only where the source lane is running, as in the wave64 path."""
    shader = make_shader("v_lshlrev_b32_e32 v1, 2, v0\nds_bpermute_b32 v2, v1, v0\ns_waitcnt lgkmcnt(0)\ns_endpgm\n",
                         name="cs_bperm32_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert not rdna2spirv.translate.last_gen.wave64
    assert re.search(
        r"(%t\d+) = OpLoad %bool %active\n\s*"
        r"(%t\d+) = OpSelect %uint \1 %t\d+ %c0\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallot %v4uint_bal %c3 %true\n\s*"
        r"(%t\d+) = OpGroupNonUniformBallotBitExtract %bool %c3 \3 (%t\d+)\n\s*"
        r"(%t\d+) = OpGroupNonUniformShuffle %uint %c3 \2 \5\n\s*"
        r"%t\d+ = OpSelect %uint \4 \6 %c0\n",
        asm,
    ), "wave32 ds_bpermute did not zero inactive sources and gate the shuffle on the running source lane"

WQM_MASK = "s_mov_b64 exec, -1\nv_cmp_gt_u32_e32 vcc_lo, 16, v0\ns_wqm_b64 s[6:7], vcc\ns_bcnt1_i32_b64 s8, s[6:7]\ns_endpgm\n"
WQM_DATA = "s_mov_b32 s20, 0x00800010\ns_mov_b32 s21, 1\ns_wqm_b64 s[22:23], s[20:21]\ns_add_u32 s24, s22, s23\ns_endpgm\n"


def test_wqm_on_a_lane_mask_gathers_the_quad_and_votes_scc(make_shader, monkeypatch):
    """s_wqm on a lane mask ORs the lane's bit with its quad's (two OpGroupNonUniformQuadSwap, horizontal then
    vertical) and SCC is the wave-wide any. Emulated, the four lanes of the quad come through %x_buf instead."""
    shader = make_shader(WQM_MASK, name="cs_wqm_mask_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert "OpCapability GroupNonUniformQuad" in asm
    assert asm.count("OpGroupNonUniformQuadSwap %uint %c3") == 2
    assert re.search(r"OpGroupNonUniformQuadSwap %uint %c3 %t\d+ %c0\n(?:.*\n)*?.*OpGroupNonUniformQuadSwap %uint %c3 %t\d+ %c1\n", asm)
    assert "OpGroupNonUniformAny" in asm                   # SCC: any bit of the result in the wave
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    emulated = shader.translate()
    shader.validate(emulated)
    assert "OpGroupNonUniform" not in emulated and "GroupNonUniformQuad" not in emulated
    assert "%x_buf" in emulated


def test_wqm_on_data_is_a_bit_operation(make_shader):
    """A register that is not a lane mask is data: per 32-bit word (x | x>>1 | x>>2 | x>>3) & 0x11111111, times 15."""
    shader = make_shader(WQM_DATA, name="cs_wqm_data_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert "OpGroupNonUniform" not in asm
    for k in (1, 2, 3):                                  # three shifts per word, two words (s_wqm_b64)
        assert len(re.findall(rf"OpShiftRightLogical %uint %t\d+ %c{k}\n", asm)) == 2
    assert len(re.findall(r"OpBitwiseAnd %uint %t\d+ %c11111111\n", asm)) == 2
    assert len(re.findall(r"OpIMul %uint %t\d+ %cf\n", asm)) == 2
    assert re.search(r"OpBitwiseOr %uint %t\d+ %t\d+\n(?:.*\n)*?.*OpStore %scc", asm)   # SCC = low | high != 0


def test_wqm_without_quad_operations_outside_fragment_and_compute(make_shader):
    """Quad operations are only guaranteed in fragment and compute shaders: a vertex shader shuffles from the four
    lanes of the quad instead."""
    shader = make_shader("v_cmp_gt_u32_e32 vcc_lo, 1, v0\ns_wqm_b32 s8, vcc_lo\nv_cndmask_b32_e64 v6, 0, 1, s8\n"
                         "exp pos0 v6, v6, v6, v6 done\ns_endpgm\n", name="vs_wqm_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert "QuadSwap" not in asm and "GroupNonUniformQuad" not in asm
    assert asm.count("OpGroupNonUniformShuffle") == 4


def test_wave64_wqm_b32_changes_and_votes_only_its_half(make_shader):
    """s_wqm_b32 exec_hi in wave64 writes the lanes 32-63 of EXEC only (exec_half_write) and SCC votes over them."""
    shader = make_shader("s_mov_b32 exec_hi, 0x102\ns_wqm_b32 exec_hi, exec_hi\ns_cselect_b32 s9, 1, 0\ns_endpgm\n",
                         name="cs_wqm_half_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.wave64
    assert re.search(r"(%t\d+) = OpUGreaterThanEqual %bool %t\d+ %c20\n(?:.*\n)*?.*OpLogicalAnd %bool \1 %t\d+\n"
                     r"(?:.*\n)*?.*OpGroupNonUniformAny", asm)


def test_wqm_in_a_pixel_shader_uses_the_fragment_quad(make_shader):
    shader = make_shader("v_cmp_gt_u32_e32 vcc_lo, 1, v0\ns_wqm_b32 s8, vcc_lo\nv_cndmask_b32_e64 v6, 0, 1, s8\n"
                         "exp mrt0 v6, v6, v6, v6 done vm\ns_endpgm\n", name="ps_wqm_ci")
    asm = shader.translate()
    shader.validate(asm)
    assert asm.count("OpGroupNonUniformQuadSwap") == 2 and "OpCapability GroupNonUniformQuad" in asm


EMULATE_SAMPLES = [p for p in SAMPLES if os.path.basename(p).startswith(("cs_", "ms_"))]


@pytest.mark.parametrize(("branch", "group", "expect_scheduler"), [
    pytest.param("", (64, 1, 1), False, id="straight-line"),
    pytest.param("s_cmp_eq_u32 s0, 0\ns_cbranch_scc0 1\n", (64, 1, 1), True, id="branch"),
    pytest.param("", (128, 1, 1), True, id="two-waves"),
])
def test_wave64_lds_scheduler_selection(branch, group, expect_scheduler, make_shader, monkeypatch):
    source = ("; mattr: +wavefrontsize64,-wavefrontsize32\n"
              "s_mov_b64 exec, -1\n"
              "v_lshlrev_b32_e32 v1, 2, v0\n"
              "ds_write_b32 v1, v0\n" + branch +
              "v_mov_b32_e32 v2, v0\ns_barrier\nds_read_b32 v3, v1\ns_endpgm\n")
    shader = make_shader(source, "cs_lds_scheduler", cs_header=(0x8088, group))
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    asm = shader.translate()
    shader.validate(asm)
    assert rdna2spirv.translate.last_gen.emulate
    assert ("OpSwitch" in asm) == expect_scheduler
    if not expect_scheduler:
        write_ptr, read_ptr = re.findall(r"(%\w+) = OpAccessChain %\w+ %lds ", asm)
        write = asm.index(f"OpStore {write_ptr} ")
        read = asm.index(f"OpLoad %uint {read_ptr}\n")
        assert "OpControlBarrier" in asm[write:read]


@pytest.mark.parametrize("path", EMULATE_SAMPLES, ids=[os.path.basename(p) for p in EMULATE_SAMPLES])
def test_wave64_emulate_uses_no_subgroup_ops(path, make_shader, monkeypatch):
    """--wave64-emulate: a wave64 compute or mesh shader validates and uses no subgroup operation (everything
    cross-lane goes through %x_buf); a wave32 one comes out exactly as without the flag."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.startswith("ms_") and stem not in SAMPLE_DXBC:
        pytest.skip(f"no synthetic PSV0 for {stem} in conftest.SAMPLE_DXBC")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), stem, dxbc=SAMPLE_DXBC.get(stem, EMPTY_DXBC))
    plain = shader.translate()
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    emulated = shader.translate()
    if not rdna2spirv.translate.last_gen.wave64:
        assert emulated == plain
        return
    shader.validate(emulated, KNOWN_VAL_ISSUES.get(stem))
    assert "OpGroupNonUniform" not in emulated and "%sg_lane" not in emulated
    assert "%x_buf" in emulated and emulated.count("OpControlBarrier") >= 2


# SHADER-36: image_gather4_l(_o) runs every texel of its 2x2 footprint through the S#'s CLAMP_X/CLAMP_Y.
def _gather_l(s_word0, dynamic=False):
    return program(
        "s_mov_b32 s8, 0\ns_mov_b32 s9, 0\ns_mov_b32 s10, 0\ns_mov_b32 s11, 0\n"
        "s_mov_b32 s12, 0\ns_mov_b32 s13, 0\ns_mov_b32 s14, 0\ns_mov_b32 s15, 0\n"
        f"s_mov_b32 s16, {s_word0}\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        + ("s_add_u32 s16, s16, 0\n" if dynamic else "")
        + "v_mov_b32 v10, 0.5\nv_mov_b32 v11, 0.5\nv_mov_b32 v12, 0\n"
        "image_gather4_l v[0:3], v[10:12], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D")


def _count(text, op):
    return len(re.findall(rf"= {op} ", text))


@pytest.mark.parametrize("mx", range(4))
@pytest.mark.parametrize("my", range(4))
def test_gather4_l_literal_sampler_emits_only_the_code_its_modes_need(mx, my, make_shader):
    """CLAMP_X = bits [2:0], CLAMP_Y = [5:3]; 0 wrap, 1 mirror, 2 clamp, 3 mirror once. Each axis maps its two
    texel coordinates: wrap = 1 OpSMod, mirror = 1 OpSMod + 1 OpSLessThan, mirror once = 1 OpSGreaterThanEqual."""
    shader = make_shader(_gather_l(mx | my << 3))
    text = shader.translate()
    shader.validate(text)
    assert _count(text, "OpSMod") == 2 * (mx in (0, 1)) + 2 * (my in (0, 1))
    assert _count(text, "OpSLessThan") == 2 * (mx == 1) + 2 * (my == 1)
    assert _count(text, "OpSGreaterThanEqual") == 2 * (mx == 3) + 2 * (my == 3)
    assert "OpBitFieldUExtract" not in text      # the mode is known at translation time


def test_gather4_l_run_time_sampler_selects_the_mode_from_the_sgpr(make_shader):
    shader = make_shader(_gather_l(0x12, dynamic=True))
    text = shader.translate()
    shader.validate(text)
    assert _count(text, "OpBitFieldUExtract") == 2          # CLAMP_X and CLAMP_Y
    assert _count(text, "OpSMod") == 8 and _count(text, "OpSLessThan") == 4 and _count(text, "OpSGreaterThanEqual") == 4


@pytest.mark.parametrize("word0", [4, 0x30, 0x7, 0x8000])
def test_gather4_l_literal_border_or_unnormalized_sampler_is_unsupported(word0, make_shader):
    with pytest.raises(rdna2spirv.Unsupported):
        make_shader(_gather_l(word0)).translate()
# ---- Irreducible CFG node splitting (compute shaders) ----

IRREDUCIBLE_PATTERNS = {
    # 2-entry SCC with s_branch + s_cbranch_scc0 (all 1-dword insns; offsets are relative dwords from next_pc)
    "scc0_branch": """\
s_mov_b32 s1, 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc0 3
s_add_u32 s1, s1, 1
s_cmp_lt_u32 s1, 4
s_cbranch_scc0 4
s_add_u32 s1, s1, 2
s_cmp_lt_u32 s1, 8
s_cbranch_scc0 1
s_branch -7
v_mov_b32 v1, s1
s_endpgm
""",
    # 2-entry SCC with s_cbranch_scc1
    "scc1": """\
s_mov_b32 s1, 0
s_cmp_eq_u32 s0, 0
s_cbranch_scc1 3
s_add_u32 s1, s1, 1
s_cmp_ge_u32 s1, 4
s_cbranch_scc1 3
s_add_u32 s1, s1, 2
s_cmp_lt_u32 s1, 8
s_cbranch_scc1 -6
v_mov_b32 v1, s1
s_endpgm
""",
    # 2-entry SCC with s_cbranch_vccnz
    "vccnz": """\
s_mov_b32 s1, 0
v_cmp_gt_u32_e32 vcc_lo, 4, v0
s_cbranch_vccnz 3
s_add_u32 s1, s1, 1
v_cmp_gt_u32_e32 vcc_lo, 8, v0
s_cbranch_vccnz 3
s_add_u32 s1, s1, 2
v_cmp_lt_u32_e32 vcc_lo, 2, v0
s_cbranch_vccnz -6
v_mov_b32 v1, s1
s_endpgm
""",
    # Nested irreducible cycle {A, B} inside an outer reducible loop headed by H (pc=2)
    "nested_in_reducible": """\
s_mov_b32 s1, 0
s_mov_b32 s2, 0
s_add_u32 s2, s2, 1
s_cmp_eq_u32 s0, 0
s_cbranch_scc0 3
s_add_u32 s1, s1, 1
s_cmp_eq_u32 s1, 3
s_cbranch_scc1 3
s_add_u32 s1, s1, 2
s_cmp_lt_u32 s1, 6
s_cbranch_scc1 -6
s_cmp_lt_u32 s2, 4
s_cbranch_scc1 -11
v_mov_b32 v1, s1
s_endpgm
""",
}


@pytest.mark.parametrize("pattern", sorted(IRREDUCIBLE_PATTERNS))
def test_irreducible_cfg_node_splitting(pattern, make_shader, monkeypatch):
    """Irreducible CFGs (2-entry SCCs and nested irreducible cycles) structure via node splitting and pass spirv-val."""
    asm = IRREDUCIBLE_PATTERNS[pattern]
    shader = make_shader(asm, name=f"cs_irred_{pattern}")

    # Without node splitting, the shader genuinely falls back to the dispatch loop.
    monkeypatch.setattr(rdna2spirv.cfg_structure, "split_irreducible", lambda succ, **kw: None)
    spv_fallback = shader.translate()
    meta_fallback = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv_fallback)
    assert meta_fallback["control_flow"] == "dispatch loop"
    assert "OpSwitch" in spv_fallback

    # With node splitting enabled, the shader becomes structured and passes spirv-val.
    monkeypatch.undo()
    spv_struct = shader.translate()
    shader.validate(spv_struct)
    meta_struct = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv_struct)
    assert meta_struct["control_flow"] == "structured"
    assert "OpLoopMerge" in spv_struct
    assert "OpSwitch" not in spv_struct


MASK_REUSE_HEAD = "; mattr: +wavefrontsize64,-wavefrontsize32\nv_mbcnt_lo_u32_b32 v1, -1, 0\nv_mbcnt_hi_u32_b32 v1, -1, v1\n"


def emulated_exchanges(make_shader, monkeypatch, body, name="cs_reuse"):
    """Translate `body` (wave64 compute) with --wave64-emulate; return (SPIR-V, number of mask-word ORs). Every
    whole-mask exchange ORs two words (lanes 0-31 and 32-63), so the count is two per exchange."""
    shader = make_shader(MASK_REUSE_HEAD + body + "s_endpgm\n", name)
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", True)
    asm = shader.translate()
    shader.validate(asm)
    return asm, asm.count("OpAtomicOr")


def test_wave64_emulate_reuses_a_mask_exchange(make_shader, monkeypatch):
    """The same register value (or EXEC) read as a whole mask twice in a block is exchanged once; a rewrite of the
    register or of EXEC gets a new exchange."""
    cmp = "v_cmp_eq_u32_e64 s[4:5], 5, v1\n"
    _, same = emulated_exchanges(make_shader, monkeypatch, cmp + "s_bcnt1_i32_b64 s6, s[4:5]\ns_bcnt1_i32_b64 s7, s[4:5]\n")
    _, rewritten = emulated_exchanges(
        make_shader, monkeypatch, cmp + "s_bcnt1_i32_b64 s6, s[4:5]\nv_cmp_gt_u32_e64 s[4:5], 40, v1\ns_bcnt1_i32_b64 s7, s[4:5]\n")
    _, same_exec = emulated_exchanges(make_shader, monkeypatch, "s_bcnt1_i32_b64 s6, exec\ns_bcnt1_i32_b64 s7, exec\n")
    _, new_exec = emulated_exchanges(
        make_shader, monkeypatch, "s_bcnt1_i32_b64 s6, exec\nv_cmpx_gt_u32_e32 10, v1\ns_bcnt1_i32_b64 s7, exec\n")
    _, pair = emulated_exchanges(  # s4 and s5 hold the same value: one exchange serves v_mbcnt_lo(s4) and _hi(s5)
        make_shader, monkeypatch, cmp + "v_mbcnt_lo_u32_b32 v3, s4, 0\nv_mbcnt_hi_u32_b32 v3, s5, v3\n")
    _, half = emulated_exchanges(  # s5 rewritten, s4 kept: two exchanges
        make_shader, monkeypatch,
        cmp + "v_mbcnt_lo_u32_b32 v3, s4, 0\nv_cmp_gt_u32_e64 s[6:7], 40, v1\ns_and_b32 s5, s5, s7\n"
        "v_mbcnt_hi_u32_b32 v3, s5, v3\n")
    assert (same, rewritten, same_exec, new_exec, pair, half) == (2, 4, 2, 4, 2, 4)


def test_mesh_s3_wave_id_emits_local_index_shift_and_bfe(make_shader):
    """Mesh shader prologue initializes s3[27:24] from LocalInvocationIndex >> 6 and s_bfe_u32 extracts it."""
    path = os.path.join(ROOT, "runtime", "test", "shaders", "ms_wave_info_s3.asm")
    with open(path, encoding="utf-8") as f:
        shader = make_shader(f.read(), "ms_wave_info_s3", dxbc=SAMPLE_DXBC["ms_wave_info_s3"])
    asm = shader.translate()
    shader.validate(asm)
    assert "%wave = OpShiftRightLogical %uint %lidx %c6" in asm
    assert "%mwi = OpShiftLeftLogical %uint %wave %c18" in asm
    assert "OpStore %s3p %mwi" in asm
    assert "OpBitFieldUExtract %uint" in asm


PK_F16_REPRO_513 = """\
; mattr: +wavefrontsize64,-wavefrontsize32
v_mov_b32_e32 v27, 0xbdd53e00
v_mov_b32_e32 v32, 0x3d3d32a5
v_and_b32_e32 v7, 7, v0
v_lshlrev_b32_e32 v7, 7, v7
v_or_b32_e32 v20, 0x3c00, v7
v_pk_add_f16 v31, v32, v20 op_sel:[0,0] op_sel_hi:[0,0]
v_pk_mul_f16 v34, v27, v31 op_sel:[0,0] op_sel_hi:[0,0]
v_lshlrev_b32_e32 v1, 2, v0
buffer_store_dword v34, v1, s[0:3], 0 offen
v_add_nc_u32_e32 v2, 256, v1
buffer_store_dword v31, v2, s[0:3], 0 offen
s_endpgm
"""


@pytest.mark.parametrize("emulate", [False, True], ids=["wave64", "wave64-emulate"])
def test_pk_f16_int_pack_repro_513(emulate, make_shader, monkeypatch):
    """SHADER-70 (#513): v_pk_*_f16 packs with integer IEEE-754 RTE instead of PackHalf2x16 so drivers cannot fold
    UnpackHalf2x16(PackHalf2x16(x)) back to unrounded f32, and produces exact RTE bits on the #513 repro."""
    import sys

    semantic_dir = os.path.join(ROOT, "tests", "semantic")
    if semantic_dir not in sys.path:
        sys.path.insert(0, semantic_dir)
    import spirv_mini_interp

    shader = make_shader(PK_F16_REPRO_513, name="cs_pk_f16_repro_513")
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", emulate)
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv

    sg_size = 32 if emulate else 64
    mem = spirv_mini_interp.run_workgroup(
        spv, 64, sg_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % sg_size},
        push_constants=[0, 0, 0xFFFFFFFF, 0] + [0] * 28,
        memory={},
    )
    assert [mem[256 + 4 * i] for i in range(3)] == [0x3CD53CD5, 0x3D553D55, 0x3DD53DD5]
    assert [mem[4 * i] for i in range(3)] == [0x3F403F40, 0x40004000, 0x40604060]


@pytest.mark.parametrize("op_asm", [
    "v_pk_add_f16 v3, v1, v2",
    "v_pk_mul_f16 v3, v1, v2",
    "v_pk_max_f16 v3, v1, v2",
    "v_pk_min_f16 v3, v1, v2",
    "v_pk_fma_f16 v3, v1, v2, v4",
    "v_add_f16_e64 v3, v1, v2",
])
def test_pk_and_scalar_f16_ops_emit_no_packhalf2x16(op_asm, make_shader):
    """All v_pk_*_f16 instructions and scalar f16_out ops validate and avoid GLSL PackHalf2x16."""
    body = (
        "v_mov_b32 v1, 0x3c003800\n"
        "v_mov_b32 v2, 0x40003c00\n"
        "v_mov_b32 v3, 0x00000000\n"
        "v_mov_b32 v4, 0x38003400\n"
        f"{op_asm}\n"
        "v_mov_b32 v0, v3"
    )
    shader = make_shader(program(body))
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv


@pytest.mark.parametrize("body", [
    "buffer_load_format_d16_xyzw v[2:3], v1, s[0:3], 0 offen\ns_waitcnt vmcnt(0)\nexp mrt0 v2, v3, v2, v3 done vm",
    "v_mov_b32 v2, 1.0\nv_mov_b32 v3, 0.5\nv_mov_b32 v4, 0.25\nv_mov_b32 v5, 2.0\n"
    "buffer_store_format_xyzw v[2:5], v1, s[0:3], 0 offen\nexp mrt0 v2, v3, v4, v5 done vm",
], ids=["d16-load", "store"])
def test_buffer_format_f16_narrowing_avoids_packhalf2x16(make_shader, body):
    """The f16 of a d16 typed buffer load, and of an f16 format a typed buffer store writes, narrow with integer RTE
    like the d16 image results (#489): PackHalf2x16 rounds toward zero on AMD drivers."""
    shader = make_shader("s_mov_b32 s0, 0x1000\ns_mov_b32 s1, 0\ns_mov_b32 s2, 0x100\ns_mov_b32 s3, 0x1fac\n"
                         "v_mov_b32 v1, 0\n" + body + "\ns_endpgm\n", "ps_buf_f16")
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv


def test_image_sample_and_gather_d16_avoid_packhalf2x16(make_shader):
    """d16 image_sample and image_gather4 pack f32->f16 via integer ops (no PackHalf2x16) and pass spirv-val."""
    sample_asm = (
        "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
        "s_mov_b32 s16, 0x92\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        "image_sample v[4:5], v[0:1], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D d16\n"
        "s_waitcnt vmcnt(0)\nexp mrt0 v4, v5, v4, v5 done vm\ns_endpgm\n"
    )
    sh1 = make_shader(sample_asm, name="ps_d16_sample_ci")
    asm1 = sh1.translate()
    sh1.validate(asm1)
    assert "PackHalf2x16" not in asm1
    assert "OpBitwiseOr %uint" in asm1 and "OpSelect %uint" in asm1

    gather_asm = (
        "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
        "s_mov_b32 s16, 0x92\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        "image_gather4_lz v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16\n"
        "s_waitcnt vmcnt(0)\nexp mrt0 v4, v5, v4, v5 done vm\ns_endpgm\n"
    )
    sh2 = make_shader(gather_asm, name="ps_d16_gather_ci")
    asm2 = sh2.translate()
    sh2.validate(asm2)
    assert "PackHalf2x16" not in asm2
    assert "OpBitwiseOr %uint" in asm2 and "OpSelect %uint" in asm2


def _load_mini_interp():
    import sys
    semantic_dir = os.path.join(ROOT, "tests", "semantic")
    if semantic_dir not in sys.path:
        sys.path.insert(0, semantic_dir)
    import spirv_mini_interp
    return spirv_mini_interp


def test_fma_mix_decodes_fimm_as_binary16_and_preserves_other_half(make_shader):
    """v_fma_mix{lo_f16,hi_f16,_f32} with op_sel_hi=1 decode fimm literals (e.g. 1.5, -2.0) as
    f16 instead of f32 bit patterns, preserve the unwritten dst half, and emit no PackHalf2x16."""
    si = _load_mini_interp()
    ps = (
        "; fma_mix_fimm\n"
        "v_mov_b32 v1, 0x40000000\n"  # f32 2.0
        "v_mov_b32 v2, 0x3e800000\n"  # f32 0.25
        "v_mov_b32 v3, 0xbeef0000\n"  # sentinel high half
        # 1.5 (f16) * 2.0 (f32) + 0.25 (f32) = 3.25 -> f16 0x4280; high half 0xbeef preserved
        "v_fma_mixlo_f16 v3, 1.5, v1, v2 op_sel:[0,0,0] op_sel_hi:[1,0,0]\n"
        "v_mov_b32 v4, 0x0000cafe\n"  # sentinel low half
        "v_mov_b32 v5, 0x3f800000\n"  # f32 1.0
        # (-2.0 (f16)) * 1.5 (f16) + 1.0 (f32) = -2.0 -> f16 0xc000; low half 0xcafe preserved
        "v_fma_mixhi_f16 v4, -2.0, 1.5, v5 op_sel:[0,0,0] op_sel_hi:[1,1,0]\n"
        # (-2.0 (f16)) * 1.5 (f16) + 0.25 (f32) = -2.75 -> f32 0xc0300000
        "v_fma_mix_f32 v6, -2.0, 1.5, v2 op_sel:[0,0,0] op_sel_hi:[1,1,0]\n"
        "exp mrt0 v3, v4, v6, v3 done vm\n"
        "s_endpgm\n"
    )
    shader = make_shader(ps, name="ps_fma_mix_fimm")
    spv = shader.translate()
    shader.validate(spv)

    out = si.run_fragment(spv)["%out_mrt0"]
    assert out[0] == 0xBEEF4280
    assert out[1] == 0xC000CAFE
    assert out[2] == 0xC0300000
    assert "PackHalf2x16" not in spv


@pytest.mark.parametrize("op,sentinel,shift,f32_bits,expected_half", [
    # RTE tie-to-even: 1.0 + 2^-11 -> 0x3c00 (even), 1.0 + 3*2^-11 -> 0x3c02 (even)
    ("v_fma_mixlo_f16", 0xBEEF0000, 0, 0x3F801000, 0x3C00),
    ("v_fma_mixlo_f16", 0xBEEF0000, 0, 0x3F803000, 0x3C02),
    ("v_fma_mixhi_f16", 0x0000CAFE, 16, 0x3F801000, 0x3C00),
    ("v_fma_mixhi_f16", 0x0000CAFE, 16, 0x3F803000, 0x3C02),
    # Subnormals & subnormal ties: 2^-15 -> 0x0200, 2^-24 -> 0x0001, 2^-25 -> 0x0000, 3*2^-25 -> 0x0002
    ("v_fma_mixlo_f16", 0x12340000, 0, 0x38000000, 0x0200),
    ("v_fma_mixlo_f16", 0x12340000, 0, 0x33800000, 0x0001),
    ("v_fma_mixlo_f16", 0x12340000, 0, 0x33000000, 0x0000),
    ("v_fma_mixhi_f16", 0x00005678, 16, 0x33C00000, 0x0002),
    # Overflow to +-Inf, -0.0, quiet NaN
    ("v_fma_mixlo_f16", 0xAAAA0000, 0, 0x477FF000, 0x7C00),
    ("v_fma_mixhi_f16", 0x0000BBBB, 16, 0xC77FF000, 0xFC00),
    ("v_fma_mixlo_f16", 0xCCCC0000, 0, 0x80000000, 0x8000),
    ("v_fma_mixhi_f16", 0x0000DDDD, 16, 0x7FC00000, 0x7E00),
])
def test_fma_mix_f16_rte_edge_cases_and_half_preservation(
    op, sentinel, shift, f32_bits, expected_half, make_shader
):
    """v_fma_mixlo_f16 / v_fma_mixhi_f16 round f32 results with exact IEEE-754 binary16 RTE
    (ties-to-even, subnormals, overflow, -0.0, quiet NaN) while preserving the other dst half."""
    si = _load_mini_interp()
    # Fma(1.0, x, -0.0) yields x (preserving -0.0 when x == -0.0)
    ps = (
        "; fma_mix_f16_rte\n"
        "v_mov_b32 v1, 0x3f800000\n"
        f"v_mov_b32 v2, {f32_bits:#010x}\n"
        "v_mov_b32 v3, 0x80000000\n"
        f"v_mov_b32 v4, {sentinel:#010x}\n"
        f"{op} v4, v1, v2, v3 op_sel:[0,0,0] op_sel_hi:[0,0,0]\n"
        "exp mrt0 v4, v4, v4, v4 done vm\n"
        "s_endpgm\n"
    )
    shader = make_shader(ps, name="ps_fma_mix_f16_rte")
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv

    out = si.run_fragment(spv)["%out_mrt0"]
    expected_word = sentinel | (expected_half << shift)
    assert out[0] == expected_word


@pytest.mark.parametrize("suffix,fmt_code,expected_word", [
    ("xy", 29, 0x3C023C00),  # format 29 = (4, "f16", (16, 16)): lo=0x3c00, hi=0x3c02
    ("x", 13, 0x00003C00),   # format 13 = (2, "f16", (16,)): lo=0x3c00
])
def test_encode_format_f16_uses_f16_bits_rte(suffix, fmt_code, expected_word, make_shader):
    """buffer_store_format_{x,xy} with f16 V# formats uses f16_bits RTE (no PackHalf2x16),
    validates with spirv-val, and stores exact tie-to-even f16 bits."""
    si = _load_mini_interp()
    base = 0x1000
    vdata = "v2" if suffix == "x" else "v[2:3]"
    ps = (
        "; buffer_store_format_f16_rte\n"
        f"s_mov_b32 s0, {base:#x}\n"
        "s_mov_b32 s1, 0\n"
        "s_mov_b32 s2, 0x100\n"
        f"s_mov_b32 s3, {fmt_code << 12:#x}\n"
        "v_mov_b32 v1, 0\n"
        "v_mov_b32 v2, 0x3f801000\n"  # 1.0 + 2^-11 -> 0x3c00 (tie to even)
        "v_mov_b32 v3, 0x3f803000\n"  # 1.0 + 3*2^-11 -> 0x3c02 (tie to even)
        f"buffer_store_format_{suffix} {vdata}, v1, s[0:3], 0 offen\n"
        "s_waitcnt vmcnt(0)\n"
        "exp mrt0 v2, v3, v2, v3 done vm\n"
        "s_endpgm\n"
    )
    shader = make_shader(ps, name=f"ps_encode_format_f16_{suffix}")
    spv = shader.translate()
    shader.validate(spv)
    assert "PackHalf2x16" not in spv

    mem = {base: 0}
    si.run_fragment(spv, memory=mem)
    assert mem[base] == expected_word


def test_remaining_d16_paths_avoid_packhalf2x16(make_shader):
    """buffer_format d16 and image_sample_c_lz d16 avoid PackHalf2x16 and validate with spirv-val."""
    tbuf_asm = (
        "s_mov_b32 s0, 0x1000\n"
        "s_mov_b32 s1, 0\n"
        "s_mov_b32 s2, 0x100\n"
        f"s_mov_b32 s3, {77 << 12:#x}\n"  # format 77 = f32x4
        "v_mov_b32 v0, 0\n"
        "buffer_load_format_d16_xy v2, v0, s[0:3], 0 offen\n"
        "s_waitcnt vmcnt(0)\n"
        "exp mrt0 v2, v2, v2, v2 done vm\n"
        "s_endpgm\n"
    )
    sh1 = make_shader(tbuf_asm, name="ps_tbuffer_d16_nopack")
    spv1 = sh1.translate()
    sh1.validate(spv1)
    assert "PackHalf2x16" not in spv1

    cmplz_asm = (
        "s_load_dwordx8 s[8:15], s[0:1], 0x0\ns_waitcnt lgkmcnt(0)\n"
        "s_mov_b32 s16, 0x92\ns_mov_b32 s17, 0\ns_mov_b32 s18, 0\ns_mov_b32 s19, 0\n"
        "image_sample_c_lz v4, v[0:2], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D d16\n"
        "s_waitcnt vmcnt(0)\n"
        "exp mrt0 v4, v4, v4, v4 done vm\n"
        "s_endpgm\n"
    )
    sh2 = make_shader(cmplz_asm, name="ps_sample_c_lz_d16_nopack")
    spv2 = sh2.translate()
    sh2.validate(spv2)
    assert "PackHalf2x16" not in spv2


WAVE64_SCMP_MASK_CASES = """\
; mattr: +wavefrontsize64,-wavefrontsize32
v_cmp_lt_u32_e64 vcc, v0, 17
s_and_b64 s[6:7], vcc, exec
s_cmp_eq_u32 vcc_lo, vcc_hi
s_cmp_lg_u32 exec_lo, exec_hi
s_cmp_eq_u32 s6, s7
s_cmp_eq_u32 vcc_lo, 0x80000003
s_cmp_lt_u32 exec_hi, s4
s_cmp_gt_i32 s6, 0
s_cmp_eq_u64 vcc, s[4:5]
s_cmp_lg_u64 exec, 0x12345678
s_endpgm
"""


@pytest.mark.parametrize("emulate", [False, True], ids=["wave64", "wave64-emulate"])
def test_wave64_scmp_mask_halves_and_scalars_validate(emulate, make_shader, monkeypatch):
    """Opposite-half s_cmp_*_u32, ordering compares on mask halves, and mask-vs-scalar/imm u32/u64
    compares translate and pass spirv-val in both native wave64 and --wave64-emulate modes."""
    shader = make_shader(WAVE64_SCMP_MASK_CASES, name="cs_w64_scmp_masks")
    monkeypatch.setattr(rdna2spirv, "WAVE64_EMULATE", emulate)
    spv = shader.translate()
    shader.validate(spv)
    if emulate:
        assert "OpGroupNonUniform" not in spv
    else:
        assert "OpGroupNonUniformBitwiseOr" in spv
