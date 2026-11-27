from pathlib import Path

import pytest

import graphics_register_replay
import rdna2spirv
from conftest import ROOT, requires_tools


@requires_tools
@pytest.mark.parametrize(("body", "reason"), [
    ("s_cbranch_scc0 1\nv_mov_b32_e32 v19, 7\nv_mov_b32_e32 v4, v19", "undefined on a branch"),
    ("s_cbranch_scc0 -1", "backward branch"),
    ("s_cbranch_scc0 300", "invalid forward branch target"),
    ("s_cbranch_scc0 3\ns_cmp_eq_u32 s0, 7\ns_cbranch_scc1 2\nv_mov_b32_e32 v19, 7\nv_mov_b32_e32 v4, v0",
     "conditional jump escapes enclosing arm"),
    ("s_cbranch_scc0 5\ns_cmp_eq_u32 s0, 7\ns_cbranch_scc1 2\nv_mov_b32_e32 v4, 7\ns_branch 2\n"
     "v_mov_b32_e32 v4, 1\nv_mov_b32_e32 v4, 2", "branch join escapes enclosing arm"),
    ("s_cbranch_scc0 3\ns_cmp_eq_u32 s0, 7\ns_cbranch_scc1 1\nv_mov_b32_e32 v19, 7\nv_mov_b32_e32 v4, v19",
     "undefined on a branch"),
])
def test_reject_invalid_graphics_replay_branch(make_shader, body, reason):
    fixture = Path(ROOT) / "runtime/test/shaders/register_replay/program.asm"
    asm = fixture.read_text(encoding="ascii").replace("v_xor_b32_e32 v4, s10, v4",
                                                      "s_cmp_ge_u32 s10, 30\n" + body)
    shader = make_shader(asm + "\nexp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n")
    with pytest.raises(rdna2spirv.Unsupported, match=reason):
        graphics_register_replay.generate(rdna2spirv.parse_file(shader.path), "ps", rdna2spirv.Unsupported)

@requires_tools
@pytest.mark.parametrize(("original", "replacement", "reason"), [
    ("v_add_nc_u32_e32 v5, 17, v5", "s_mov_b32 s3, 17", "expected vector ALU"),
    ("s_mov_b64 exec, s[16:17]", "s_mov_b64 exec, s[14:15]", "matching EXEC"),
    ("v_xor_b32_e32 v4, s10, v4", "v_xor_b32_e32 v4, s16, v4", "undefined on a branch"),
])
def test_reject_unsupported_exec_region(make_shader, original, replacement, reason):
    folder = Path(ROOT) / "runtime/test/shaders/register_replay"
    block = (folder / "divergent.asm").read_text(encoding="ascii").replace(original, replacement)
    asm = (folder / "program.asm").read_text(encoding="ascii").replace("v_xor_b32_e32 v4, s10, v4", block)
    shader = make_shader(asm + "\nexp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n")
    with pytest.raises(rdna2spirv.Unsupported, match=reason):
        graphics_register_replay.generate(rdna2spirv.parse_file(shader.path), "ps", rdna2spirv.Unsupported)

@requires_tools
@pytest.mark.parametrize(("original", "replacement", "reason"), [
    ("s_and_saveexec_b64 s[18:19]", "s_and_saveexec_b64 s[16:17]", "overwrites live saved EXEC"),
    ("v_cmp_ne_u32_e64 s[14:15], v19", "v_cmp_ne_u32_e64 s[16:17], v19", "expected SGPR mask pairs"),
    ("s_mov_b64 exec, s[18:19]", "s_mov_b64 exec, s[16:17]", "matching EXEC"),
    ("v_add_nc_u32_e32 v4, 5, v4", "v_add_nc_u32_e32 v4, s18, v4", "undefined on a branch"),
])
def test_reject_invalid_nested_exec(make_shader, original, replacement, reason):
    folder = Path(ROOT) / "runtime/test/shaders/register_replay"
    block = (folder / "nested_exec.asm").read_text(encoding="ascii").replace(original, replacement, 1)
    asm = (folder / "program.asm").read_text(encoding="ascii").replace("v_xor_b32_e32 v4, s10, v4", block)
    shader = make_shader(asm + "\nexp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n")
    with pytest.raises(rdna2spirv.Unsupported, match=reason):
        graphics_register_replay.generate(rdna2spirv.parse_file(shader.path), "ps", rdna2spirv.Unsupported)


@requires_tools
@pytest.mark.parametrize(("old", "new", "reason"), [
    ("s[20:21]", "s[16:17]", "unprotected mask"),
    ("s_bcnt1_i32_b64 s23, s[20:21]", "s_bcnt1_i32_b64 s23, s[14:15]", "matching popcount"),
    ("v_add_nc_u32_e32 v4, s23, v4",
     ("v_cmp_ne_u32_e64 s[20:21], v22, 0\ns_bcnt1_i32_b64 s24, s[20:21]\n") * 5, "seven ballot captures"),
])
def test_reject_invalid_masked_ballot(make_shader, old, new, reason):
    folder = Path(ROOT) / "runtime/test/shaders/register_replay"
    block = (folder / "ballot_then.asm").read_text(encoding="ascii").replace(old, new)
    asm = (folder / "program.asm").read_text(encoding="ascii").replace("v_xor_b32_e32 v4, s10, v4", block)
    shader = make_shader(asm + "\nexp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n")
    with pytest.raises(rdna2spirv.Unsupported, match=reason):
        graphics_register_replay.generate(rdna2spirv.parse_file(shader.path), "ps", rdna2spirv.Unsupported)
