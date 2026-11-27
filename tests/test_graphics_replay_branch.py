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
])
def test_reject_invalid_graphics_replay_branch(make_shader, body, reason):
    fixture = Path(ROOT) / "runtime/test/shaders/register_replay/program.asm"
    asm = fixture.read_text(encoding="ascii").replace("v_xor_b32_e32 v4, s10, v4",
                                                      "s_cmp_ge_u32 s10, 30\n" + body)
    shader = make_shader(asm + "\nexp mrt0 v4, v5, v6, v7 done vm\ns_endpgm\n")
    with pytest.raises(rdna2spirv.Unsupported, match=reason):
        graphics_register_replay.generate(rdna2spirv.parse_file(shader.path), "ps", rdna2spirv.Unsupported)
