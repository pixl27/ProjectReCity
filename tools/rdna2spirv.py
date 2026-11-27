"""RDNA (gfx10) shader -> SPIR-V assembly, version 0.

Model: one SPIR-V invocation runs one lane. VGPRs and SGPRs are Function-scope uint arrays; float ops bitcast,
so register contents keep exact bit patterns. EXEC is treated as all-ones (valid for branch-free code).

Supported in v0 (anything else is reported, never guessed):
  vector ALU  v_mov_b32 v_add/sub/subrev/mul/mul_legacy/fma/fmac/fmaak/fmamk/mac/max/min_f32 v_rcp/rsq/sqrt/
              exp/log/floor/fract/ceil/trunc_f32 v_cvt_f32_u32/i32/ubyte0..3 v_cvt_u32/i32_f32
              v_add/sub_nc_u32 v_and/or/xor_b32 v_lshlrev/lshrrev/ashrrev_i32 v_cndmask_b32 v_mul_u32_u24
              v_bfe_u32 v_med3_f32 v_mad_u32_u24
  scalar      s_mov_b32 s_add_u32 s_and_b32 s_or_b32 s_lshl_b32 s_lshr_b32 s_mul_i32
  no-ops      s_waitcnt* s_nop s_inst_prefetch s_endpgm s_setprio s_clause s_delay_alu
  export      exp pos0..3 / param0..31 (vertex) and mrt0..7 (pixel)
Vertex inputs: v0 = VertexIndex (the full-screen-triangle shaders read only that).

Usage (needs spirv-as/spirv-val on PATH, from the Vulkan SDK or spirv-tools):
    python3 -I rdna2spirv.py <disasm dir or .s> --out DIR [--limit N]
Writes <name>.spvasm for each translated shader and <name>.spv only for validated modules.
Prints coverage and validation results.
"""
import argparse
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from shader_ir import parse_file  # noqa: E402
import cfg_structure  # noqa: E402
import fssm_header  # noqa: E402
import lane_masks  # noqa: E402
import shader_meta as _shader_meta  # noqa: E402

STRUCTURED = True     # False (--dispatch-loop): every shader with branches as one loop around a switch
WAVE64_EMULATE = False   # --wave64-emulate: wave64 compute/mesh shaders exact on any subgroup size (Gen.x_round)

# GFX10 buffer format codes (Mesa src/amd/registers/gfx10-rsrc.json, enum GFX10_FORMAT).
import json as _json  # noqa: E402
GFX10_FORMATS = {int(k): v for k, v in _json.load(open(os.path.join(os.path.dirname(__file__), "gfx10_formats.json"))).items()}

# User SGPRs s0..s(USER_SGPRS-1) are initialised from push constants (128 bytes = Vulkan's guaranteed minimum).
USER_SGPRS = 32

# v_pipeflush (GFX10 VOP1) only flushes the VALU pipeline: no register, memory or EXEC effect to translate.
# s_icache_inv invalidates the instruction cache; s_ttracedata / s_ttracedata_imm send thread-trace data;
# s_code_end pads the end of a code object; s_version records the ISA version. None of them has a register, memory,
# SCC or EXEC effect. (Data-cache invalidations such as buffer_gl0_inv are about cross-wave coherence and are not
# here.)
NOOPS = {"s_sleep", "v_nop", "v_pipeflush", "s_nop", "s_inst_prefetch", "s_endpgm", "s_setprio", "s_clause",
         "s_delay_alu", "s_waitcnt", "s_waitcnt_vscnt", "s_waitcnt_vmcnt", "s_waitcnt_expcnt", "s_waitcnt_lgkmcnt",
         "s_waitcnt_depctr", "s_icache_inv", "s_ttracedata", "s_ttracedata_imm", "s_code_end", "s_version"}

F2 = {"v_add_f32": "OpFAdd", "v_sub_f32": "OpFSub", "v_mul_f32": "OpFMul",
      "v_max_f32": "max", "v_min_f32": "min"}
U2 = {"v_add_nc_u32": "OpIAdd", "v_sub_nc_u32": "OpISub", "v_and_b32": "OpBitwiseAnd", "v_or_b32": "OpBitwiseOr",
      "v_xor_b32": "OpBitwiseXor", "v_mul_u32_u24": "mul24"}
SHIFT_REV = {"v_lshlrev_b32": "OpShiftLeftLogical", "v_lshrrev_b32": "OpShiftRightLogical",
             "v_ashrrev_i32": "OpShiftRightArithmetic"}
F1 = {"v_rcp_f32": "rcp", "v_rsq_f32": "InverseSqrt", "v_sqrt_f32": "Sqrt", "v_exp_f32": "Exp2",
      "v_log_f32": "Log2", "v_floor_f32": "Floor", "v_fract_f32": "Fract", "v_ceil_f32": "Ceil",
      "v_trunc_f32": "Trunc"}
S2 = {"s_and_b32": "OpBitwiseAnd", "s_or_b32": "OpBitwiseOr",
      "s_lshl_b32": "OpShiftLeftLogical", "s_lshr_b32": "OpShiftRightLogical", "s_mul_i32": "OpIMul"}


def base_op(op):
    for suf in ("_e32", "_e64", "_sdwa", "_dpp"):
        if op.endswith(suf):
            return op[: -len(suf)]
    return op


class Unsupported(Exception):
    pass


class Gen:
    def __init__(self, stage):
        self.stage = stage
        self.body = []
        self.consts = {}
        self.n = 0
        self.outputs = {}
        self.inputs = {}
        self.interpolated = set()   # inputs read through v_interp_p1/p2; the rest are flat (v_interp_mov only)
        self.uses_psb = False
        self.uses_active = False
        self.uses_perm = False     # emulate: ds_permute_b32 parks its winning source lane in %x_perm
        self.uses_tex = False
        self.uses_subgroup = False
        self.uses_tex_array = False
        self.uses_storage = False
        self.uses_query = False
        self.uses_tex_1d = False
        self.uses_tex_3d = False
        self.uses_v4uint = False
        self.uses_store_views = set()  # image_store dims used (index into STORE_DIMS)
        self.uses_store_reads = False  # typed image_load through the uint/sint views
        self.writes_depth = False
        self.uses_smul = False
        self.uses_minlod = False
        self.uses_img_offset = False
        self.uses_pops = False
        self.uses_fhalf = False        # %fhalf_g (0.5) constant
        self.uses_f64 = False
        self.uses_bvh = False
        self.lconsts = {}              # 64-bit constants: value -> id
        self.uses_vote = False         # wave-uniform branches (OpGroupNonUniformAny)
        self.uses_ballot = False       # wave mask ballot (OpGroupNonUniformBallot)
        self.uses_wave_bits = False    # a whole lane mask gathered from the wave (OpGroupNonUniformBitwiseOr)
        self.uses_quad = False         # s_wqm on a lane mask (OpGroupNonUniformQuadSwap)
        self.lane_masks = {}           # id(instruction) -> registers holding a lane mask there (lane_masks.py)
        self.cur_masks = {}
        self.mask_writes = set()       # id(instruction) of the ones that write a lane mask to a VGPR
        self.write_all_lanes = False
        self.parked = {}               # (vgpr, lane) a mask is parked in -> the spare SGPR slot that keeps it per lane
        self.uses_lds = False
        self.lds_dwords = 16384        # CS: set from RSRC2 LDS_SIZE (power of two, covers every offset used)
        self.cs = None             # CS: user SGPR count, enabled workgroup ids, thread-id VGPR count, group size
        self.ms = None             # MS: {"maxv", "maxp", "threads"} from PSV0
        self.ms_params = set()     # MS: param export indices used
        self.ms_alloc = False      # MS: s_sendmsg GS_ALLOC_REQ seen
        self.pos_map = {}          # VS/MS: pos index (1..3) -> {comp: ("clip"|"cull", idx)} from OSG1
        self.pos_misc = set()      # VS: misc system values in pos1 (4=Layer, 5=ViewportIndex) from OSG1
        self.clip_count = 0        # ClipDistance array length (1..8)
        self.cull_count = 0        # CullDistance array length (1..8)
        self.uses_clip = False     # True once a pos1+ export writes a ClipDistance slot
        self.uses_cull = False     # True once a pos1+ export writes a CullDistance slot
        self.uses_layer = False    # VS: True once pos1 misc writes BuiltIn Layer
        self.uses_viewport = False # VS: True once pos1 misc writes BuiltIn ViewportIndex
        self.gs = None             # GS: {"in": [(reg, sv)], "out": [(reg, sv)], "nreg": n, "maxv": n}
        self.wave64 = False        # compiled for 64-lane waves (64-bit EXEC writes, exec_hi, v_mbcnt_hi, ...)
        self.emulate = False       # --wave64-emulate: exact wave64 on any subgroup size (cs/ms; see x_round)
        self.x_seg = 0             # emulate: number of the next continuation segment (past the block numbers)
        self.x_marks = []          # emulate: (index into body, segment) where a segment of the current block ends
        self.x_segs = {}           # emulate: continuation segment -> its SPIR-V lines
        self.x_spills = []         # emulate: (Function variable, pointer type) carrying a value across a round
        self.x_ptr_types = {}      # emulate: value type -> Function pointer type declared for spills
        self.x_dirty = set()       # emulate: (memory kind "lds"/"mem", "r"/"w") accessed since the last round
        self.x_tok = {}            # emulate: register -> the SSA id last stored to it in this block (its value's identity)
        self.x_gen = 0             # emulate: bumped at every label (control flow inside a block): forgets x_tok
        self.x_mcache = {}         # emulate: wave_mask key -> (lo, hi) already exchanged in this block
        self.x_lanes = 64          # emulate: workgroup invocations rounded up to whole waves
        self.uses_storage3d = False
        self.uses_heap = False
        self.uses_codebase = False
        self.uses_long = False
        self.vertex_inputs = []    # (D3D input register, component type 1=uint 2=sint 3=float) for fetch-calling VS
        self.vertex_signature = []   # full ISG1: (register, mask, system value, component type) for PSO lookup
        self.dynamic_format = False  # True when a buffer_load used a runtime V# FORMAT (unknown codes -> 0 at runtime)
        self.fixed_format_codes = set()  # static tbuffer/fixed format codes seen (validated against FORMATS)
        self.frag_inputs = {}      # PS: VGPR -> gl_FragCoord component, from the verified system-VGPR layout
        self.pixel_interpolants = None  # PS: (count, fp16_mask) from the FSSM pixel-input table; None = no table
        self.blocks = []      # per basic block: list of SPIR-V lines (dispatch-loop form), empty = straight-line
        self.succ = []        # per basic block: the blocks it can go to next (none = the program ends there)
        self.desc_fn = None   # lines of the descriptor-lookup function, once an instruction needs it
        self.sgpr_imm = {}    # SGPR -> literal an s_mov_b32 put there earlier in this basic block (see gather_addr_modes)
        self.desc_cache = {}  # (first SGPR, dwords) -> id of the %desc_lookup_fn result in the current basic block
        self._scan = (None, 0, [])   # (body list, lines already seen, open merge labels): see desc_cache_ok
        self.block_label = "%entry"   # label of the SPIR-V block the current instruction starts in
        self.build_meta = {}          # runtime build metadata from tools/shader_meta.py

    def tmp(self):
        self.n += 1
        return f"%t{self.n}"

    def cu(self, v):
        v &= 0xFFFFFFFF
        if v not in self.consts:
            self.consts[v] = f"%c{v:x}"
        return self.consts[v]

    def emit(self, s):
        self.body.append("         " + s)
        if self.emulate:
            if s.startswith(("OpStore %vcc ", "OpStore %active ")):
                reg, val = s.split()[1:3]
                self.x_tok[reg] = val
            elif s.endswith("= OpLabel"):
                self.x_gen += 1

    # ---- register access ----
    def load_reg(self, kind, r):
        arr = "%vgpr" if kind == "vgpr" else "%sgpr"
        p, t = self.tmp(), self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_Function_uint {arr} {self.cu(r)}")
        self.emit(f"{t} = OpLoad %uint {p}")
        return t

    def store_reg(self, kind, r, val):
        if kind == "sgpr":
            self.sgpr_imm.pop(r, None)
            self.x_tok[("sgpr", r)] = val
            if self.desc_cache:          # every write to the %sgpr array comes through here (see desc_lookup_sgpr)
                self.desc_cache = {k: v for k, v in self.desc_cache.items() if not (k[0] <= r < k[0] + k[1])}
        arr = "%vgpr" if kind == "vgpr" else "%sgpr"
        p = self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_Function_uint {arr} {self.cu(r)}")
        self.emit(f"OpStore {p} {val}")

    def src_u(self, a):
        k = a["k"]
        if k in ("vgpr", "sgpr"):
            if a.get("n", 1) != 1:
                raise Unsupported("64-bit operand")
            v = self.load_reg(k, a["r"])
        elif k == "special" and a["t"] == "null":
            v = self.cu(0)                # RDNA 'null' source reads as zero (e.g. soffset)
        elif k == "special" and a["t"] in ("exec_lo", "exec", "exec_hi"):
            v = self.exec_value()
        elif k == "special" and a["t"] == "m0":
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %m0")
        elif k == "special" and a["t"] == "vcc_hi":
            # wave64: vcc_hi is the upper half of VCC when VCC holds a lane mask (each lane's bit lives in %vcc).
            # wave32 (or plain data): vcc_hi is an ordinary SGPR.
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint {'%vcc' if (self.wave64 and 'vcc' in self.cur_masks) else '%vcc_hi'}")
        elif k == "special" and a["t"] == "scc":
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %scc")
        elif k == "special" and a["t"] == "src_pops_exiting_wave_id":
            # POPS (primitive-ordered pixel shading): the hardware wait loop compares this against the
            # collision word. Ordering comes from fragment-shader interlock around the whole shader instead
            # (see assemble), so the id reads as 0 and the runtime puts 0 (no collision) in the collision SGPR.
            v = self.cu(0)
            self.uses_pops = True
        elif k == "imm":
            v = self.cu(a["v"])
        elif k == "fimm":
            v = self.cu(struct.unpack("<I", struct.pack("<f", a["v"]))[0])
        elif k == "special" and a["t"] in ("vcc_lo", "vcc"):
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %vcc")
        else:
            raise Unsupported(f"operand {a.get('t', k)}")
        if "sdwa_sel" in a:
            v = self.sdwa_src(v, a)
        return self.modify(v, a)

    def modify(self, v, a):
        if a.get("abs") or a.get("neg"):
            f = self.tof(v)
            if a.get("abs"):
                g = self.tmp()
                self.emit(f"{g} = OpExtInst %float %glsl FAbs {f}")
                f = g
            if a.get("neg"):
                g = self.tmp()
                self.emit(f"{g} = OpFNegate %float {f}")
                f = g
            v = self.tou(f)
        return v

    def write_lo16(self, d, lo16, preserve_high=True):
        """Write a 16-bit result into the low half of VGPR d. On GFX10, f16 VOP1/VOP2 ops and opsel-capable VOP3
        ops are partial writes that keep the high half (Mesa ACO aco_ir.cpp: instr_is_16bit / can_use_opsel);
        non-opsel VOP2 16-bit integer ops define the whole register (high half modelled as zero)."""
        if not preserve_high or d["k"] != "vgpr":
            self.dst(d, lo16)
            return
        old, keep, r = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{old} = OpCopyObject %uint {self.load_reg('vgpr', d['r'])}")
        self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF0000)}")
        self.emit(f"{r} = OpBitwiseOr %uint {keep} {lo16}")
        self.dst(d, r)

    @staticmethod
    def opsel_bits(mods, nsrc):
        """Parse the VOP3 ``op_sel:[...]`` modifier of an op with ``nsrc`` sources -> (source selects, dst bit).
        A select bit of 1 reads the high half of that source; the last bit sends the result to the high half.
        No modifier means all zeros."""
        sel = next((m for m in mods if m.startswith("op_sel:")), None)
        if sel is None:
            return [0] * nsrc, 0
        bits = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")] + [0] * (nsrc + 1)
        return bits[:nsrc], bits[nsrc]

    def src_h(self, s):
        """Raw 32 bits of an operand of a 16-bit float op, before its neg/abs. An inline float constant (0.5, 1.0,
        -2.0, ...) is the f16 value in the low half over a zero high half, not f32 bits. Measured on an RX 6800
        (gfx1030, raw kernel, wave32 and wave64): v_add_f16 d, 1.0, v with v.lo = 0.25 gives 1.25,
        v_cvt_f32_f16 d, 0.5 gives 0x3f000000, v_pack_b32_f16 d, 1.0, v puts 0x3c00 in the low half, and op_sel's
        high half of the constant reads +0.0."""
        if s["k"] == "fimm":
            try:
                return self.cu(struct.unpack("<H", struct.pack("<e", s["v"]))[0])
            except OverflowError:
                raise Unsupported("f16 inline constant out of range") from None
        return self.src_u({**s, "neg": False, "abs": False})

    def pack_f16_pair_int(self, f0, f1):
        """Two f32 values -> one u32 (lo f16 | hi f16 << 16), each narrowed by f16_bits: integer ops only, no
        GLSL.std.450 at all (#489: an amdvlk64 pipeline-creation crash came from PackHalf2x16 on a d16
        image_sample result feeding a structured loop with OpGroupNonUniformAny). For d16 memory results (image
        samples and gathers, typed buffer loads): those are converted outside the VALU, so they stay round to
        nearest even and don't go through f16_pack2. f16_bits replaced #489's own f32_to_f16_int on 2026-11-22; the
        two gave the same bits on 30,648 inputs (every f32 exponent, ties, the subnormal and overflow edges, NaN)."""
        lo = self.f16_bits(f0)
        hi = self.f16_bits(f1)
        hi_s, out = self.tmp(), self.tmp()
        self.emit(f"{hi_s} = OpShiftLeftLogical %uint {hi} {self.cu(16)}")
        self.emit(f"{out} = OpBitwiseOr %uint {lo} {hi_s}")
        return out

    def f16_in(self, s, hi=0):
        """Read operand ``s`` as an f16 (low half, or high half when ``hi``) widened to f32, then apply its neg/abs."""
        u = self.src_h(s)
        v2, f = self.tmp(), self.tmp()
        self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
        self.emit(f"{f} = OpCompositeExtract %float {v2} {1 if hi else 0}")
        return self.modify_f(f, s)

    def fma_mix_srcs(self, a, mods):
        """Read the 3 operands of v_fma_mix*: f16 at op_sel[i] when op_sel_hi[i] is set (via src_h so fimm is binary16), else f32."""
        sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
        selhi = next((m for m in mods if m.startswith("op_sel_hi:")), "op_sel_hi:[0,0,0]")
        sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")]
        hb = [int(x) for x in selhi.split("[")[1].rstrip("]").split(",")]
        return [self.f16_in(src, sb[i]) if hb[i] else self.src_f(src) for i, src in enumerate(a[1:4])]

    def f16_bits(self, f, rtz=False):
        """f32 ``f`` -> f16 bits in the low 16 bits of a %uint, computed with integer ops only (SHADER-56).

        GLSL.std.450 PackHalf2x16 leaves the rounding mode to the implementation, and drivers fold a
        PackHalf2x16 -> UnpackHalf2x16 round trip to a no-op (#513: NVIDIA under --wave64-emulate; #512: RTX
        5080), so a following f16 op would see the unrounded f32. Integer ops can't be folded through.
        ``rtz=False``: round to nearest even, |x| >= 65520 -> inf (v_cvt_f16_f32 and every f16 ALU result).
        ``rtz=True``: round toward zero, finite values above the range -> +-65504 (v_cvt_pkrtz_f16_f32).
        f16 denormals are produced (RDNA2's default FP16 denorm mode keeps them); f32 denormal inputs give +-0.
        NaN -> quiet NaN with the top payload bits kept; +-inf -> +-inf."""
        t = self.tmp
        x, sgn, sg, a = t(), t(), t(), t()
        self.emit(f"{x} = OpBitcast %uint {f}")
        self.emit(f"{sgn} = OpShiftRightLogical %uint {x} {self.cu(16)}")
        self.emit(f"{sg} = OpBitwiseAnd %uint {sgn} {self.cu(0x8000)}")
        self.emit(f"{a} = OpBitwiseAnd %uint {x} {self.cu(0x7FFFFFFF)}")
        # normal range: rebias the exponent (127 -> 15) and drop 13 mantissa bits
        m, nh = t(), t()
        self.emit(f"{m} = OpISub %uint {a} {self.cu(0x38000000)}")
        if rtz:
            self.emit(f"{nh} = OpShiftRightLogical %uint {m} {self.cu(13)}")
        else:
            lsb, bias, mb, = t(), t(), t()
            self.emit(f"{lsb} = OpBitFieldUExtract %uint {m} {self.cu(13)} {self.cu(1)}")
            self.emit(f"{bias} = OpIAdd %uint {lsb} {self.cu(0xFFF)}")
            self.emit(f"{mb} = OpIAdd %uint {m} {bias}")
            self.emit(f"{nh} = OpShiftRightLogical %uint {mb} {self.cu(13)}")
        # subnormal range (2^-25 <= |x| < 2^-14): shift the full mantissa right by 126 - e (14..25)
        e, man, sh, short, shc, sub = t(), t(), t(), t(), t(), t()
        self.emit(f"{e} = OpShiftRightLogical %uint {a} {self.cu(23)}")
        self.emit(f"{man} = OpBitwiseOr %uint {self.cu(0x800000)} {self.emit_and(a, 0x7FFFFF)}")
        self.emit(f"{sh} = OpISub %uint {self.cu(126)} {e}")
        # min(sh, 25) without GLSL.std.450, so the d16 memory results can use this too (see pack_f16_pair_int)
        self.emit(f"{short} = OpULessThan %bool {sh} {self.cu(25)}")
        self.emit(f"{shc} = OpSelect %uint {short} {sh} {self.cu(25)}")
        self.emit(f"{sub} = OpShiftRightLogical %uint {man} {shc}")
        if not rtz:
            one, msk, rem, half, odd, gt, eq, tie, up, upu, sub2 = (t() for _ in range(11))
            self.emit(f"{one} = OpShiftLeftLogical %uint {self.cu(1)} {shc}")
            self.emit(f"{msk} = OpISub %uint {one} {self.cu(1)}")
            self.emit(f"{rem} = OpBitwiseAnd %uint {man} {msk}")
            self.emit(f"{half} = OpShiftRightLogical %uint {one} {self.cu(1)}")
            self.emit(f"{odd} = OpBitwiseAnd %uint {sub} {self.cu(1)}")
            self.emit(f"{gt} = OpUGreaterThan %bool {rem} {half}")
            self.emit(f"{eq} = OpIEqual %bool {rem} {half}")
            self.emit(f"{tie} = OpLogicalAnd %bool {eq} {self.emit_ne0(odd)}")
            self.emit(f"{up} = OpLogicalOr %bool {gt} {tie}")
            self.emit(f"{upu} = OpSelect %uint {up} {self.cu(1)} {self.cu(0)}")
            self.emit(f"{sub2} = OpIAdd %uint {sub} {upu}")
            sub = sub2
        # pick the range
        isn, issub, tiny, big, isinf, isnan = t(), t(), t(), t(), t(), t()
        self.emit(f"{isn} = OpUGreaterThanEqual %bool {a} {self.cu(0x38800000)}")
        self.emit(f"{tiny} = OpULessThan %bool {e} {self.cu(101)}")
        r1, r2, r3, r4, r5, nanb, res = t(), t(), t(), t(), t(), t(), t()
        self.emit(f"{r1} = OpSelect %uint {isn} {nh} {sub}")
        self.emit(f"{r2} = OpSelect %uint {tiny} {self.cu(0)} {r1}")
        if rtz:
            self.emit(f"{big} = OpUGreaterThanEqual %bool {a} {self.cu(0x47800000)}")
            self.emit(f"{r3} = OpSelect %uint {big} {self.cu(0x7BFF)} {r2}")
        else:
            self.emit(f"{big} = OpUGreaterThanEqual %bool {a} {self.cu(0x477FF000)}")
            self.emit(f"{r3} = OpSelect %uint {big} {self.cu(0x7C00)} {r2}")
        self.emit(f"{isinf} = OpIEqual %bool {a} {self.cu(0x7F800000)}")
        self.emit(f"{r4} = OpSelect %uint {isinf} {self.cu(0x7C00)} {r3}")
        pay = t()
        self.emit(f"{pay} = OpShiftRightLogical %uint {a} {self.cu(13)}")
        self.emit(f"{nanb} = OpBitwiseOr %uint {self.emit_and(pay, 0x3FF)} {self.cu(0x7E00)}")
        self.emit(f"{isnan} = OpUGreaterThan %bool {a} {self.cu(0x7F800000)}")
        self.emit(f"{r5} = OpSelect %uint {isnan} {nanb} {r4}")
        self.emit(f"{res} = OpBitwiseOr %uint {r5} {sg}")
        return res

    def emit_and(self, v, k):
        r = self.tmp()
        self.emit(f"{r} = OpBitwiseAnd %uint {v} {self.cu(k)}")
        return r

    def emit_ne0(self, v):
        r = self.tmp()
        self.emit(f"{r} = OpINotEqual %bool {v} {self.cu(0)}")
        return r

    def fma_f16_odd(self, x, y, z):
        """fma(x, y, z) for f16 inputs (widened to f32), rounded to f32 with round-to-odd (SHADER-55).

        RDNA2's v_fma_f16 / v_fmac_f16 / v_pk_fma_f16 round once, to f16 (measured: 0 of 12,612 mismatches against a
        correctly rounded f16 FMA on an RX 6700, #374). GLSL Fma rounds to f32 first, so narrowing it to f16 is a
        double rounding that can be 1 ulp off. Here: x*y is exact in f32 (11-bit significands give 22 bits, and the
        f16 range squared stays inside the f32 normal range), TwoSum splits p + z into the rounded sum s and its
        exact error, and when the error is non-zero and s is even, s moves one f32 ulp towards the error. That makes
        s the round-to-odd f32 value of the exact result, and round-to-odd at 24 bits then rounds correctly to any
        format with at most 22 bits, so f16_bits gives the single-rounded f16. The FAdd/FMul carry NoContraction
        (no_contraction), so the driver can't fuse or reorder TwoSum."""
        t = self.tmp
        pr, s_, bb, sb, da, db, err = (t() for _ in range(7))
        self.emit(f"{pr} = OpFMul %float {x} {y}")
        self.emit(f"{s_} = OpFAdd %float {pr} {z}")
        self.emit(f"{bb} = OpFSub %float {s_} {pr}")
        self.emit(f"{sb} = OpFSub %float {s_} {bb}")
        self.emit(f"{da} = OpFSub %float {pr} {sb}")
        self.emit(f"{db} = OpFSub %float {z} {bb}")
        self.emit(f"{err} = OpFAdd %float {da} {db}")
        sbits, ebits, lsb, even, nz, fin, need, same, xs, up, dn, adj, res, r = (t() for _ in range(14))
        self.emit(f"{sbits} = OpBitcast %uint {s_}")
        self.emit(f"{ebits} = OpBitcast %uint {err}")
        self.emit(f"{lsb} = OpBitwiseAnd %uint {sbits} {self.cu(1)}")
        self.emit(f"{even} = OpIEqual %bool {lsb} {self.cu(0)}")
        self.emit(f"{nz} = OpFUnordNotEqual %bool {err} %f0")          # true for a NaN error too...
        self.emit(f"{fin} = OpFOrdEqual %bool {err} {err}")            # ...so require a non-NaN error
        self.emit(f"{need} = OpLogicalAnd %bool {even} {self.emit_land(nz, fin)}")
        self.emit(f"{xs} = OpBitwiseXor %uint {sbits} {ebits}")
        self.emit(f"{same} = OpULessThan %bool {xs} {self.cu(0x80000000)}")   # same sign: away from zero
        self.emit(f"{up} = OpIAdd %uint {sbits} {self.cu(1)}")
        self.emit(f"{dn} = OpISub %uint {sbits} {self.cu(1)}")
        self.emit(f"{adj} = OpSelect %uint {same} {up} {dn}")
        self.emit(f"{res} = OpSelect %uint {need} {adj} {sbits}")
        self.emit(f"{r} = OpBitcast %float {res}")
        return r

    def emit_land(self, a, b):
        r = self.tmp()
        self.emit(f"{r} = OpLogicalAnd %bool {a} {b}")
        return r

    def f16_pack2(self, lo, hi):
        """Two f32 values -> one %uint of two RTE f16 halves (lo | hi << 16), without PackHalf2x16."""
        a, b, sh, r = self.f16_bits(lo), self.f16_bits(hi), self.tmp(), self.tmp()
        self.emit(f"{sh} = OpShiftLeftLogical %uint {b} {self.cu(16)}")
        self.emit(f"{r} = OpBitwiseOr %uint {a} {sh}")
        return r

    def f16_out(self, d, r, hi=0):
        """Narrow f32 ``r`` to f16 (round to nearest even, f16_bits) and write it to the low half of ``d``, or to the high half when
        ``hi`` (op_sel destination bit). Either way the other half is kept (see write_lo16)."""
        lo = self.f16_bits(r)
        if not hi:
            self.write_lo16(d, lo)
            return
        if d["k"] != "vgpr":
            raise Unsupported("op_sel destination")
        old, keep, sh16, v = (self.tmp() for _ in range(4))
        self.emit(f"{old} = OpCopyObject %uint {self.load_reg('vgpr', d['r'])}")
        self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF)}")
        self.emit(f"{sh16} = OpShiftLeftLogical %uint {lo} {self.cu(16)}")
        self.emit(f"{v} = OpBitwiseOr %uint {keep} {sh16}")
        self.dst(d, v)

    def f16_roundtrip(self, f):
        """f32 -> f16 -> f32 (round to nearest; f16 denormals flushed to signed zero), modelling a 32-bit
        v_interp read of a 16-bit interpolant (FSSM pixel-input table flag bit 4 marks the position; see
        the PS input path in translate).

        OpQuantizeToF16 returns +0 or -0 when the magnitude is too small for a normal f16, and the spec
        pins no rounding mode (round-to-nearest is measured on hardware, not guaranteed). Do not reuse
        this helper where f16 denormals or a specific rounding mode matter.

        Single OpQuantizeToF16 instruction on purpose: a PackHalf2x16/UnpackHalf2x16 round-trip is
        observably folded to a no-op by at least one Vulkan driver (RTX 5080, validated), while a lone
        OpQuantizeToF16 is implemented as a real conversion."""
        r = self.tmp()
        self.emit(f"{r} = OpQuantizeToF16 %float {f}")
        return r

    def f16_rtz(self, f):
        """f32 -> f16 bits (low 16) with round-toward-zero, as v_cvt_pkrtz does. Finite values above the f16 range
        become +-65504, never inf; inf/NaN stay inf/NaN. Integer-only (f16_bits): the earlier PackHalf2x16 ->
        UnpackHalf2x16 -> compare version relied on the round trip not being folded, and on llvmpipe and NVIDIA
        it was (FUZZ-2: v_cvt_pkrtz_f16_f32 gave the nearest value, one ulp above RTZ)."""
        return self.f16_bits(f, rtz=True)

    def modify_f(self, f, a):
        return self.tof(self.modify(self.tou(f), a))

    def host_addr(self, xbox_va):
        """Xbox GPU virtual address (ulong id) -> host buffer device address: heap base (push constant
        dwords 32..33) + VA. The runtime maps the console's GPU address window into one host buffer."""
        lo, hi, l64, h64, sh, base, r = (self.tmp() for _ in range(7))
        self.emit(f"{lo} = OpLoad %uint %ud_base_lo_p")
        self.emit(f"{hi} = OpLoad %uint %ud_base_hi_p")
        self.emit(f"{l64} = OpUConvert %ulong {lo}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{r} = OpIAdd %ulong {base} {xbox_va}")
        self.uses_heap = True
        self.cu(1), self.cu(2)        # member indices of the push-constant block, declared before assembly
        return r

    def tof(self, u):
        f = self.tmp()
        self.emit(f"{f} = OpBitcast %float {u}")
        return f

    def tou(self, f):
        u = self.tmp()
        self.emit(f"{u} = OpBitcast %uint {f}")
        return u

    def src_f(self, a):
        return self.tof(self.src_u(a))

    # ---- EXEC (this lane's active bit) ----
    def active_b(self):
        b = self.tmp()
        self.emit(f"{b} = OpLoad %bool %active")
        return b

    def exec_value(self):
        """EXEC as a lane mask for this invocation: all-ones if active, else 0."""
        r = self.tmp()
        self.emit(f"{r} = OpSelect %uint {self.active_b()} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
        return r

    def set_exec(self, mask_u):
        self.emit(f"OpStore %active {self.tmp_bool_nz(mask_u)}")
        self.uses_active = True

    def dst(self, a, val):
        if a.get("sdwa_dst") and a["k"] == "vgpr":
            val = self.sdwa_dst(a, val)
        if a["k"] == "special" and a["t"] == "null":
            return                                    # result discarded
        if a["k"] == "special" and a["t"] in ("exec_lo", "exec"):
            self.set_exec(val)
            return
        if a["k"] == "vgpr" and a.get("n", 1) == 1:
            # VALU writes only land in active lanes. The exception is a lane mask on its way through a VGPR
            # (lane_masks.py): every lane works out its own bit of it, active or not.
            if self.write_all_lanes:
                self.store_reg("vgpr", a["r"], val)
                return
            p, old, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_Function_uint %vgpr {self.cu(a['r'])}")
            self.emit(f"{old} = OpLoad %uint {p}")
            self.emit(f"{r} = OpSelect %uint {self.active_b()} {val} {old}")
            self.emit(f"OpStore {p} {r}")
        elif a["k"] == "sgpr" and a.get("n", 1) == 1:
            self.store_reg(a["k"], a["r"], val)
        elif a["k"] == "special" and a["t"] in ("vcc_lo", "vcc"):
            self.emit(f"OpStore %vcc {val}")
        elif a["k"] == "special" and a["t"] == "m0":
            self.emit(f"OpStore %m0 {val}")
        elif a["k"] == "special" and a["t"] == "vcc_hi":
            self.emit(f"OpStore %vcc_hi {val}")
        elif a["k"] == "special" and a["t"] == "scc":
            self.emit(f"OpStore %scc {val}")
        else:
            raise Unsupported(f"destination {a.get('t', a['k'])}")

    def mov64(self, d, s):
        """64-bit scalar move, including EXEC save/restore (EXEC = this lane's active bit)."""
        if d["k"] == "special" and d["t"] == "exec":
            self.set_exec(self.mask_src(s))
            return
        if d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
            self.emit(f"OpStore %vcc {self.mask_src(s)}")
            return
        if d["k"] != "sgpr" or d.get("n") != 2:
            raise Unsupported("s_mov_b64 destination")
        if s["k"] == "special" and s["t"] == "exec":
            lo = hi = self.exec_value()
        elif s["k"] == "sgpr" and s.get("n") == 2:
            lo, hi = self.load_reg("sgpr", s["r"]), self.load_reg("sgpr", s["r"] + 1)
        elif s["k"] == "imm":
            v = s["v"] & 0xFFFFFFFFFFFFFFFF
            if s["v"] < 0:
                v = s["v"] & 0xFFFFFFFFFFFFFFFF
            lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
        elif s["k"] == "special" and s["t"] == "vcc":
            lo = hi = self.tmp()            # this lane's bit in both halves, as for EXEC
            self.emit(f"{lo} = OpLoad %uint %vcc")
        else:
            raise Unsupported("s_mov_b64 source")
        self.store_reg("sgpr", d["r"], lo)
        self.store_reg("sgpr", d["r"] + 1, hi)

    def compare(self, op, a, scalar, exec_only=False):
        """v_cmp_* / s_cmp_*: lane result into VCC/SGPR mask (vector), SCC (scalar), or EXEC (v_cmpx)."""
        parts = op.split("_")          # v_cmp_lt_f32 / s_cmp_eq_u32
        cond, ty = parts[2], parts[3]
        srcs = a[0:2] if scalar else a[-2:]
        dst = None if (scalar or exec_only) else a[0]
        if ty in ("f32", "f16"):
            if ty == "f32":
                x, y = self.src_f(srcs[0]), self.src_f(srcs[1])
            else:   # f16 compare: the low half of each source, compared as f32 (exact for f16 values)
                vals = []
                for s in srcs:
                    u = self.src_h(s)
                    v2, f = self.tmp(), self.tmp()
                    self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                    self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
                    vals.append(self.modify_f(f, s))
                x, y = vals
            ops = {"lt": "OpFOrdLessThan", "le": "OpFOrdLessThanEqual", "gt": "OpFOrdGreaterThan",
                   "ge": "OpFOrdGreaterThanEqual", "eq": "OpFOrdEqual", "neq": "OpFUnordNotEqual",
                   "lg": "OpFOrdNotEqual", "nlt": "OpFUnordGreaterThanEqual", "ngt": "OpFUnordLessThanEqual",
                   "nle": "OpFUnordGreaterThan", "nge": "OpFUnordLessThan"}
        elif ty in ("u16", "i16"):
            x, y = self.src_u(srcs[0]), self.src_u(srcs[1])
            xs, ys = self.tmp(), self.tmp()
            f = "OpBitFieldSExtract" if ty == "i16" else "OpBitFieldUExtract"
            self.emit(f"{xs} = {f} %uint {x} {self.cu(0)} {self.cu(16)}")
            self.emit(f"{ys} = {f} %uint {y} {self.cu(0)} {self.cu(16)}")
            x, y = xs, ys
            if ty == "i16":
                x, y = self.as_int(x), self.as_int(y)
            s = ty == "i16"
            ops = {"lt": "OpSLessThan" if s else "OpULessThan", "le": "OpSLessThanEqual" if s else "OpULessThanEqual",
                   "gt": "OpSGreaterThan" if s else "OpUGreaterThan",
                   "ge": "OpSGreaterThanEqual" if s else "OpUGreaterThanEqual",
                   "eq": "OpIEqual", "ne": "OpINotEqual", "lg": "OpINotEqual"}
        elif ty in ("u32", "i32"):
            use_wave_equal = False
            wave_equal_half = None
            m0, m1 = self.holds_lane_mask(srcs[0]), self.holds_lane_mask(srcs[1])
            if scalar and (m0 or m1):
                h0 = self.mask_half(srcs[0]) if m0 else None
                h1 = self.mask_half(srcs[1]) if m1 else None

                def _is_u32_zero_or_all_ones(src):
                    return src["k"] == "imm" and (src["v"] & 0xFFFFFFFF) in (0, 0xFFFFFFFF)

                can_vote = False
                if cond in ("eq", "ne", "lg"):
                    if m0 and m1:
                        if not self.wave64:
                            can_vote = h0 != "hi" and h1 != "hi"
                        elif h0 is not None and h0 == h1:
                            can_vote = True
                            wave_equal_half = h0
                    else:
                        other = srcs[1] if m0 else srcs[0]
                        mh = h0 if m0 else h1
                        if _is_u32_zero_or_all_ones(other):
                            if not self.wave64:
                                can_vote = mh != "hi"
                            elif mh is not None:
                                can_vote = True
                                wave_equal_half = mh

                if can_vote:
                    use_wave_equal = True
                    x, y = self.src_u(srcs[0]), self.src_u(srcs[1])
                else:
                    def _mask_word(src, is_mask):
                        if not is_mask:
                            return self.src_u(src)
                        lo, hi = self.wave_mask(src)
                        return hi if self.mask_half(src) == "hi" else lo
                    x, y = _mask_word(srcs[0], m0), _mask_word(srcs[1], m1)
            else:
                x, y = self.src_u(srcs[0]), self.src_u(srcs[1])
            if ty == "i32":
                x, y = self.as_int(x), self.as_int(y)
            s = ty == "i32"
            ops = {"lt": "OpSLessThan" if s else "OpULessThan", "le": "OpSLessThanEqual" if s else "OpULessThanEqual",
                   "gt": "OpSGreaterThan" if s else "OpUGreaterThan",
                   "ge": "OpSGreaterThanEqual" if s else "OpUGreaterThanEqual",
                   "eq": "OpIEqual", "ne": "OpINotEqual", "lg": "OpINotEqual"}
        else:
            raise Unsupported(op)
        is_float = ty in ("f32", "f16")
        b, r = self.tmp(), self.tmp()
        if cond in (("f", "tru", "o", "u") if is_float else ("f", "t")):
            # Constant compares: f is always false, t / tru always true. o (ordered) is true when neither source
            # is NaN, u (unordered) when either is.
            if cond in ("o", "u"):
                nx, ny, either = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{nx} = OpIsNan %bool {x}")
                self.emit(f"{ny} = OpIsNan %bool {y}")
                self.emit(f"{either} = OpLogicalOr %bool {nx} {ny}")
                if cond == "u":
                    b = either
                else:
                    self.emit(f"{b} = OpLogicalNot %bool {either}")
            else:
                self.emit(f"{b} = OpCopyObject %bool {'%false' if cond == 'f' else '%true'}")
        elif cond not in ops:
            raise Unsupported(op)
        else:
            self.emit(f"{b} = {ops[cond]} %bool {x} {y}")
        if scalar and ty in ("u32", "i32") and use_wave_equal:
            b = self.wave_equal(b, cond == "eq", half=wave_equal_half)
        # Lane masks are all-ones/zero per invocation (one lane), SCC is 1/0. A VOPC result bit is 0 for lanes
        # that are off in EXEC (as on hardware), so wave-wide reductions of the mask (s_cbranch_vccz) are exact.
        bm = b
        if not scalar and not exec_only:
            bm = self.tmp()
            self.emit(f"{bm} = OpLogicalAnd %bool {b} {self.active_b()}")
        self.emit(f"{r} = OpSelect %uint {bm} {self.cu(1 if scalar else 0xFFFFFFFF)} {self.cu(0)}")
        if scalar:
            self.emit(f"OpStore %scc {r}")
        elif dst is None:          # v_cmpx: EXEC &= cond
            a2 = self.tmp()
            self.emit(f"{a2} = OpLoad %bool %active")
            a3 = self.tmp()
            self.emit(f"{a3} = OpLogicalAnd %bool {a2} {b}")
            self.emit(f"OpStore %active {a3}")
            self.uses_active = True
        elif dst["k"] == "special" and dst["t"] in ("vcc", "vcc_lo"):
            self.emit(f"OpStore %vcc {r}")
        elif dst["k"] == "sgpr":
            self.mask_dst(dst, r)
        else:
            raise Unsupported("v_cmp destination")

    def compare_x(self, op, a):
        """v_cmpx_*: like v_cmp but the result ANDs into EXEC. e32 form has only the two sources; e64 also names
        a destination (vcc/sgpr) which gets the compare result too."""
        srcs = a[-2:]
        self.compare("v_cmp_" + op[len("v_cmpx_"):], [None] + list(srcs), scalar=False, exec_only=True)

    def bitcmp(self, op, a):
        """s_bitcmp0/1_b32/b64: test if bit (a[1]) of scalar a[0] is 0 or 1, result in SCC."""
        is_64 = op.endswith("b64")
        test_one = op.startswith("s_bitcmp1")
        if self.holds_lane_mask(a[0]):
            lo, hi = self.wave_mask(a[0])
            if not is_64 and self.wave64:
                half = self.mask_half(a[0])
                if half == "hi":
                    lo = hi
        elif is_64 and a[0]["k"] == "sgpr" and a[0].get("n") == 2:
            lo, hi = self.load_reg("sgpr", a[0]["r"]), self.load_reg("sgpr", a[0]["r"] + 1)
        elif is_64 and a[0]["k"] == "imm":
            v = a[0]["v"] & 0xFFFFFFFFFFFFFFFF
            lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
        elif is_64:
            lo, hi = self.mask_src(a[0]), self.cu(0)
        else:
            lo = self.src_u(a[0])

        sh = self.src_u(a[1])
        mask_bits = 63 if is_64 else 31
        m = self.tmp()
        self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(mask_bits)}")

        if is_64:
            is_hi = self.tmp()
            self.emit(f"{is_hi} = OpUGreaterThanEqual %bool {m} {self.cu(32)}")
            word = self.tmp()
            self.emit(f"{word} = OpSelect %uint {is_hi} {hi} {lo}")
            sh32 = self.tmp()
            self.emit(f"{sh32} = OpBitwiseAnd %uint {m} {self.cu(31)}")
        else:
            word = lo
            sh32 = m

        shifted, bit = self.tmp(), self.tmp()
        self.emit(f"{shifted} = OpShiftRightLogical %uint {word} {sh32}")
        self.emit(f"{bit} = OpBitwiseAnd %uint {shifted} {self.cu(1)}")
        cond = self.tmp()
        self.emit(f"{cond} = {'OpINotEqual' if test_one else 'OpIEqual'} %bool {bit} {self.cu(0)}")
        res = self.tmp()
        self.emit(f"{res} = OpSelect %uint {cond} {self.cu(1)} {self.cu(0)}")
        self.emit(f"OpStore %scc {res}")

    def swizzle(self, a, mods, ins):
        """ds_swizzle_b32 vD, vS offset:swizzle(...): a subgroup shuffle (one invocation per lane)."""
        line = " ".join(m for m in mods) + " " + " ".join(x.get("t", "") for x in a if x["k"] in ("special", "raw"))
        import re as _re
        text = line + " " + str(ins.get("raw", ""))
        bm = _re.search(r'swizzle\(BITMASK_PERM,\s*"([01pi]{5})"\)', text)
        m = _re.search(r"swizzle\((\w+)((?:,\s*\d+)*)\)", text)
        if not m and not bm:
            raise Unsupported("ds_swizzle pattern")
        kind = "BITMASK_PERM" if bm else m.group(1)
        args = [int(x) for x in m.group(2).split(",")[1:]] if m and m.group(2) else []
        self.uses_subgroup = True
        lane = self.tmp()
        self.sub_lane(lane)
        if kind == "BITMASK_PERM":
            # One character per lane-index bit, bit 4 first: p = keep, i = invert, 0 / 1 = force.
            # Source lane = ((lane & and_mask) | or_mask) ^ xor_mask, within each group of 32 lanes.
            and_m = or_m = xor_m = 0
            for k, ch in enumerate(bm.group(1)):
                bit = 1 << (4 - k)
                if ch in "pi":
                    and_m |= bit
                if ch == "i":
                    xor_m |= bit
                if ch == "1":
                    or_m |= bit
            t1, t2, src = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{t1} = OpBitwiseAnd %uint {lane} {self.cu(and_m | 0xFFFFFFE0)}")
            self.emit(f"{t2} = OpBitwiseOr %uint {t1} {self.cu(or_m)}")
            self.emit(f"{src} = OpBitwiseXor %uint {t2} {self.cu(xor_m)}")
        elif kind == "QUAD_PERM":
            sel, base, src = self.tmp(), self.tmp(), self.cu(0)
            self.emit(f"{sel} = OpBitwiseAnd %uint {lane} {self.cu(3)}")
            self.emit(f"{base} = OpBitwiseAnd %uint {lane} {self.cu(0xFFFFFFFC)}")
            pick = self.cu(args[0])
            for i in (1, 2, 3):
                e, nxt = self.tmp(), self.tmp()
                self.emit(f"{e} = OpIEqual %bool {sel} {self.cu(i)}")
                self.emit(f"{nxt} = OpSelect %uint {e} {self.cu(args[i])} {pick}")
                pick = nxt
            src = self.tmp()
            self.emit(f"{src} = OpBitwiseOr %uint {base} {pick}")
        elif kind == "SWAP":
            src = self.tmp()
            self.emit(f"{src} = OpBitwiseXor %uint {lane} {self.cu(args[0])}")
        elif kind == "REVERSE":
            src = self.tmp()
            self.emit(f"{src} = OpBitwiseXor %uint {lane} {self.cu(args[0] - 1)}")
        elif kind == "BROADCAST":
            src, keep = self.tmp(), self.tmp()
            self.emit(f"{keep} = OpBitwiseAnd %uint {lane} {self.cu(~(args[0] - 1) & 0xFFFFFFFF)}")
            self.emit(f"{src} = OpBitwiseOr %uint {keep} {self.cu(args[1])}")
        else:
            raise Unsupported("ds_swizzle " + kind)
        v, r = self.src_u(a[1]), self.tmp()
        self.g_shuffle(r, v, src)
        self.dst(a[0], r)

    def buffer_load(self, op, a, mods):
        """buffer_load_dword[xN] vD, vaddr, s[V#], soffset [offen|idxen] offset:N  -> buffer-device-address loads.
        addr = V#.base + soffset + inst_offset + (offen ? vaddr : 0) + (idxen ? vindex * V#.stride : 0)."""
        if any(m in mods for m in ("lds", "tfe")) or "format" in op:
            raise Unsupported(op + " form")
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        vd = a[0]
        addr, _, in_b = self.buffer_address(a, mods)
        cond = self.tmp()
        self.emit(f"{cond} = OpLogicalAnd %bool {self.active_b()} {in_b}")
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        pre_label = self.current_label_placeholder()
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {cond} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(addr)}")
        loaded = []
        for i in range(n):
            p, v = self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
            self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
            loaded.append(v)
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")
        phis = []
        for v_do in loaded:
            v = self.tmp()
            self.emit(f"{v} = OpPhi %uint {v_do} {lbl_do} {self.cu(0)} {pre_label}")
            phis.append(v)
        for i, v in enumerate(phis):
            self.dst({"k": "vgpr", "r": vd["r"] + i, "n": 1}, v)

    # GFX10 buffer formats (from Mesa src/amd/registers/gfx10-rsrc.json, GFX10_FORMAT):
    #   code: (bytes, kind, component bit widths from bit 0)
    # The names list the widths from the top bit down, so 2_10_10_10 (50) is R10G10B10A2.
    FORMATS = {22: (4, "f32", (32,)), 64: (8, "f32", (32, 32)), 74: (12, "f32", (32, 32, 32)),
               77: (16, "f32", (32, 32, 32, 32)), 20: (4, "u32", (32,)), 62: (8, "u32", (32, 32)),
               75: (16, "u32", (32, 32, 32, 32)), 13: (2, "f16", (16,)), 29: (4, "f16", (16, 16)),
               71: (8, "f16", (16, 16, 16, 16)), 1: (1, "unorm", (8,)), 14: (2, "unorm", (8, 8)),
               56: (4, "unorm", (8, 8, 8, 8)), 57: (4, "snorm", (8, 8, 8, 8)), 60: (4, "uint", (8, 8, 8, 8)),
               7: (2, "unorm", (16,)), 23: (4, "unorm", (16, 16)), 24: (4, "snorm", (16, 16)),
               65: (8, "unorm", (16, 16, 16, 16)), 66: (8, "snorm", (16, 16, 16, 16)),
               44: (4, "unorm", (2, 10, 10, 10)), 50: (4, "unorm", (10, 10, 10, 2)),
               69: (8, "uint", (16, 16, 16, 16)), 27: (4, "uint", (16, 16)),
               2: (1, "snorm", (8,)), 8: (2, "snorm", (16,)), 15: (2, "snorm", (8, 8))}

    def encode_format(self, fcomps, icomps, code):
        """Encode one element of format `code` -> list of uint ids (whole dwords; elements under 4 bytes return one
        value in the low bits). fcomps are f32 bit patterns (float/normalized kinds), icomps integers (u32/uint)."""
        try:
            size, kind, widths = self.FORMATS[code]
        except KeyError:
            raise Unsupported(f"format {code} (not in FORMATS)")
        if kind == "f32":
            return list(fcomps[:len(widths)])
        if kind == "u32":
            return list(icomps[:len(widths)])
        words = [self.cu(0)] * max(1, (size + 3) // 4)
        bit = 0
        for c, w in enumerate(widths):
            if kind == "f16":
                field = self.f16_bits(self.tof(fcomps[c]))
            elif kind == "unorm":
                cl, sc, rd, u = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{cl} = OpExtInst %float %glsl FClamp {self.tof(fcomps[c])} %f0 %f1")
                self.emit(f"{sc} = OpFMul %float {cl} {self.fconst(float((1 << w) - 1))}")
                self.emit(f"{rd} = OpExtInst %float %glsl RoundEven {sc}")
                self.emit(f"{u} = OpConvertFToU %uint {rd}")
                field = u
            elif kind == "snorm":
                cl, sc, rd, i, u = self.tmp(), self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{cl} = OpExtInst %float %glsl FClamp {self.tof(fcomps[c])} {self.fconst(-1.0)} %f1")
                self.emit(f"{sc} = OpFMul %float {cl} {self.fconst(float((1 << (w - 1)) - 1))}")
                self.emit(f"{rd} = OpExtInst %float %glsl RoundEven {sc}")
                self.emit(f"{i} = OpConvertFToS %int {rd}")
                self.emit(f"{u} = OpBitcast %uint {i}")
                field = u
            else:   # uint
                field = icomps[c]
            wi = bit // 32
            ins = self.tmp()
            self.emit(f"{ins} = OpBitFieldInsert %uint {words[wi]} {field} {self.cu(bit % 32)} {self.cu(w)}")
            words[wi] = ins
            bit += w
        return words

    def norm_bits(self, m, n):
        """f32 bits of m / (2^n - 1), correctly rounded, for an n-bit unsigned id m (1 <= n <= 16), integer ops only.

        In binary, m / (2^n - 1) is m's n bits repeated forever. T = m * sum(2^(64 - n(k+1))) holds the first 64 of
        them (the partial block at the bottom is dropped, below every bit used here). For 0 < m < 2^n - 1 the top 1 of
        T is in its high dword (T >= 2^(64-n)), so p = FindUMsb(T >> 32) is 16..31 and the value is in [2^(p-32),
        2^(p-31)). The mantissa is T >> (p + 9) (24 bits with the hidden one) and the round bit the one below it. The
        repeating tail is never all zero, so a tie can't happen: round to nearest is "add the round bit". A carry
        out of the mantissa moves into the exponent through the plain add. 0 and 2^n - 1 are exact (0 and 1.0)."""
        rep = sum(1 << (64 - n * (k + 1)) for k in range(64 // n))
        m64, t, h64, hi, pi, p, s, s64, q64, q, s1, s164, r64, rb, e, eb, b0, b1, b2, z, one, b3, b4 = \
            (self.tmp() for _ in range(23))
        self.emit(f"{m64} = OpUConvert %ulong {m}")
        self.emit(f"{t} = OpIMul %ulong {m64} {self.cl(rep)}")
        self.emit(f"{h64} = OpShiftRightLogical %ulong {t} {self.cl(32)}")
        self.emit(f"{hi} = OpUConvert %uint {h64}")
        self.emit(f"{pi} = OpExtInst %int %glsl FindUMsb {hi}")
        self.emit(f"{p} = OpBitcast %uint {pi}")
        self.emit(f"{s} = OpIAdd %uint {p} {self.cu(9)}")               # m = 0 gives p = -1: s = 8, result unused
        self.emit(f"{s64} = OpUConvert %ulong {s}")
        self.emit(f"{q64} = OpShiftRightLogical %ulong {t} {s64}")
        self.emit(f"{q} = OpUConvert %uint {q64}")
        self.emit(f"{s1} = OpISub %uint {s} {self.cu(1)}")
        self.emit(f"{s164} = OpUConvert %ulong {s1}")
        self.emit(f"{r64} = OpShiftRightLogical %ulong {t} {s164}")
        self.emit(f"{rb} = OpUConvert %uint {r64}")
        self.emit(f"{e} = OpIAdd %uint {p} {self.cu(95)}")              # biased exponent p - 32 + 127
        self.emit(f"{eb} = OpShiftLeftLogical %uint {e} {self.cu(23)}")
        self.emit(f"{b0} = OpISub %uint {q} {self.cu(0x800000)}")       # drop the hidden one
        self.emit(f"{b1} = OpIAdd %uint {eb} {b0}")
        self.emit(f"{b2} = OpBitwiseAnd %uint {rb} {self.cu(1)}")
        self.emit(f"{b3} = OpIAdd %uint {b1} {b2}")
        self.emit(f"{z} = OpIEqual %bool {m} {self.cu(0)}")
        self.emit(f"{one} = OpIEqual %bool {m} {self.cu((1 << n) - 1)}")
        self.emit(f"{b4} = OpSelect %uint {one} {self.cu(0x3F800000)} {b3}")
        res = self.tmp()
        self.emit(f"{res} = OpSelect %uint {z} {self.cu(0)} {b4}")
        return res

    def decode_format(self, words, code):
        """Decode one element of format `code` from raw dwords -> 4 uint ids holding f32 bits (or uint for *_UINT/uint).

        Missing channels repeat the format's own (component c = c mod n): measured on RX 6700 for SHADER-86.
        UNORM/SNORM give the correctly rounded quotient, as the RX 6700 does, built with integer ops (norm_bits):
        OpFDiv is allowed 2.5 ULP and the AMD and NVIDIA drivers return the reciprocal-multiply result for it (#599)."""
        try:
            size, kind, widths = self.FORMATS[code]
        except KeyError:
            raise Unsupported(f"format {code} (not in FORMATS)")
        if kind in ("f32", "u32"):
            n = len(widths)
            vals = list(words[:n])
            while len(vals) < 4:
                vals.append(vals[len(vals) % n])
            return vals
        out, bit = [], 0
        for w in widths:
            src = words[bit // 32]
            f = self.tmp()
            self.emit(f"{f} = OpBitFieldUExtract %uint {src} {self.cu(bit % 32)} {self.cu(w)}")
            if kind == "f16":
                v2, x = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {f}")
                self.emit(f"{x} = OpCompositeExtract %float {v2} 0")
                out.append(self.tou(x))
            elif kind == "unorm":
                out.append(self.norm_bits(f, w))
            elif kind == "snorm":
                # magnitude over 2^(w-1)-1 with the same integer rounding; the most negative code is -1.0
                si, neg, ab, au, sg, sel, lo, res = (self.tmp() for _ in range(8))
                self.emit(f"{si} = OpBitFieldSExtract %int {self.as_int(src)} {self.cu(bit % 32)} {self.cu(w)}")
                self.emit(f"{neg} = OpSLessThan %bool {si} {self.as_int(self.cu(0))}")
                self.emit(f"{ab} = OpExtInst %int %glsl SAbs {si}")
                self.emit(f"{au} = OpBitcast %uint {ab}")
                mag = self.norm_bits(au, w - 1)
                self.emit(f"{sg} = OpBitwiseOr %uint {mag} {self.cu(0x80000000)}")
                self.emit(f"{sel} = OpSelect %uint {neg} {sg} {mag}")
                self.emit(f"{lo} = OpULessThan %bool {au} {self.cu(1 << (w - 1))}")
                self.emit(f"{res} = OpSelect %uint {lo} {sel} {self.cu(0xBF800000)}")
                out.append(res)
            else:   # uint: raw integer
                out.append(f)
            bit += w
        n = len(out)
        while len(out) < 4:
            out.append(out[len(out) % n])
        return out

    BUF_ATOMICS = {"add": ("OpAtomicIAdd", 0), "sub": ("OpAtomicISub", 0), "umin": ("OpAtomicUMin", 0xFFFFFFFF),
                   "umax": ("OpAtomicUMax", 0), "smin": ("OpAtomicSMin", 0x7FFFFFFF), "smax": ("OpAtomicSMax", 0x80000000),
                   "and": ("OpAtomicAnd", 0xFFFFFFFF), "or": ("OpAtomicOr", 0), "xor": ("OpAtomicXor", 0),
                   "swap": ("OpAtomicExchange", None), "cmpswap": ("OpAtomicCompareExchange", None)}

    def buffer_atomic(self, op, a, mods):
        """buffer_atomic_<op> vdata, vaddr, s[V#], soffset ...: 32-bit atomic at the element address (buffer device
        address). With 'glc' the pre-op value is returned in vdata. Inactive lanes don't touch memory."""
        kind = op[len("buffer_atomic_"):]
        if kind not in self.BUF_ATOMICS or "x2" in kind:
            raise Unsupported(op)
        spv, neutral = self.BUF_ATOMICS[kind]
        addr, _, in_b = self.buffer_address(a, mods)
        vd = a[0]
        val = self.load_reg("vgpr", vd["r"])
        cond = self.tmp()
        self.emit(f"{cond} = OpLogicalAnd %bool {self.active_b()} {in_b}")
        # Guard all atomics with a structured if so inactive or OOB lanes never touch memory.
        lbl_do, lbl_end, res_do, old = self.tmp() + "_do", self.tmp() + "_end", self.tmp(), self.tmp()
        pre_label = self.current_label_placeholder()
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {cond} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        ptr, p = self.tmp(), self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(addr)}")
        self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(0)}")
        if kind == "cmpswap":
            cmp = self.load_reg("vgpr", vd["r"] + 1)
            self.emit(f"{res_do} = OpAtomicCompareExchange %uint {p} {self.cu(1)} {self.cu(0)} {self.cu(0)} {val} {cmp}")
        elif kind == "swap":
            self.emit(f"{res_do} = OpAtomicExchange %uint {p} {self.cu(1)} {self.cu(0)} {val}")
        else:
            self.emit(f"{res_do} = {spv} %uint {p} {self.cu(1)} {self.cu(0)} {val}")
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")
        self.emit(f"{old} = OpPhi %uint {res_do} {lbl_do} {self.cu(0)} {pre_label}")
        if "glc" in mods:
            self.dst({"k": "vgpr", "r": vd["r"], "n": 1}, old)

    def current_label_placeholder(self):
        """Label of the block currently being emitted (needed for OpPhi). Blocks are opened by OpLabel lines;
        under --wave64-emulate only those of the current segment count (an earlier one runs in another round)."""
        for line in reversed(self.body[self.x_marks[-1][0] if self.x_marks else 0:]):
            s = line.strip()
            if s.endswith("= OpLabel"):
                return s.split(" = ")[0]
        return self.block_label

    def image_get_lod(self, a, mods):
        """image_get_lod vD, vaddr, T#, S#: (clamped lod, unclamped lod) like OpImageQueryLod (2D, 2D array). For
        arrays the query takes only the 2D coordinate; the layer doesn't affect the LOD."""
        dim = next((m for m in mods if m.startswith("dim:")), "dim:SQ_RSRC_IMG_2D")
        if dim not in ("dim:SQ_RSRC_IMG_2D", "dim:SQ_RSRC_IMG_2D_ARRAY") or self.stage != "ps" or "a16" in mods:
            raise Unsupported("image_get_lod form")
        arr = dim.endswith("_ARRAY")
        if arr:
            self.uses_tex_array = True
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x3")
        dmask = int(dm.split(":")[1], 0)
        vdata, addr = a[0], self.addr_regs(a[1])
        img_idx = self.desc_lookup_sgpr(a[2], 8)
        smp_idx = self.desc_lookup_sgpr(a[3], 4)
        ip, img, sp, smp, si = (self.tmp() for _ in range(5))
        self.emit(f"{ip} = OpAccessChain %{'_ptr_UC_imga' if arr else '_ptr_UC_img'} "
                  f"{'%images_arr' if arr else '%images'} {img_idx}")
        self.emit(f"{img} = OpLoad {'%img2da' if arr else '%img2d'} {ip}")
        self.emit(f"{sp} = OpAccessChain %_ptr_UC_smp %samplers {smp_idx}")
        self.emit(f"{smp} = OpLoad %smp {sp}")
        self.emit(f"{si} = OpSampledImage {'%simg2da' if arr else '%simg2d'} {img} {smp}")
        f = [self.tof(self.load_reg("vgpr", r)) for r in addr[:2]]
        uv, lod = self.tmp(), self.tmp()
        self.emit(f"{uv} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
        self.emit(f"{lod} = OpImageQueryLod %v2float {si} {uv}")
        self.uses_query = True
        out = 0
        for c in range(2):
            if dmask >> c & 1:
                comp = self.tmp()
                self.emit(f"{comp} = OpCompositeExtract %float {lod} {c}")
                self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, self.tou(comp))
                out += 1

    def buffer_load_format(self, op, a, mods, fixed_fmt=None):
        """buffer_load_format_[x|xy|xyz|xyzw]: element at V#.base + offsets, decoded per the V# FORMAT field
        (word3 bits 12-18), then swizzled by DST_SEL_X/Y/Z/W (word3 bits 0-11: 0=0, 1=1, 4..7 = X..W).
        Formats outside FORMATS decode to 0. An element that starts mid-dword may spill into later dwords:
        need = ceil((byte_shift + size) / 4) loaded dwords, then the window is shifted so decode sees a
        packed stream starting at the element. num_records bounds checking uses the starting offset only."""
        if fixed_fmt is not None:
            self.fixed_format_codes.add(fixed_fmt)
        else:
            self.dynamic_format = True
        if any(m in mods for m in ("lds", "tfe", "d16")):
            raise Unsupported(op + " form")
        d16 = "_d16_" in op
        ncomp = len(op.rsplit("_", 1)[1])
        vd = a[0]
        addr, d3, in_b = self.buffer_address(a, mods)
        if fixed_fmt is not None:
            fmt = self.cu(fixed_fmt)          # tbuffer: format from the instruction
        else:
            fmt = self.tmp()
            self.emit(f"{fmt} = OpBitFieldUExtract %uint {d3} {self.cu(12)} {self.cu(7)}")
        # Byte address of the element, split into a dword-aligned address and a byte shift (for 1/2-byte formats).
        alo, bsh, al64, aligned, b3 = (self.tmp() for _ in range(5))
        self.emit(f"{alo} = OpUConvert %uint {addr}")
        self.emit(f"{b3} = OpBitwiseAnd %uint {alo} {self.cu(3)}")
        self.emit(f"{bsh} = OpIMul %uint {b3} {self.cu(8)}")
        self.emit(f"{al64} = OpUConvert %ulong {b3}")
        self.emit(f"{aligned} = OpISub %ulong {addr} {al64}")
        # How many dwords the element occupies, including a mid-dword start:
        # need = ceil((byte_shift + size) / 4) = (b3 + size + 3) >> 2. A 16-byte
        # element that starts at byte 3 spans 5 dwords. Extra unused loads re-read
        # dword 0 so a load never walks an out-of-range index.
        need = self.cu(1)
        for code, (size, _, _) in self.FORMATS.items():
            e, sum_, n2 = (self.tmp() for _ in range(3))
            self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code)}")
            self.emit(f"{sum_} = OpIAdd %uint {b3} {self.cu(size + 3)}")
            self.emit(f"{n2} = OpShiftRightLogical %uint {sum_} {self.cu(2)}")
            sel = self.tmp()
            self.emit(f"{sel} = OpSelect %uint {e} {n2} {need}")
            need = sel
        cond = self.tmp()
        self.emit(f"{cond} = OpLogicalAnd %bool {self.active_b()} {in_b}")
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        pre_label = self.current_label_placeholder()
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {cond} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(aligned)}")
        loaded = []
        for i in range(5):
            ok, idx, p, v = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{ok} = OpULessThan %bool {self.cu(i)} {need}")
            self.emit(f"{idx} = OpSelect %uint {ok} {self.cu(i)} {self.cu(0)}")
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {idx}")
            self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
            loaded.append(v)
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")
        words = []
        for v_do in loaded:
            v = self.tmp()
            self.emit(f"{v} = OpPhi %uint {v_do} {lbl_do} {self.cu(0)} {pre_label}")
            words.append(v)
        # Shift the whole window so dword 0 starts at the element. (32 - bsh) & 31
        # keeps the left shift below 32 when bsh is 0; that result is discarded.
        packed = []
        for i in range(4):
            w0s, w1sh0, w1sh, w1s, nz, merged, bw = (self.tmp() for _ in range(7))
            self.emit(f"{w0s} = OpShiftRightLogical %uint {words[i]} {bsh}")
            self.emit(f"{w1sh0} = OpISub %uint {self.cu(32)} {bsh}")
            self.emit(f"{w1sh} = OpBitwiseAnd %uint {w1sh0} {self.cu(31)}")
            self.emit(f"{w1s} = OpShiftLeftLogical %uint {words[i + 1]} {w1sh}")
            self.emit(f"{nz} = OpINotEqual %bool {bsh} {self.cu(0)}")
            self.emit(f"{merged} = OpBitwiseOr %uint {w0s} {w1s}")
            self.emit(f"{bw} = OpSelect %uint {nz} {merged} {words[i]}")
            packed.append(bw)
        words = packed
        # Decode every supported format, then pick by FORMAT.
        comps = [self.cu(0)] * 4
        for code in self.FORMATS:
            dec = self.decode_format(words, code)
            e = self.tmp()
            self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code)}")
            nxt = []
            for c in range(4):
                s = self.tmp()
                self.emit(f"{s} = OpSelect %uint {e} {dec[c]} {comps[c]}")
                nxt.append(s)
            comps = nxt
        # DST_SEL swizzle: 0 -> 0, 1 -> 1.0 (float/norm) or integer 1 (UINT/SINT; SHADER-86), 4..7 -> X..W.
        is_int_fmt = "%false"
        for code_v, (_, kind, _) in self.FORMATS.items():
            if kind in ("u32", "uint"):
                e, o = self.tmp(), self.tmp()
                self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code_v)}")
                self.emit(f"{o} = OpLogicalOr %bool {is_int_fmt} {e}")
                is_int_fmt = o
        dst1 = self.tmp()
        self.emit(f"{dst1} = OpSelect %uint {is_int_fmt} {self.cu(1)} {self.cu(0x3F800000)}")
        outs = []
        for c in range(ncomp):
            sel = self.tmp()
            self.emit(f"{sel} = OpBitFieldUExtract %uint {d3} {self.cu(3 * c)} {self.cu(3)}")
            r = self.cu(0)
            for code_v, src in ((1, dst1), (4, comps[0]), (5, comps[1]), (6, comps[2]), (7, comps[3])):
                e, s = self.tmp(), self.tmp()
                self.emit(f"{e} = OpIEqual %bool {sel} {self.cu(code_v)}")
                self.emit(f"{s} = OpSelect %uint {e} {src} {r}")
                r = s
            rb = self.tmp()
            self.emit(f"{rb} = OpSelect %uint {in_b} {r} {self.cu(0)}")
            if d16:
                outs.append(rb)
            else:
                self.dst({"k": "vgpr", "r": vd["r"] + c, "n": 1}, rb)
        if d16:
            # d16: two 16-bit results per register (x | y << 16, z | w << 16): f16 for float/normalized formats,
            # the low 16 bits for integer formats (FORMATS kinds u32/uint).
            is_int = "%false"
            for code_v, (_, kind, _) in self.FORMATS.items():
                if kind in ("u32", "uint"):
                    e, o = self.tmp(), self.tmp()
                    self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code_v)}")
                    self.emit(f"{o} = OpLogicalOr %bool {is_int} {e}")
                    is_int = o
            for i in range(0, len(outs), 2):
                lo, hi = outs[i], (outs[i + 1] if i + 1 < len(outs) else self.cu(0))
                ph = self.pack_f16_pair_int(self.tof(lo), self.tof(hi))   # a memory d16 result, see there
                il, ih, pi, r = (self.tmp() for _ in range(4))
                self.emit(f"{il} = OpBitwiseAnd %uint {lo} {self.cu(0xFFFF)}")
                self.emit(f"{ih} = OpShiftLeftLogical %uint {hi} {self.cu(16)}")
                self.emit(f"{pi} = OpBitwiseOr %uint {il} {ih}")
                self.emit(f"{r} = OpSelect %uint {is_int} {pi} {ph}")
                self.dst({"k": "vgpr", "r": vd["r"] + i // 2, "n": 1}, r)

    def global_mem(self, op, a, mods):
        """global_load_dword[xN] vD, vaddr, saddr|off offset:N / global_store_dword[xN] vaddr, vD, saddr|off.
        Address = (saddr: 64-bit SGPR pair + zero-extended 32-bit vaddr | off: 64-bit VGPR pair) + offset,
        an Xbox GPU VA translated through the heap base. Stores and loads only take effect in active lanes."""
        load = op.startswith("global_load")
        if any(m in mods for m in ("lds", "tfe", "d16")):
            raise Unsupported(op + " form")
        sub = {"global_load_ushort": (16, False), "global_load_sshort": (16, True),
               "global_load_ubyte": (8, False), "global_load_sbyte": (8, True)}.get(op)
        n = 1 if (op.endswith("dword") or sub) else int(op.rsplit("x", 1)[1])
        data, vaddr, saddr = (a[0], a[1], a[2]) if load else (a[1], a[0], a[2])
        # The instruction offset is a 12-bit signed byte offset, and llvm-mc prints it signed (offset:-4 for the
        # field 0xffc; it rejects offset:0xffc and anything outside -2048..2047), which is the only spelling the
        # disassembler and build.py deliver. RX 6800 (gfx1030): pointer 0x1000, offset:-4 reads byte 0xffc.
        inst = [m.split(":", 1)[1] for m in mods if m.startswith("offset:")]
        if len(inst) > 1:
            raise Unsupported(f"{op} offset form")
        try:
            inst = int(inst[0], 0) if inst else 0
        except ValueError:
            raise Unsupported(f"{op} offset:{inst[0]}") from None
        if 0x800 <= inst <= 0xFFF:
            inst -= 0x1000            # the raw 12-bit field from a tool that prints it unsigned: same offset
        if not -0x800 <= inst <= 0x7FF:
            raise Unsupported(f"{op} offset:{inst}")     # not a 12-bit value: don't guess an address
        if saddr["k"] == "sgpr" and saddr.get("n") == 2:
            lo, hi = self.load_reg("sgpr", saddr["r"]), self.load_reg("sgpr", saddr["r"] + 1)
            voff = self.load_reg("vgpr", vaddr["r"])
            l64, h64, sh, base, v64, va = (self.tmp() for _ in range(6))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
            self.emit(f"{v64} = OpUConvert %ulong {voff}")
            self.emit(f"{va} = OpIAdd %ulong {base} {v64}")
        elif saddr["k"] == "special" and saddr["t"] == "off" and vaddr["k"] == "vgpr" and vaddr.get("n") == 2:
            lo, hi = self.load_reg("vgpr", vaddr["r"]), self.load_reg("vgpr", vaddr["r"] + 1)
            l64, h64, sh, va = (self.tmp() for _ in range(4))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{va} = OpBitwiseOr %ulong {sh} {l64}")
        else:
            raise Unsupported(op + " addressing")
        self.uses_psb = True
        off64, full = self.tmp(), self.tmp()
        ioff = self.tmp()
        self.emit(f"{ioff} = OpBitcast %uint {self.as_int(self.cu(inst & 0xFFFFFFFF))}")
        self.emit(f"{off64} = OpSConvert %long {self.as_int(ioff)}")
        o2 = self.tmp()
        self.emit(f"{o2} = OpBitcast %ulong {off64}")
        self.emit(f"{full} = OpIAdd %ulong {va} {o2}")
        self.uses_long = True
        act = self.active_b()
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        if load:
            # Inactive lanes don't load (and keep their registers): guard the access, merge with OpPhi per dword.
            olds = [self.load_reg("vgpr", data["r"] + i) for i in range(n)]
            pre = self.current_label_placeholder()
            self.emit(f"OpSelectionMerge {lbl_end} None")
            self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
            self.emit(f"{lbl_do} = OpLabel")
            vals = []
            if sub:
                # 8/16-bit load: read the containing dword and extract (zero- or sign-extended)
                bits, signed = sub
                lo32, b3, b64, al, ptr1, w, shb, v = (self.tmp() for _ in range(8))
                self.emit(f"{lo32} = OpUConvert %uint {full}")
                self.emit(f"{b3} = OpBitwiseAnd %uint {lo32} {self.cu(3)}")
                self.emit(f"{b64} = OpUConvert %ulong {b3}")
                self.emit(f"{al} = OpISub %ulong {full} {b64}")
                self.emit(f"{ptr1} = OpConvertUToPtr %_ptr_PSB_uint {self.host_addr(al)}")
                self.emit(f"{w} = OpLoad %uint {ptr1} Aligned 4")
                self.emit(f"{shb} = OpShiftLeftLogical %uint {b3} {self.cu(3)}")
                self.emit(f"{v} = {'OpBitFieldSExtract' if signed else 'OpBitFieldUExtract'} %uint {w} {shb} {self.cu(bits)}")
                vals.append(v)
            else:
                ptr = self.tmp()
                self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
                for i in range(n):
                    p, v = self.tmp(), self.tmp()
                    self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                    self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
                    vals.append(v)
            self.emit(f"OpBranch {lbl_end}")
            self.emit(f"{lbl_end} = OpLabel")
            phis = []
            for i in range(n):                 # all OpPhi first: they must open the merge block
                ph = self.tmp()
                self.emit(f"{ph} = OpPhi %uint {vals[i]} {lbl_do} {olds[i]} {pre}")
                phis.append(ph)
            for i, ph in enumerate(phis):
                self.store_reg("vgpr", data["r"] + i, ph)
        else:
            vals = [self.load_reg("vgpr", data["r"] + i) for i in range(n)]
            self.emit(f"OpSelectionMerge {lbl_end} None")
            self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
            self.emit(f"{lbl_do} = OpLabel")
            ptr = self.tmp()
            self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
            for i in range(n):
                p = self.tmp()
                self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                self.emit(f"OpStore {p} {vals[i]} Aligned 4")
            self.emit(f"OpBranch {lbl_end}")
            self.emit(f"{lbl_end} = OpLabel")

    def buffer_address(self, a, mods):
        """(64-bit element address id, V# word3 id, in_bounds bool id) for a MUBUF instruction: vdata, vaddr, srsrc, soffset."""
        vaddr, rsrc = a[1], a[2]
        soff = a[3] if len(a) > 3 else {"k": "imm", "v": 0}
        if rsrc["k"] != "sgpr" or rsrc.get("n") != 4:
            raise Unsupported("buffer V#")
        d0, d1, d2, d3 = (self.load_reg("sgpr", rsrc["r"] + i) for i in (0, 1, 2, 3))
        hi, stride = self.tmp(), self.tmp()
        self.emit(f"{hi} = OpBitwiseAnd %uint {d1} {self.cu(0xFFFF)}")
        self.emit(f"{stride} = OpBitFieldUExtract %uint {d1} {self.cu(16)} {self.cu(14)}")
        inst = next((int(m.split(":")[1], 0) for m in mods if m.startswith("offset:")), 0)
        total = self.tmp()
        self.emit(f"{total} = OpIAdd %uint {self.src_u(soff)} {self.cu(inst)}")
        vregs = [] if vaddr["k"] == "special" else [vaddr["r"] + i for i in range(vaddr.get("n", 1))]
        idx = None
        if "idxen" in mods:
            idx = self.load_reg("vgpr", vregs[0])
            m, t = self.tmp(), self.tmp()
            self.emit(f"{m} = OpIMul %uint {idx} {stride}")
            self.emit(f"{t} = OpIAdd %uint {total} {m}")
            total = t
        if "offen" in mods:
            o = self.load_reg("vgpr", vregs[1] if "idxen" in mods else vregs[0])
            t = self.tmp()
            self.emit(f"{t} = OpIAdd %uint {total} {o}")
            total = t
        self.uses_psb = True
        l64, h64, sh, base, o64, addr = (self.tmp() for _ in range(6))
        self.emit(f"{l64} = OpUConvert %ulong {d0}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{base} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{o64} = OpUConvert %ulong {total}")
        self.emit(f"{addr} = OpIAdd %ulong {base} {o64}")
        is_raw, tot_ok, in_b = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{is_raw} = OpIEqual %bool {stride} {self.cu(0)}")
        self.emit(f"{tot_ok} = OpULessThan %bool {total} {d2}")
        if idx is not None:
            idx_ok = self.tmp()
            self.emit(f"{idx_ok} = OpULessThan %bool {idx} {d2}")
            self.emit(f"{in_b} = OpSelect %bool {is_raw} {tot_ok} {idx_ok}")
        else:
            d2_64, st_64, lim, str_ok = (self.tmp() for _ in range(4))
            self.emit(f"{d2_64} = OpUConvert %ulong {d2}")
            self.emit(f"{st_64} = OpUConvert %ulong {stride}")
            self.emit(f"{lim} = OpIMul %ulong {d2_64} {st_64}")
            self.emit(f"{str_ok} = OpULessThan %bool {o64} {lim}")
            self.emit(f"{in_b} = OpSelect %bool {is_raw} {tot_ok} {str_ok}")
        return addr, d3, in_b

    def buffer_store(self, op, a, mods):
        """buffer_store_dword[xN] (raw) and buffer_store_format_* (f32 formats stored raw, f16x2/x4, unorm8x4).
        Only active lanes store, inside a structured if."""
        if any(m in mods for m in ("lds", "tfe", "d16")):
            raise Unsupported(op + " form")
        d16 = "_d16_" in op
        vd = a[0]
        addr, d3, in_b = self.buffer_address(a, mods)
        if op.startswith("buffer_store_dword"):
            n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
            words = [self.load_reg("vgpr", vd["r"] + i) for i in range(n)]
            plans = None
        else:
            n = len(op.rsplit("_", 1)[1])
            if d16:
                # two 16-bit values per register: f16 for float kinds, 16-bit integers for integer kinds
                regs = [self.load_reg("vgpr", vd["r"] + i) for i in range((n + 1) // 2)]
                comps, icomps = [], []
                for i in range(n):
                    r = regs[i // 2]
                    v2, f, iv = self.tmp(), self.tmp(), self.tmp()
                    sh = self.tmp()
                    self.emit(f"{sh} = OpShiftRightLogical %uint {r} {self.cu(16 * (i % 2))}")
                    self.emit(f"{iv} = OpBitwiseAnd %uint {sh} {self.cu(0xFFFF)}")
                    self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {iv}")
                    self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
                    comps.append(self.tou(f))
                    icomps.append(iv)
            else:
                comps = [self.load_reg("vgpr", vd["r"] + i) for i in range(n)]
                icomps = comps
            fmt = self.tmp()
            self.emit(f"{fmt} = OpBitFieldUExtract %uint {d3} {self.cu(12)} {self.cu(7)}")
            # Encodings for the formats that appear in store-heavy shaders; others store nothing.
            f32 = {22: 1, 64: 2, 74: 3, 77: 4}
            # Float views of the components, computed before any branch so every branch can use them.
            fl = [self.tof(c) for c in comps] + ["%f0"] * (4 - len(comps))
            plans = (fmt, comps, f32, fl, icomps)
            words = None
        act = self.tmp()
        self.emit(f"{act} = OpLogicalAnd %bool {self.active_b()} {in_b}")
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(addr)}")
        if words is not None:
            for i, w in enumerate(words):
                p = self.tmp()
                self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
                self.emit(f"OpStore {p} {w} Aligned 4")
        else:
            fmt, comps, f32, fl, icomps = plans
            # Every format in FORMATS (the same table the loads decode); unknown formats store nothing.
            fc = comps + [self.cu(0)] * (4 - len(comps))
            ic = icomps + [self.cu(0)] * (4 - len(icomps))
            for code, (size, kind, widths) in self.FORMATS.items():
                e = self.tmp()
                self.emit(f"{e} = OpIEqual %bool {fmt} {self.cu(code)}")
                ld, le = self.tmp() + "_s", self.tmp() + "_se"
                self.emit(f"OpSelectionMerge {le} None")
                self.emit(f"OpBranchConditional {e} {ld} {le}")
                self.emit(f"{ld} = OpLabel")
                ncomp = min(len(widths), n)
                fcs = fc[:ncomp] + [self.cu(0)] * (len(widths) - ncomp)
                ics = ic[:ncomp] + [self.cu(0)] * (len(widths) - ncomp)
                words = self.encode_format(fcs + [self.cu(0)] * 4, ics + [self.cu(0)] * 4, code)
                if size >= 4:
                    for k, w in enumerate(words[:size // 4]):
                        pp = self.tmp()
                        self.emit(f"{pp} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(k)}")
                        self.emit(f"OpStore {pp} {w} Aligned 4")
                else:
                    # 1- or 2-byte element: merge into its dword with atomics so neighbours are kept.
                    al, sh, sh8, ap, mk, mks, inv, val, vs, t1, t2 = (self.tmp() for _ in range(11))
                    self.emit(f"{sh} = OpUConvert %uint {addr}")
                    lo3, lo64 = self.tmp(), self.tmp()
                    self.emit(f"{lo3} = OpBitwiseAnd %uint {sh} {self.cu(3)}")
                    self.emit(f"{lo64} = OpUConvert %ulong {lo3}")
                    self.emit(f"{al} = OpISub %ulong {addr} {lo64}")
                    self.emit(f"{sh8} = OpShiftLeftLogical %uint {sh} {self.cu(3)}")
                    sh24 = self.tmp()
                    self.emit(f"{sh24} = OpBitwiseAnd %uint {sh8} {self.cu(24)}")
                    # MoltenVK 1.4.0 needs an addressable array element for the Metal atomic operations.
                    dword_arr = self.tmp()
                    self.emit(f"{dword_arr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(al)}")
                    self.emit(f"{ap} = OpAccessChain %_ptr_PSB_uint {dword_arr} {self.cu(0)}")
                    self.emit(f"{mk} = OpCopyObject %uint {self.cu((1 << (8 * size)) - 1)}")
                    self.emit(f"{mks} = OpShiftLeftLogical %uint {mk} {sh24}")
                    self.emit(f"{inv} = OpNot %uint {mks}")
                    self.emit(f"{val} = OpBitwiseAnd %uint {words[0]} {mk}")
                    self.emit(f"{vs} = OpShiftLeftLogical %uint {val} {sh24}")
                    self.emit(f"{t1} = OpAtomicAnd %uint {ap} {self.cu(1)} {self.cu(0)} {inv}")
                    self.emit(f"{t2} = OpAtomicOr %uint {ap} {self.cu(1)} {self.cu(0)} {vs}")
                self.emit(f"OpBranch {le}")
                self.emit(f"{le} = OpLabel")
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")

    def cmp_class(self, a, exec_only=False):
        """Test x's class against the mask and write the result to VCC/SGPR or EXEC.
        Bits: 0 sNaN, 1 qNaN, 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal, 9 +inf."""
        x, mask = self.src_f(a[-2]), self.src_u(a[-1])
        u = self.tou(x)
        sign, absu = self.tmp(), self.tmp()
        self.emit(f"{sign} = OpShiftRightLogical %uint {u} {self.cu(31)}")
        self.emit(f"{absu} = OpBitwiseAnd %uint {u} {self.cu(0x7FFFFFFF)}")
        exp_, man = self.tmp(), self.tmp()
        self.emit(f"{exp_} = OpShiftRightLogical %uint {absu} {self.cu(23)}")
        self.emit(f"{man} = OpBitwiseAnd %uint {absu} {self.cu(0x7FFFFF)}")
        neg = self.tmp_bool_nz(sign)

        def b(cond_line):
            t = self.tmp()
            self.emit(f"{t} = {cond_line}")
            return t
        is_zero = b(f"OpIEqual %bool {absu} {self.cu(0)}")
        exp0 = b(f"OpIEqual %bool {exp_} {self.cu(0)}")
        expmax = b(f"OpIEqual %bool {exp_} {self.cu(255)}")
        man0 = b(f"OpIEqual %bool {man} {self.cu(0)}")
        man_nz = b(f"OpLogicalNot %bool {man0}")
        is_inf = b(f"OpLogicalAnd %bool {expmax} {man0}")
        is_nan = b(f"OpLogicalAnd %bool {expmax} {man_nz}")
        quiet = b(f"OpUGreaterThanEqual %bool {man} {self.cu(0x400000)}")
        snan = b(f"OpLogicalAnd %bool {is_nan} " + b(f"OpLogicalNot %bool {quiet}"))
        qnan = b(f"OpLogicalAnd %bool {is_nan} {quiet}")
        denorm = b(f"OpLogicalAnd %bool {exp0} {man_nz}")
        normal = b(f"OpLogicalAnd %bool " + b(f"OpLogicalNot %bool {exp0}") + " " + b(f"OpLogicalNot %bool {expmax}"))
        pos = b(f"OpLogicalNot %bool {neg}")
        classes = [snan, qnan, b(f"OpLogicalAnd %bool {is_inf} {neg}"), b(f"OpLogicalAnd %bool {normal} {neg}"),
                   b(f"OpLogicalAnd %bool {denorm} {neg}"), b(f"OpLogicalAnd %bool {is_zero} {neg}"),
                   b(f"OpLogicalAnd %bool {is_zero} {pos}"), b(f"OpLogicalAnd %bool {denorm} {pos}"),
                   b(f"OpLogicalAnd %bool {normal} {pos}"), b(f"OpLogicalAnd %bool {is_inf} {pos}")]
        acc = "%false"
        for i, c in enumerate(classes):
            bit = b(f"OpBitwiseAnd %uint {mask} {self.cu(1 << i)}")
            on = b(f"OpINotEqual %bool {bit} {self.cu(0)}")
            hit = b(f"OpLogicalAnd %bool {on} {c}")
            acc = b(f"OpLogicalOr %bool {acc} {hit}")
        acc = b(f"OpLogicalAnd %bool {acc} {self.active_b()}")   # inactive lanes write 0
        if exec_only:
            self.emit(f"OpStore %active {acc}")
            self.uses_active = True
            return
        r = b(f"OpSelect %uint {acc} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
        d = a[0]
        if len(a) == 2 or (d["k"] == "special" and d["t"] in ("vcc", "vcc_lo")):
            self.emit(f"OpStore %vcc {r}")
        else:
            self.mask_dst(d, r)

    @staticmethod
    def mod_int(mods, name, default):
        m = next((m for m in mods if m.startswith(name + ":")), None)
        return int(m.split(":", 1)[1], 0) if m else default

    # LDS atomics (non-returning name -> SPIR-V atomic); _rtn_ forms share the entry.
    LDS_ATOMICS = {"ds_add_u32": "OpAtomicIAdd", "ds_sub_u32": "OpAtomicISub", "ds_min_u32": "OpAtomicUMin",
                   "ds_max_u32": "OpAtomicUMax", "ds_min_i32": "OpAtomicSMin", "ds_max_i32": "OpAtomicSMax",
                   "ds_and_b32": "OpAtomicAnd", "ds_or_b32": "OpAtomicOr", "ds_xor_b32": "OpAtomicXor",
                   "ds_wrxchg_rtn_b32": "OpAtomicExchange"}

    def lds_ptr(self, addr, byte_off):
        """Pointer to the LDS dword at (addr + byte_off) / 4, wrapped to the array (LDS is a Workgroup uint array)."""
        s1, i, m, p = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{s1} = OpIAdd %uint {addr} {self.cu(byte_off)}")
        self.emit(f"{i} = OpShiftRightLogical %uint {s1} {self.cu(2)}")
        self.emit(f"{m} = OpBitwiseAnd %uint {i} {self.cu(self.lds_dwords - 1)}")
        self.emit(f"{p} = OpAccessChain %_ptr_WG_uint %lds {m}")
        return p

    def lds_op(self, op, a, mods):
        """LDS (GFX10 DS encoding). Addresses are bytes. offset:N is bytes; read2/write2 offset0/offset1 count
        data-size units (x64 for st64). Writes and atomics run only in active lanes; results land in VGPRs through
        dst (inactive lanes unchanged). Sub-dword writes use atomic and/or so neighbouring lanes don't race."""
        if self.stage not in ("cs", "ms"):
            raise Unsupported(op + " outside compute")
        self.uses_lds = True
        rtn = "_rtn_" in op
        name = op if op == "ds_wrxchg_rtn_b32" else op.replace("_rtn_", "_")
        act = self.active_b()
        lbl_do, lbl_end = self.tmp() + "_do", self.tmp() + "_end"
        self.emit(f"OpSelectionMerge {lbl_end} None")
        self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
        self.emit(f"{lbl_do} = OpLabel")
        off = self.mod_int(mods, "offset", 0)
        sc, sem = self.cu(2), self.cu(0)
        sizes = {"b32": 1, "b64": 2, "b96": 3, "b128": 4}
        if op in ("ds_read_b32", "ds_read_b64", "ds_read_b96", "ds_read_b128"):
            addr = self.src_u(a[1])
            for k in range(sizes[op.rsplit("_", 1)[1]]):
                v = self.tmp()
                self.emit(f"{v} = OpLoad %uint {self.lds_ptr(addr, off + 4 * k)}")
                self.dst({"k": "vgpr", "r": a[0]["r"] + k, "n": 1}, v)
        elif op in ("ds_write_b32", "ds_write_b64", "ds_write_b96", "ds_write_b128"):
            addr = self.src_u(a[0])
            for k in range(sizes[op.rsplit("_", 1)[1]]):
                v = self.load_reg("vgpr", a[1]["r"] + k)
                self.emit(f"OpStore {self.lds_ptr(addr, off + 4 * k)} {v}")
        elif op in ("ds_read2_b32", "ds_read2st64_b32", "ds_write2_b32", "ds_write2st64_b32",
                    "ds_read2_b64", "ds_read2st64_b64", "ds_write2_b64", "ds_write2st64_b64"):
            # Two elements at ADDR + offset0 * unit and ADDR + offset1 * unit; unit = element size (4 or 8 bytes),
            # times 64 for st64. A b64 element is two consecutive dwords (low first): read2_b64 fills
            # v[d:d+3], write2_b64 takes two VGPR pairs.
            words = 2 if op.endswith("_b64") else 1
            unit = 4 * words * (64 if "st64" in op else 1)
            offs = (self.mod_int(mods, "offset0", 0) * unit, self.mod_int(mods, "offset1", 0) * unit)
            if op.startswith("ds_read2"):
                addr = self.src_u(a[1])
                vals = []
                for o in offs:
                    for w in range(words):
                        v = self.tmp()
                        self.emit(f"{v} = OpLoad %uint {self.lds_ptr(addr, o + 4 * w)}")
                        vals.append(v)
                for k, v in enumerate(vals):         # every load before the first write (vdst may alias vaddr)
                    self.dst({"k": "vgpr", "r": a[0]["r"] + k, "n": 1}, v)
            else:
                addr = self.src_u(a[0])
                for k, o in enumerate(offs):
                    data = a[1 + k]
                    for w in range(words):
                        if words == 1:
                            v = self.src_u(data)
                        elif data["k"] == "vgpr" and data.get("n") == 2:
                            v = self.load_reg("vgpr", data["r"] + w)
                        else:
                            raise Unsupported(op + " data")
                        self.emit(f"OpStore {self.lds_ptr(addr, o + 4 * w)} {v}")
        elif op in ("ds_read_u16", "ds_read_i16", "ds_read_u8", "ds_read_i8", "ds_read_u16_d16_hi", "ds_read_u16_d16",
                    "ds_read_u8_d16", "ds_read_u8_d16_hi", "ds_read_i8_d16", "ds_read_i8_d16_hi"):
            bits = 16 if "16" in op.split("_")[2] else 8
            addr = self.src_u(a[1])
            s1, sh, sh2, w, v = (self.tmp() for _ in range(5))
            self.emit(f"{s1} = OpIAdd %uint {addr} {self.cu(off)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {s1} {self.cu(3)}")     # byte address -> bit offset
            self.emit(f"{sh2} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{w} = OpLoad %uint {self.lds_ptr(addr, off)}")
            f = "OpBitFieldSExtract" if op.split("_")[2].startswith("i") else "OpBitFieldUExtract"
            self.emit(f"{v} = {f} %uint {w} {sh2} {self.cu(bits)}")
            d = a[0]
            if "_d16" in op:
                # u8/i8: v is the 8-bit field extended to 32 bits; the insert keeps only its low 16 bits, so the
                # i8 sign extension is 8 -> 16 as the ISA says. The other half of the destination is kept.
                hi = op.endswith("_hi")
                r = self.tmp()
                self.emit(f"{r} = OpBitFieldInsert %uint {self.load_reg('vgpr', d['r'])} {v} "
                          f"{self.cu(16 if hi else 0)} {self.cu(16)}")
                v = r
            self.dst(d, v)
        elif op in ("ds_write_b16", "ds_write_b8", "ds_write_b16_d16_hi", "ds_write_b8_d16_hi"):
            bits = 16 if "b16" in op else 8
            addr = self.src_u(a[0])
            val = self.src_u(a[1])
            if op.endswith("_d16_hi"):
                hv = self.tmp()
                self.emit(f"{hv} = OpShiftRightLogical %uint {val} {self.cu(16)}")
                val = hv
            s1, sh, sh2, fld, pos, msk, inv = (self.tmp() for _ in range(7))
            self.emit(f"{s1} = OpIAdd %uint {addr} {self.cu(off)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {s1} {self.cu(3)}")
            self.emit(f"{sh2} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{fld} = OpBitwiseAnd %uint {val} {self.cu((1 << bits) - 1)}")
            self.emit(f"{pos} = OpShiftLeftLogical %uint {fld} {sh2}")
            self.emit(f"{msk} = OpShiftLeftLogical %uint {self.cu((1 << bits) - 1)} {sh2}")
            self.emit(f"{inv} = OpNot %uint {msk}")
            p = self.lds_ptr(addr, off)
            t1, t2 = self.tmp(), self.tmp()
            self.emit(f"{t1} = OpAtomicAnd %uint {p} {sc} {sem} {inv}")
            self.emit(f"{t2} = OpAtomicOr %uint {p} {sc} {sem} {pos}")
        elif name in self.LDS_ATOMICS:
            d, ad, dat = (a[0], a[1], a[2]) if rtn else (None, a[0], a[1])
            p = self.lds_ptr(self.src_u(ad), off)
            old = self.tmp()
            self.emit(f"{old} = {self.LDS_ATOMICS[name]} %uint {p} {sc} {sem} {self.src_u(dat)}")
            if d is not None:
                self.dst(d, old)
        elif op in ("ds_cmpst_rtn_b32", "ds_cmpst_b32"):
            # tmp = M[A]; M[A] = (tmp == DATA) ? DATA2 : tmp; the _rtn_ form returns tmp, ds_cmpst_b32 drops it.
            d, ad, cmp_, src = (a[0], a[1], a[2], a[3]) if rtn else (None, a[0], a[1], a[2])
            p = self.lds_ptr(self.src_u(ad), off)
            old = self.tmp()
            self.emit(f"{old} = OpAtomicCompareExchange %uint {p} {sc} {sem} {sem} {self.src_u(src)} {self.src_u(cmp_)}")
            if d is not None:
                self.dst(d, old)
        else:
            raise Unsupported(op)
        self.emit(f"OpBranch {lbl_end}")
        self.emit(f"{lbl_end} = OpLabel")

    def cl(self, v):
        """64-bit unsigned constant id (declared with the PhysicalStorageBuffer types)."""
        v &= 0xFFFFFFFFFFFFFFFF
        if v not in self.lconsts:
            self.lconsts[v] = f"%cl{v:x}"
        self.uses_psb = True
        return self.lconsts[v]

    def local_index(self):
        """LocalInvocationIndex of a compute or mesh shader invocation (thread i of its workgroup): the mesh shader
        loads the built-in; a compute shader flattens LocalInvocationId with the group size from its header."""
        li = self.tmp()
        if self.stage == "ms":
            self.emit(f"{li} = OpLoad %uint %local_index")
            return li
        lid, x, y, z, lx, t1, t2 = (self.tmp() for _ in range(7))
        sx, sy, _ = self.cs["size"]
        self.emit(f"{lid} = OpLoad %v3uint %local_id")
        self.emit(f"{x} = OpCompositeExtract %uint {lid} 0")
        self.emit(f"{y} = OpCompositeExtract %uint {lid} 1")
        self.emit(f"{z} = OpCompositeExtract %uint {lid} 2")
        self.emit(f"{t1} = OpIMul %uint {y} {self.cu(sx)}")
        self.emit(f"{t2} = OpIMul %uint {z} {self.cu(sx * sy)}")
        self.emit(f"{lx} = OpIAdd %uint {x} {t1}")
        self.emit(f"{li} = OpIAdd %uint {lx} {t2}")
        return li

    def wave_lane(self):
        """This invocation's hardware lane within its wave. Compute and mesh shaders: LocalInvocationIndex mod the
        wave size (exact: thread i of a group is wave i / W, lane i % W). Other stages: the subgroup lane, which is
        the hardware lane when the runtime runs them with subgroup size == wave size."""
        lane = self.tmp()
        if self.stage in ("cs", "ms"):
            self.emit(f"{lane} = OpBitwiseAnd %uint {self.local_index()} {self.cu(63 if self.wave64 else 31)}")
        else:
            self.uses_subgroup = True
            self.emit(f"{lane} = OpLoad %uint %sg_lane")
        return lane

    # ---- wave64 emulation (--wave64-emulate) ----
    # A wave64 compute or mesh shader runs exact on any subgroup size. Each lane is one invocation as usual, but no
    # subgroup operation is used: everything a lane needs from the rest of its wave goes through a Workgroup buffer
    # (%x_buf). The shader runs in rounds (assemble: the emulated scheduler). In a round every wave that isn't done
    # or waiting at s_barrier runs one segment of its program; then the whole workgroup meets at one OpControlBarrier
    # in uniform control flow. A cross-lane operation ends a segment after its lane has written its share, and the
    # code after it runs in the next round, when every lane of the wave has. Buffer layout, L = x_lanes:
    #   [0, 2L)              per-lane data, double-buffered by round parity (written in round r, read in r + 1)
    #   [2L, 2L + 6W)        two mask words per wave, ORed atomically, 3 slots by round mod 3 (W = waves)
    #   [2L + 6W, +6)        per round slot: lanes not done, lanes neither done nor waiting at s_barrier
    #   last word            a sink for the stores of lanes that have nothing to reset
    def x_sizes(self):
        w = self.x_lanes // 64
        mask = 2 * self.x_lanes
        sched = mask + 6 * w
        return w, mask, sched, sched + 6

    def x_split(self):
        """End the current segment here: the rest of the block runs as a new segment in the next round."""
        seg = self.x_seg
        self.x_seg += 1
        self.emit(f"OpStore %pc {self.cu(seg)}")
        self.x_marks.append((len(self.body), seg))
        self.desc_cache.clear()
        self.sgpr_imm.clear()
        self.block_label = f"%blk{seg}"
        self.x_dirty = set()

    def x_word(self, idx):
        p, v = self.tmp(), self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_WG_uint_x %x_buf {idx}")
        self.emit(f"{v} = OpLoad %uint {p}")
        return v

    def x_mask_slot(self, it):
        """Index of the wave's first mask word in the slot of round `it`."""
        s, a, b, c = (self.tmp() for _ in range(4))
        w, mask, _, _ = self.x_sizes()
        self.emit(f"{s} = OpUMod %uint {it} {self.cu(3)}")
        self.emit(f"{a} = OpIMul %uint {s} {self.cu(w)}")
        self.emit(f"{b} = OpIAdd %uint {a} %x_wave")
        self.emit(f"{c} = OpIMul %uint {b} {self.cu(2)}")
        r = self.tmp()
        self.emit(f"{r} = OpIAdd %uint {c} {self.cu(mask)}")
        return r

    def x_round(self, data=None, bits=None):
        """One exchange: this lane's uint `data` goes to its slot, `bits` (lo, hi; None = nothing) are ORed into
        the wave's two mask words. Returns (read, (lo, hi)): read(lane) loads `data` of a lane (0-63) of this wave,
        and (lo, hi) is the OR over all 64 lanes. Both are only valid in the code emitted after this call, which
        runs one round later."""
        it = self.tmp()
        self.emit(f"{it} = OpLoad %uint %x_it")
        if data is not None:
            par, base, i, p = (self.tmp() for _ in range(4))
            self.emit(f"{par} = OpBitwiseAnd %uint {it} {self.cu(1)}")
            self.emit(f"{base} = OpIMul %uint {par} {self.cu(self.x_lanes)}")
            self.emit(f"{i} = OpIAdd %uint {base} %x_li")
            self.emit(f"{p} = OpAccessChain %_ptr_WG_uint_x %x_buf {i}")
            self.emit(f"OpStore {p} {data}")
        if bits is not None:
            m = self.x_mask_slot(it)
            for k, b in enumerate(bits):
                if b is None:
                    continue
                i, p, old = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{i} = OpIAdd %uint {m} {self.cu(k)}")
                self.emit(f"{p} = OpAccessChain %_ptr_WG_uint_x %x_buf {i}")
                self.emit(f"{old} = OpAtomicOr %uint {p} {self.cu(2)} {self.cu(0)} {b}")
        self.x_split()
        now, prev = self.tmp(), self.tmp()
        self.emit(f"{now} = OpLoad %uint %x_it")
        self.emit(f"{prev} = OpISub %uint {now} {self.cu(1)}")
        res = None
        if bits is not None:
            m = self.x_mask_slot(prev)
            hi_i = self.tmp()
            self.emit(f"{hi_i} = OpIAdd %uint {m} {self.cu(1)}")
            res = (self.x_word(m), self.x_word(hi_i))

        def read(lane):
            par, base, i = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{par} = OpBitwiseAnd %uint {prev} {self.cu(1)}")
            self.emit(f"{base} = OpIMul %uint {par} {self.cu(self.x_lanes)}")
            self.emit(f"{i} = OpIAdd %uint {base} %x_wbase")
            j = self.tmp()
            self.emit(f"{j} = OpIAdd %uint {i} {lane}")
            return self.x_word(j)
        return read, res

    def x_memory_order(self, op):
        """On hardware a wave's lanes run each instruction together: every lane's access of one instruction comes
        before every lane's access of the next. Two halves of an emulated wave can be in different subgroups, so an
        access that conflicts with an earlier one of the same segment (read after write, write after read, write
        after write, on the same memory) waits for the next round, whose barrier orders the memory. Not ordered:
        scalar loads (s_load / s_buffer_load) of memory the wave wrote, and sampling an image it wrote."""
        if op in ("ds_swizzle_b32", "ds_bpermute_b32", "ds_permute_b32"):
            return
        if op.startswith("ds_"):
            kind = "lds"
        elif op.startswith(("buffer_", "tbuffer_", "global_", "flat_", "image_load", "image_store", "image_atomic")):
            kind = "mem"
        else:
            return
        store = "store" in op or op.startswith("ds_write")
        reads = not store
        writes = store or "atomic" in op or (kind == "lds" and not op.startswith("ds_read"))
        if (kind, "w") in self.x_dirty or (writes and (kind, "r") in self.x_dirty):
            self.x_split()
        if reads:
            self.x_dirty.add((kind, "r"))
        if writes:
            self.x_dirty.add((kind, "w"))

    def g_vote(self, r, kind, b):
        """r = OpGroupNonUniform{Any,All} of bool b over the wave."""
        if not self.emulate:
            self.emit(f"{r} = OpGroupNonUniform{kind} %bool {self.cu(3)} {b}")
            return
        if kind == "All":
            nb = self.tmp()
            self.emit(f"{nb} = OpLogicalNot %bool {b}")
            b = nb
        bit = self.tmp()
        self.emit(f"{bit} = OpSelect %uint {b} {self.cu(1)} {self.cu(0)}")
        lo = self.x_round(bits=(bit, None))[1][0]
        self.emit(f"{r} = {'OpIEqual' if kind == 'All' else 'OpINotEqual'} %bool {lo} {self.cu(0)}")

    def g_shuffle(self, r, v, src, zero_inactive=False):
        """r = v of lane `src` of this wave (OpGroupNonUniformShuffle; emulated: any lane 0-63). zero_inactive:
        a source lane that is off in EXEC gives 0 (ds_bpermute)."""
        if not self.emulate:
            self.emit(f"{r} = OpGroupNonUniformShuffle %uint %c3 {v} {src}")
            return
        if zero_inactive:
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.active_b()} {v} {self.cu(0)}")
            v = z
        read = self.x_round(data=v)[0]
        self.emit(f"{r} = OpCopyObject %uint {read(src)}")

    def ds_permute(self, a, mods):
        """ds_permute_b32 vD, vAddr, vData offset:N, the forward (scatter) permute: every source lane that is on in
        EXEC sends vData to lane ((vAddr + offset) / 4) mod 32 of its own half (as ds_bpermute_b32, RDNA ISA
        10.4.4: a wave64 runs it as two wave32s); a destination lane nobody sends to gets 0; EXEC gates only the
        destination write. Several sources to one lane: the highest-numbered source wins (ISA: "the final value is
        not predictable but will be the value from one of the writers"). Vulkan has no subgroup scatter, so each
        destination lane scans the 32 source lanes of its half in order and keeps the last one that targets it.
        Each source lane publishes its target plus one (0 = off in EXEC, so a lane that doesn't exist in a partial
        wave reads as "no target" as well)."""
        d, ad, dat = a[0], self.src_u(a[1]), self.src_u(a[2])
        off = self.mod_int(mods, "offset", 0)
        s1, sh, dl, dp, tp = (self.tmp() for _ in range(5))
        self.emit(f"{s1} = OpIAdd %uint {ad} {self.cu(off)}")
        self.emit(f"{sh} = OpShiftRightLogical %uint {s1} {self.cu(2)}")
        self.emit(f"{dl} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
        self.emit(f"{dp} = OpIAdd %uint {dl} {self.cu(1)}")
        self.emit(f"{tp} = OpSelect %uint {self.active_b()} {dp} {self.cu(0)}")
        if self.emulate:
            # Round 1 publishes the targets; the next segment finds each lane's winning source and parks it in
            # %x_perm; round 2 publishes the data; the segment after that reads the winner's data.
            self.uses_perm = True
            read = self.x_round(data=tp)[0]
            lane, rl, mine, half, win = (self.tmp() for _ in range(5))
            self.emit(f"{lane} = OpCopyObject %uint {self.wave_lane()}")
            self.emit(f"{rl} = OpBitwiseAnd %uint {lane} {self.cu(31)}")
            self.emit(f"{mine} = OpIAdd %uint {rl} {self.cu(1)}")
            self.emit(f"{half} = OpBitwiseAnd %uint {lane} {self.cu(32)}")
            acc = self.cu(32)
            for j in range(32):
                src, v, hit, nxt = (self.tmp() for _ in range(4))
                self.emit(f"{src} = OpBitwiseOr %uint {half} {self.cu(j)}")
                self.emit(f"{v} = OpCopyObject %uint {read(src)}")
                self.emit(f"{hit} = OpIEqual %bool {v} {mine}")
                self.emit(f"{nxt} = OpSelect %uint {hit} {self.cu(j)} {acc}")
                acc = nxt
            self.emit(f"OpStore %x_perm {acc}")
            dat2 = self.src_u(a[2])
            read2 = self.x_round(data=dat2)[0]
            w, ok, half2, src, got, r, lane2 = (self.tmp() for _ in range(7))
            self.emit(f"{w} = OpLoad %uint %x_perm")
            self.emit(f"{ok} = OpULessThan %bool {w} {self.cu(32)}")
            self.emit(f"{lane2} = OpCopyObject %uint {self.wave_lane()}")
            self.emit(f"{half2} = OpBitwiseAnd %uint {lane2} {self.cu(32)}")
            wc = self.tmp()
            self.emit(f"{wc} = OpBitwiseAnd %uint {w} {self.cu(31)}")
            self.emit(f"{src} = OpBitwiseOr %uint {half2} {wc}")
            self.emit(f"{got} = OpCopyObject %uint {read2(src)}")
            self.emit(f"{r} = OpSelect %uint {ok} {got} {self.cu(0)}")
            self.dst(d, r)
            return
        self.uses_subgroup = self.uses_ballot = True
        lane, rl, mine, half = (self.tmp() for _ in range(4))
        self.emit(f"{lane} = OpCopyObject %uint {self.wave_lane()}")
        self.emit(f"{rl} = OpBitwiseAnd %uint {lane} {self.cu(31)}")
        self.emit(f"{mine} = OpIAdd %uint {rl} {self.cu(1)}")
        if self.wave64:
            sgl = self.tmp()
            self.emit(f"{sgl} = OpLoad %uint %sg_lane")
            self.emit(f"{half} = OpBitwiseAnd %uint {sgl} {self.cu(32)}")
        bal = self.tmp()
        self.emit(f"{bal} = OpGroupNonUniformBallot %v4uint_bal %c3 %true")
        acc = self.cu(0)
        for j in range(32):
            src = self.cu(j)
            if self.wave64:
                src = self.tmp()
                self.emit(f"{src} = OpBitwiseOr %uint {half} {self.cu(j)}")
            ex, shv, v, shd, hit, nxt = (self.tmp() for _ in range(6))
            # a shuffle from a lane that isn't running (the missing lanes of a partial wave) is undefined
            self.emit(f"{ex} = OpGroupNonUniformBallotBitExtract %bool %c3 {bal} {src}")
            self.emit(f"{shv} = OpGroupNonUniformShuffle %uint %c3 {tp} {src}")
            self.emit(f"{v} = OpSelect %uint {ex} {shv} {self.cu(0)}")
            self.emit(f"{shd} = OpGroupNonUniformShuffle %uint %c3 {dat} {src}")
            self.emit(f"{hit} = OpIEqual %bool {v} {mine}")
            self.emit(f"{nxt} = OpSelect %uint {hit} {shd} {acc}")
            acc = nxt
        self.dst(d, acc)

    def sub_lane(self, r):
        """r = the lane the subgroup shuffles above index: the subgroup lane, or the wave lane when emulated."""
        if self.emulate:
            self.emit(f"{r} = OpCopyObject %uint {self.wave_lane()}")
        else:
            self.emit(f"{r} = OpLoad %uint %sg_lane")

    def exec_half_write(self, half, val, const=None):
        """Write one 32-bit half of EXEC. In the per-lane model a lane is active iff its bit is set. A constant
        sets each lane from its own bit; a register value (mask representation: all-ones / zero per lane) sets
        the lanes of that half from their own copy. Lanes of the other half are unchanged."""
        lane = self.wave_lane()
        rel, inh, bit, on, cur, new = (self.tmp() for _ in range(6))
        if half == 0:
            self.emit(f"{inh} = OpULessThan %bool {lane} {self.cu(32)}")
            self.emit(f"{rel} = OpCopyObject %uint {lane}")
        else:
            self.emit(f"{inh} = OpUGreaterThanEqual %bool {lane} {self.cu(32)}")
            self.emit(f"{rel} = OpISub %uint {lane} {self.cu(32)}")
        if const is not None:
            r31 = self.tmp()
            self.emit(f"{r31} = OpBitwiseAnd %uint {rel} {self.cu(31)}")
            self.emit(f"{bit} = OpBitFieldUExtract %uint {self.cu(const & 0xFFFFFFFF)} {r31} {self.cu(1)}")
            self.emit(f"{on} = OpINotEqual %bool {bit} {self.cu(0)}")
        else:
            self.emit(f"{on} = OpINotEqual %bool {val} {self.cu(0)}")
        self.emit(f"{cur} = OpLoad %bool %active")
        self.emit(f"{new} = OpSelect %bool {inh} {on} {cur}")
        self.emit(f"OpStore %active {new}")
        self.uses_active = True

    def set_scc(self, b):
        """SCC = bool id (1/0)."""
        r = self.tmp()
        self.emit(f"{r} = OpSelect %uint {b} {self.cu(1)} {self.cu(0)}")
        self.emit(f"OpStore %scc {r}")

    def scc_nonzero(self, v):
        self.set_scc(self.tmp_bool_nz(v))

    def scc_add_overflow(self, x, y, r, sub=False):
        """Signed overflow of r = x + y (or x - y): SCC for s_add_i32 / s_sub_i32 / s_addk_i32."""
        a1, a2, m, sh = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        if sub:
            self.emit(f"{a1} = OpBitwiseXor %uint {x} {y}")
            self.emit(f"{a2} = OpBitwiseXor %uint {x} {r}")
        else:
            self.emit(f"{a1} = OpBitwiseXor %uint {x} {r}")
            self.emit(f"{a2} = OpBitwiseXor %uint {y} {r}")
        self.emit(f"{m} = OpBitwiseAnd %uint {a1} {a2}")
        self.emit(f"{sh} = OpShiftRightLogical %uint {m} {self.cu(31)}")
        self.scc_nonzero(sh)

    def bfe_width(self, off, wid):
        """A bitfield-extract width cut to 32 - off (off in 0-31): SPIR-V leaves offset + count > 32 undefined. The
        field then runs to bit 31, which is the hardware result: LLVM's constantFoldBFE (AMDGPUISelLowering.cpp)
        gives Src0 >> Offset there, logical for the unsigned and arithmetic for the signed extract."""
        room, fits, w = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{room} = OpISub %uint {self.cu(32)} {off}")
        self.emit(f"{fits} = OpULessThan %bool {wid} {room}")
        self.emit(f"{w} = OpSelect %uint {fits} {wid} {room}")
        return w

    def tmp_bool_nz(self, v):
        b = self.tmp()
        self.emit(f"{b} = OpINotEqual %bool {v} {self.cu(0)}")
        return b

    def scc_from_mask(self, v):
        """SCC after a lane-mask instruction: 1 if the wave's mask is non-zero. One invocation holds one lane's
        bit, so that is a vote across the subgroup, as for s_cbranch_vccz. Taken per lane, the lanes of one wave
        could follow different s_cbranch_scc paths, which a hardware wave never does: it has one SCC."""
        any_, z = self.tmp(), self.tmp()
        self.g_vote(any_, "Any", self.tmp_bool_nz(v))
        self.uses_vote = True
        self.emit(f"{z} = OpSelect %uint {any_} {self.cu(1)} {self.cu(0)}")
        self.emit(f"OpStore %scc {z}")

    def scc_logic(self, a, r):
        """SCC of a 32-bit bitwise SOP2 (s_and/or/xor/andn2/orn2/nand/nor/xnor_b32): D != 0. In a wave32 program
        an operation on lane masks (a source holds one, or it writes EXEC) has the whole wave's mask as D, so SCC
        is the vote of scc_from_mask, as for the _b64 forms. Until 2026-11-22 it was this lane's bit instead:
        `s_and_b32 s0, vcc_lo, exec_lo; s_cbranch_scc0` skipped the block in the lanes whose own bit was 0
        although the wave runs it, and a loop closed by `s_andn2_b32 exec_lo, ...; s_cbranch_scc1` let each lane
        leave on its last iteration, before the wave did (tools/cf_diff.py, FUZZ-8). In a wave64 program the
        same op works on one half of a mask (lanes 0-31 or 32-63), whose SCC would be a vote of that half only;
        that case still takes the lane's bit and is not covered by the fuzzer yet."""
        writes_exec = a[0]["k"] == "special" and a[0].get("t", "").startswith("exec")
        if not self.wave64 and (writes_exec or self.holds_lane_mask(a[1]) or self.holds_lane_mask(a[2])):
            self.scc_from_mask(r)
        else:
            self.scc_nonzero(r)

    SGPRS = 128                     # size of the %sgpr array without the parking slots

    def park_masks(self, blocks):
        """Run the lane-mask analysis and give every VGPR lane a mask is fetched back from a slot of its own,
        after the real registers in the %sgpr array (the array grows by as many)."""
        self.lane_masks, self.mask_writes = lane_masks.analyse(*blocks)
        slots = lane_masks.parked(blocks[0], self.lane_masks)
        self.parked = {slot: self.SGPRS + i for i, slot in enumerate(slots)}

    def mask_half(self, a):
        """For a 32-bit lane-mask operand in wave64, return 'lo' (lanes 0-31), 'hi' (lanes 32-63), or None."""
        if a["k"] == "special":
            t = a.get("t", "")
            if t in ("exec_lo", "vcc_lo"):
                return "lo"
            if t in ("exec_hi", "vcc_hi"):
                return "hi"
            return None
        if a["k"] == "sgpr" and a.get("n", 1) == 1:
            k = self.cur_masks.get(a["r"])
            if k == lane_masks.MASK_LO:
                return "lo"
            if k == lane_masks.MASK_HI:
                return "hi"
        return None

    def wave_equal(self, mine, equal, half=None):
        """s_cmp_eq / s_cmp_lg of a lane mask (with 0, or with another mask): the masks are equal when every lane's
        bit is, so the lanes vote. `mine` is this lane's result for the comparison asked (equal, or not equal).
        When `half` is 'lo' or 'hi' (32-bit compare on one half of a wave64 mask), restrict the vote to that half."""
        self.uses_vote = True
        if half is not None:
            lane = self.wave_lane()
            in_h = self.tmp()
            op = "OpULessThan" if half == "lo" else "OpUGreaterThanEqual"
            self.emit(f"{in_h} = {op} %bool {lane} {self.cu(32)}")
            eff = self.tmp()
            if equal:
                self.emit(f"{eff} = OpSelect %bool {in_h} {mine} %true")
            else:
                self.emit(f"{eff} = OpLogicalAnd %bool {in_h} {mine}")
            mine = eff
        r = self.tmp()
        self.g_vote(r, "All" if equal else "Any", mine)
        return r

    def holds_lane_mask(self, a):
        """Whether operand `a` certainly holds a lane mask here: EXEC, or VCC / an SGPR that lane_masks.py found
        written as a mask on every path. An SGPR pair filled from a buffer (a light mask, say) is data."""
        if a["k"] == "special":
            t = a.get("t", "")
            return t.startswith("exec") or (t in ("vcc", "vcc_lo") and "vcc" in self.cur_masks) or \
                (t == "vcc_hi" and self.wave64 and "vcc" in self.cur_masks)
        return a["k"] == "sgpr" and all(a["r"] + i in self.cur_masks for i in range(a.get("n", 1)))

    def wave_mask(self, a):
        """The whole wave's lane mask in operand `a` as real bits: (lanes 0-31, lanes 32-63). A register holds
        only this lane's bit (all ones or zero), so each lane puts its bit at its own position and the subgroup
        (= the wave) ORs them together. For the instructions that look at a mask as a whole: until 2026-11-21
        they saw the one lane, so the lane count of a wave was 64 or 0 and its first lane 0 or none."""
        self.uses_subgroup = self.uses_wave_bits = True
        special = a["k"] == "special" and a.get("t", "").startswith("exec")
        key = None
        if self.emulate:
            # The same value of the same register needs no second exchange: the lanes' bits are what they were.
            # Values are told apart by the SSA id last stored to the register in this block (x_tok, forgotten at
            # a label); a register not written yet has its value from the start of the block.
            if special:
                key = ("exec", self.x_gen, self.x_tok.get("%active", "start"))
            elif a["k"] == "sgpr":
                key = ("v", self.x_gen, self.x_tok.get(("sgpr", a["r"]), ("start", a["r"])))
            elif a["k"] == "special" and a["t"] in ("vcc", "vcc_lo", "vcc_hi"):
                key = ("v", self.x_gen, self.x_tok.get("%vcc", ("start", "vcc")))
            if key in self.x_mcache:
                return self.x_mcache[key]
        mine = self.tmp_bool_nz(self.exec_value() if special else self.mask_src(a))
        if self.emulate:
            res = self.x_round(bits=self.lane_bits(mine))[1]
            if key is not None:
                self.x_mcache[key] = res
            return res
        lane = self.wave_lane()
        low, sh, bit, b, lo_in, hi_in, lo, hi = (self.tmp() for _ in range(8))
        self.emit(f"{low} = OpULessThan %bool {lane} {self.cu(32)}")
        self.emit(f"{sh} = OpBitwiseAnd %uint {lane} {self.cu(31)}")
        self.emit(f"{bit} = OpShiftLeftLogical %uint {self.cu(1)} {sh}")
        self.emit(f"{b} = OpSelect %uint {mine} {bit} {self.cu(0)}")
        self.emit(f"{lo_in} = OpSelect %uint {low} {b} {self.cu(0)}")
        self.emit(f"{hi_in} = OpSelect %uint {low} {self.cu(0)} {b}")
        self.emit(f"{lo} = OpGroupNonUniformBitwiseOr %uint {self.cu(3)} Reduce {lo_in}")
        self.emit(f"{hi} = OpGroupNonUniformBitwiseOr %uint {self.cu(3)} Reduce {hi_in}")
        return lo, hi

    def quad_any(self, v):
        """This lane's value after whole-quad mode: all ones if any of the 4 lanes of its quad (4n .. 4n+3 of the
        wave) has `v` non-zero, else 0. A quad never crosses the 32/32 half of a wave64, so a 32-wide subgroup is
        exact. Without emulation the subgroup's quad ops do it (the invocations of a quad are 4 consecutive subgroup
        invocations; in a fragment shader the 2x2 quad, which on hardware is also 4 consecutive lanes; vertex,
        geometry and mesh shaders shuffle, as quad operations are only guaranteed in fragment and compute stages);
        emulated, the four lanes' values are read through the exchange buffer. Not handled: an absent invocation in
        a partial quad (a compute group whose size is not a multiple of 4) gives an undefined value natively."""
        b = self.tmp()
        self.emit(f"{b} = OpSelect %uint {self.tmp_bool_nz(v)} {self.cu(1)} {self.cu(0)}")
        if self.emulate:
            lane, base = self.wave_lane(), self.tmp()
            self.emit(f"{base} = OpBitwiseAnd %uint {lane} {self.cu(0xFFFFFFFC)}")
            read = self.x_round(data=b)[0]
            acc = None
            for k in range(4):
                i = self.tmp()
                self.emit(f"{i} = OpIAdd %uint {base} {self.cu(k)}")
                w = read(i)
                if acc is None:
                    acc = w
                else:
                    o = self.tmp()
                    self.emit(f"{o} = OpBitwiseOr %uint {acc} {w}")
                    acc = o
        elif self.stage in ("ps", "cs"):
            self.uses_subgroup = self.uses_quad = True
            acc = b
            for d in (0, 1):                      # horizontal, then vertical neighbour
                sw, o = self.tmp(), self.tmp()
                self.emit(f"{sw} = OpGroupNonUniformQuadSwap %uint {self.cu(3)} {acc} {self.cu(d)}")
                self.emit(f"{o} = OpBitwiseOr %uint {acc} {sw}")
                acc = o
        else:
            # Quad operations are only guaranteed in fragment and compute shaders (quadOperationsInAllStages):
            # vertex, geometry and mesh shaders shuffle from the four lanes of the quad instead.
            self.uses_subgroup = True
            lane, base = self.tmp(), self.tmp()
            self.sub_lane(lane)
            self.emit(f"{base} = OpBitwiseAnd %uint {lane} {self.cu(0xFFFFFFFC)}")
            acc = b
            for k in range(4):
                i, w, o = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{i} = OpIAdd %uint {base} {self.cu(k)}")
                self.g_shuffle(w, b, i)
                self.emit(f"{o} = OpBitwiseOr %uint {acc} {w}")
                acc = o
        r = self.tmp()
        self.emit(f"{r} = OpSelect %uint {self.tmp_bool_nz(acc)} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
        return r

    def wqm_word(self, w):
        """s_wqm on 32 bits of plain data: each group of 4 bits becomes 1111 if any of its bits is set."""
        acc = w
        for k in (1, 2, 3):
            x, o = self.tmp(), self.tmp()
            self.emit(f"{x} = OpShiftRightLogical %uint {w} {self.cu(k)}")
            self.emit(f"{o} = OpBitwiseOr %uint {acc} {x}")
            acc = o
        m, r = self.tmp(), self.tmp()
        self.emit(f"{m} = OpBitwiseAnd %uint {acc} {self.cu(0x11111111)}")
        self.emit(f"{r} = OpIMul %uint {m} {self.cu(15)}")
        return r

    def put_mask(self, d, v, half):
        """mask_dst, except that a 32-bit half of EXEC / VCC in wave64 only changes the lanes of that half."""
        if half and d["k"] == "special" and d.get("t") in ("exec_lo", "exec_hi", "vcc_lo", "vcc_hi"):
            h = 1 if d["t"].endswith("hi") else 0
            if d["t"].startswith("exec"):
                self.exec_half_write(h, v)
            else:
                lane, inh, cur, new = (self.tmp() for _ in range(4))
                self.emit(f"{lane} = OpCopyObject %uint {self.wave_lane()}")
                self.emit(f"{inh} = {'OpUGreaterThanEqual' if h else 'OpULessThan'} %bool {lane} {self.cu(32)}")
                self.emit(f"{cur} = OpLoad %uint %vcc")
                self.emit(f"{new} = OpSelect %uint {inh} {v} {cur}")
                self.emit(f"OpStore %vcc {new}")
            return
        self.mask_dst(d, v)

    def scc_any_half(self, v, half):
        """SCC = some lane of the wave (of one 32-lane half of a wave64 when `half`) has `v` non-zero."""
        mine = self.tmp_bool_nz(v)
        if half:
            lane, inh, both = (self.tmp() for _ in range(3))
            self.emit(f"{lane} = OpCopyObject %uint {self.wave_lane()}")
            self.emit(f"{inh} = {'OpUGreaterThanEqual' if half == 'hi' else 'OpULessThan'} %bool {lane} {self.cu(32)}")
            self.emit(f"{both} = OpLogicalAnd %bool {inh} {mine}")
            mine = both
        any_ = self.tmp()
        self.g_vote(any_, "Any", mine)
        self.uses_vote = True
        z = self.tmp()
        self.emit(f"{z} = OpSelect %uint {any_} {self.cu(1)} {self.cu(0)}")
        self.emit(f"OpStore %scc {z}")

    def wqm(self, op, a):
        """s_wqm_b32 / s_wqm_b64: D = every quad (4 consecutive bits) of S0 that has a bit set becomes 1111;
        SCC = D != 0. A lane mask is one bit per lane here (this lane's), so the quad's bits are gathered with
        quad_any and SCC is the wave-wide any. Anything else is data and is computed on the bits. A 32-bit form
        in wave64 (exec_lo / exec_hi, vcc_lo / vcc_hi, a half-mask SGPR) only changes, and only votes over, its own
        half: it is the half of the source, or of the destination when that is one of the halves."""
        half = None
        if self.wave64 and op.endswith("b32"):
            half = self.mask_half(a[1]) or self.mask_half(a[0])
        if self.holds_lane_mask(a[1]):
            r = self.quad_any(self.mask_src(a[1]))
            self.put_mask(a[0], r, half)
            self.scc_any_half(r, half)
            return
        src = a[1]
        if op.endswith("b64") and src["k"] == "sgpr" and src.get("n") == 2:
            words = [self.load_reg("sgpr", src["r"]), self.load_reg("sgpr", src["r"] + 1)]
        elif op.endswith("b64") and src["k"] == "imm":
            words = [self.cu(src["v"] & 0xFFFFFFFF), self.cu((src["v"] >> 32) & 0xFFFFFFFF)]
        else:
            words = [self.src_u(src)] + ([self.cu(0)] if op.endswith("b64") else [])
        res = [self.wqm_word(w) for w in words]
        d = a[0]
        if d["k"] == "sgpr":
            self.store_reg("sgpr", d["r"], res[0])
            if len(res) == 2:
                self.store_reg("sgpr", d["r"] + 1, res[1])
        else:
            # EXEC / VCC filled from data: this lane takes its own bit of the result.
            lane = self.wave_lane()
            sh, b, pick = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{sh} = OpBitwiseAnd %uint {lane} {self.cu(31)}")
            word = res[0]
            if len(res) == 2:
                hi, word = self.tmp(), self.tmp()
                self.emit(f"{hi} = OpUGreaterThanEqual %bool {lane} {self.cu(32)}")
                self.emit(f"{word} = OpSelect %uint {hi} {res[1]} {res[0]}")
            self.emit(f"{b} = OpBitFieldUExtract %uint {word} {sh} {self.cu(1)}")
            self.emit(f"{pick} = OpSelect %uint {self.tmp_bool_nz(b)} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
            self.put_mask(d, pick, half)
        nz = res[0]
        if len(res) == 2:
            nz = self.tmp()
            self.emit(f"{nz} = OpBitwiseOr %uint {res[0]} {res[1]}")
        self.scc_nonzero(nz)

    def lane_bits(self, b):
        """This lane's bool as its bit of a 64-lane mask: (lanes 0-31 word, lanes 32-63 word), one of them 0."""
        lane = self.wave_lane()
        low, sh, bit, v, lo_in, hi_in = (self.tmp() for _ in range(6))
        self.emit(f"{low} = OpULessThan %bool {lane} {self.cu(32)}")
        self.emit(f"{sh} = OpBitwiseAnd %uint {lane} {self.cu(31)}")
        self.emit(f"{bit} = OpShiftLeftLogical %uint {self.cu(1)} {sh}")
        self.emit(f"{v} = OpSelect %uint {b} {bit} {self.cu(0)}")
        self.emit(f"{lo_in} = OpSelect %uint {low} {v} {self.cu(0)}")
        self.emit(f"{hi_in} = OpSelect %uint {low} {self.cu(0)} {v}")
        return lo_in, hi_in

    def mask_src(self, a):
        """A lane mask operand (sgpr pair, vcc, exec, imm) as one uint for this lane."""
        if a["k"] == "sgpr":
            return self.load_reg("sgpr", a["r"])
        if a["k"] == "special" and a["t"] in ("vcc", "vcc_lo", "vcc_hi"):
            v = self.tmp()
            self.emit(f"{v} = OpLoad %uint %vcc")
            return v
        if a["k"] == "special" and a["t"] in ("exec", "exec_lo", "exec_hi"):
            return self.exec_value()
        if a["k"] == "imm":
            return self.cu(0xFFFFFFFF if a["v"] & 1 else 0)
        raise Unsupported("mask operand")

    def mask_dst(self, d, v):
        if d["k"] == "special" and d["t"] in ("exec", "exec_lo"):
            self.set_exec(v)
        elif d["k"] == "sgpr":
            self.store_reg("sgpr", d["r"], v)
            if d.get("n") == 2:
                self.store_reg("sgpr", d["r"] + 1, v)
        elif d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
            self.emit(f"OpStore %vcc {v}")
        else:
            raise Unsupported("mask destination")

    def clamp_mods(self, v, mods):
        # Output modifier (mul:2 / mul:4 / div:2) is applied before clamp, as on hardware.
        for m in mods:
            if m in ("mul:2", "mul:4", "div:2"):
                k = {"mul:2": "2", "mul:4": "4", "div:2": "0.5"}[m]
                c = self.fconst(float(k))
                g = self.tmp()
                self.emit(f"{g} = OpFMul %float {v} {c}")
                v = g
        if "clamp" in mods:
            g = self.tmp()
            self.emit(f"{g} = OpExtInst %float %glsl FClamp {v} %f0 %f1")
            return g
        return v

    def clamp_mods_f64(self, v, mods):
        """clamp_mods for a %double result (VOP3 f64 arithmetic): omod scales first, then clamp to [0, 1]. As
        for f32 the clamp is GLSL.std.450 FClamp, so a NaN result and the sign of a -0 result are left to the
        driver (the hardware, with DX10_CLAMP set, gives +0 for NaN)."""
        for m in mods:
            if m in ("mul:2", "mul:4", "div:2"):
                g = self.tmp()
                self.emit(f"{g} = OpFMul %double {v} {self.dconst({'mul:2': 2.0, 'mul:4': 4.0, 'div:2': 0.5}[m])}")
                v = g
        if "clamp" in mods:
            g = self.tmp()
            self.emit(f"{g} = OpExtInst %double %glsl FClamp {v} {self.dconst(0.0)} {self.dconst(1.0)}")
            return g
        return v

    def fconst(self, x):
        bits = struct.unpack("<I", struct.pack("<f", x))[0]
        return self.tof(self.cu(bits))

    def dconst(self, x):
        """A %double value (needs uses_f64): the two words of x, bitcast."""
        bits = struct.unpack("<Q", struct.pack("<d", x))[0]
        v2, d = self.tmp(), self.tmp()
        self.emit(f"{v2} = OpCompositeConstruct %v2uint_d {self.cu(bits & 0xFFFFFFFF)} {self.cu(bits >> 32)}")
        self.emit(f"{d} = OpBitcast %double {v2}")
        return d

    # ---- instructions ----
    SDWA_SEL = {"BYTE_0": (0, 8), "BYTE_1": (8, 8), "BYTE_2": (16, 8), "BYTE_3": (24, 8),
                "WORD_0": (0, 16), "WORD_1": (16, 16), "DWORD": (0, 32)}

    def insn(self, ins):
        """SDWA (sub-dword addressing): each source is narrowed to src<i>_sel (zero-extended, or sign-extended
        under sext()) before the op and its neg/abs, and a VGPR result is placed at dst_sel with dst_unused
        deciding the other bits (PAD = zero, SEXT = sign-extend, PRESERVE = keep the old bits). VOPC has no
        dst_sel; v_cmpx (GFX10) has no written destination operand."""
        self.cur_masks = self.lane_masks.get(id(ins), {})
        self.write_all_lanes = id(ins) in self.mask_writes
        if self.emulate:
            self.x_memory_order(base_op(ins["op"]))
        if not ins["op"].endswith("_sdwa"):
            return self._insn(ins)
        op, mods = base_op(ins["op"]), ins["mods"]
        kv = dict(m.split(":", 1) for m in mods if m.split(":", 1)[0] in ("src0_sel", "src1_sel", "dst_sel", "dst_unused"))
        args = [dict(x) for x in ins["args"]]
        first = 0 if op.startswith("v_cmpx") else 1
        for i, key in enumerate(("src0_sel", "src1_sel")):
            j = first + i
            if j < len(args) and (kv.get(key, "DWORD") != "DWORD" or args[j].get("sext")):
                args[j]["sdwa_sel"] = kv.get(key, "DWORD")
        if not op.startswith("v_cmp") and args and (kv.get("dst_sel", "DWORD") != "DWORD"):
            args[0]["sdwa_dst"] = (kv["dst_sel"], kv.get("dst_unused", "UNUSED_PAD"))
        return self._insn({**ins, "args": args})

    def sdwa_src(self, v, a):
        off, width = self.SDWA_SEL[a["sdwa_sel"]]
        if width == 32:
            return v
        r = self.tmp()
        f = "OpBitFieldSExtract" if a.get("sext") else "OpBitFieldUExtract"
        self.emit(f"{r} = {f} %uint {v} {self.cu(off)} {self.cu(width)}")
        return r

    def sdwa_dst(self, d, val):
        sel, unused = d["sdwa_dst"]
        off, width = self.SDWA_SEL[sel]
        if width == 32:
            return val
        r = self.tmp()
        if unused == "UNUSED_PRESERVE":
            self.emit(f"{r} = OpBitFieldInsert %uint {self.load_reg('vgpr', d['r'])} {val} {self.cu(off)} {self.cu(width)}")
            return r
        lo = self.tmp()
        ext = "OpBitFieldSExtract" if unused == "UNUSED_SEXT" else "OpBitFieldUExtract"
        self.emit(f"{lo} = {ext} %uint {val} {self.cu(0)} {self.cu(width)}")
        # SEXT: the sign-extended field shifted up leaves sign copies above it; bits below stay zero.
        self.emit(f"{r} = OpShiftLeftLogical %uint {lo} {self.cu(off)}")
        return r

    def _insn(self, ins):
        op, a, mods = base_op(ins["op"]), ins["args"], ins["mods"]
        if op in NOOPS or op.startswith("s_waitcnt"):
            return
        if op == "s_sendmsg" and a and self.stage == "gs":
            t = a[0].get("t", "")
            if "GS_OP_EMIT" in t:
                self.gs_emit()
            elif "GS_OP_CUT" in t:
                self.gs_when_active("OpEndPrimitive")
            elif "MSG_GS_DONE" not in t:
                raise Unsupported("gs sendmsg " + t)
            return
        if op == "s_sendmsg" and a and "MSG_GS_ALLOC_REQ" in a[0].get("t", "") and self.stage == "ms":
            m = self.tmp()
            self.emit(f"{m} = OpLoad %uint %m0")
            self.emit(f"OpStore %ms_counts {m}")
            self.ms_alloc = True
            return
        if op == "s_sendmsg" and a and "MSG_ORDERED_PS_DONE" in a[0].get("t", ""):
            # End of the POPS ordered section. The fragment-shader interlock already ends at shader exit (a longer
            # critical section, same ordering), so nothing to do here.
            return
        if op in ("s_setreg_imm32_b32", "s_setreg_b32", "s_denorm_mode", "s_round_mode"):
            # MODE register (rounding/denormals). SPIR-V float controls aren't modelled yet: Vulkan defaults are
            # used, which can differ in denormal handling. Recorded here so the gap is visible.
            return
        if op == "s_mov_b32" and a[0]["k"] == "special" and a[0]["t"] in ("exec_lo", "exec_hi") and \
                (a[0]["t"] == "exec_hi" or a[1]["k"] in ("imm", "fimm")):
            src = a[1]
            const = (src["v"] & 0xFFFFFFFF) if src["k"] == "imm" else None
            if src["k"] == "fimm":
                const = struct.unpack("<I", struct.pack("<f", src["v"]))[0]
            self.exec_half_write(0 if a[0]["t"] == "exec_lo" else 1, None if const is not None else self.src_u(src),
                                 const)
        elif op == "v_mov_b32" or op == "s_mov_b32":
            self.dst(a[0], self.src_u(a[1]))
            if op == "s_mov_b32" and a[0]["k"] == "sgpr" and a[1]["k"] == "imm" and self.desc_cache_ok():
                self.sgpr_imm[a[0]["r"]] = a[1]["v"] & 0xFFFFFFFF      # a literal S# word, see gather_addr_modes
        elif op == "s_mov_b64":
            self.mov64(a[0], a[1])
        elif op in ("v_max3_f32", "v_min3_f32"):
            x, y, z = (self.src_f(t) for t in a[1:4])
            f = "FMax" if "max" in op else "FMin"
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpExtInst %float %glsl {f} {x} {y}")
            self.emit(f"{r} = OpExtInst %float %glsl {f} {m} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_and_or_b32":
            x, y, z = (self.src_u(t) for t in a[1:4])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {x} {y}")
            self.emit(f"{r} = OpBitwiseOr %uint {m} {z}")
            self.dst(a[0], r)
        elif op == "s_brev_b64":
            # D = the 64 bits of S0 in reverse order: D.lo = reverse(S0.hi), D.hi = reverse(S0.lo). SCC is not
            # written. The pair is 64-bit data (as for s_lshl_b64); a lane-mask source stays Unsupported (#210).
            d, x = a[0], a[1]
            if d["k"] != "sgpr" or d.get("n") != 2 or self.holds_lane_mask(x):
                raise Unsupported(op + " operands")
            if x["k"] == "sgpr" and x.get("n") == 2:
                lo, hi = self.load_reg("sgpr", x["r"]), self.load_reg("sgpr", x["r"] + 1)
            elif x["k"] == "imm":
                v = x["v"] & 0xFFFFFFFFFFFFFFFF          # 64-bit inline/literal constant: sign-extended
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " source")
            rlo, rhi = self.tmp(), self.tmp()
            self.emit(f"{rlo} = OpBitReverse %uint {hi}")
            self.emit(f"{rhi} = OpBitReverse %uint {lo}")
            self.store_reg("sgpr", d["r"], rlo)
            self.store_reg("sgpr", d["r"] + 1, rhi)
        elif op == "s_brev_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpBitReverse %uint {x}")
            self.dst(a[0], r)
        elif op == "v_cvt_f16_f32":
            # Low 16 bits get the f16; on GFX10 the high half is kept (partial write: Mesa ACO instr_is_16bit lists
            # v_cvt_f16_f32 as 16-bit for gfx_level >= GFX10).
            x = self.src_f(a[1])
            self.write_lo16(a[0], self.f16_bits(x))
        elif op == "s_mulk_i32":
            v = a[1]["v"] & 0xFFFF
            k = v - 0x10000 if v & 0x8000 else v
            x = self.src_u(a[0])
            r = self.tmp()
            self.emit(f"{r} = OpIMul %uint {x} {self.cu(k)}")
            self.dst(a[0], r)
        elif op == "v_readfirstlane_b32":
            # sD = vS of the first lane that is on in EXEC (lane 0 if none is): one value for the whole wave, so
            # a ballot of the active lanes, its lowest bit, and a shuffle from that lane. Until 2026-11-21 every
            # lane took its own value. That is the same thing only when all lanes hold the value, and after a
            # kill the dead lanes don't: where the result is an address (a loop that picks a descriptor per group
            # of lanes), they loaded from wherever their stale register pointed.
            if a[1]["k"] == "vgpr" and ("v", a[1]["r"]) in self.cur_masks:
                # A lane mask that went through the vector unit (three SGPRs ORed with v_or3_b32, say). The
                # hardware reads the first lane's copy of the whole mask; here the lane's own copy is its bit.
                self.dst(a[0], self.src_u(a[1]))
                return
            if self.emulate:
                # The wave's active lanes as two mask words and every lane's value, in one round.
                v, act = self.src_u(a[1]), self.active_b()
                lo_in, hi_in = self.lane_bits(act)
                read, (lo, hi) = self.x_round(data=v, bits=(lo_in, hi_in))
                lz, hz, l, h, h32, fh, first = (self.tmp() for _ in range(7))
                self.emit(f"{lz} = OpINotEqual %bool {lo} {self.cu(0)}")
                self.emit(f"{hz} = OpINotEqual %bool {hi} {self.cu(0)}")
                self.emit(f"{l} = OpExtInst %uint %glsl FindILsb {lo}")
                self.emit(f"{h} = OpExtInst %uint %glsl FindILsb {hi}")
                self.emit(f"{h32} = OpIAdd %uint {h} {self.cu(32)}")
                self.emit(f"{fh} = OpSelect %uint {hz} {h32} {self.cu(0)}")
                self.emit(f"{first} = OpSelect %uint {lz} {l} {fh}")
                self.dst(a[0], read(first))
                return
            self.uses_subgroup = self.uses_ballot = self.uses_vote = True
            v, act = self.src_u(a[1]), self.active_b()
            bal, lsb, any_, first, r = (self.tmp() for _ in range(5))
            self.emit(f"{bal} = OpGroupNonUniformBallot %v4uint_bal {self.cu(3)} {act}")
            self.emit(f"{lsb} = OpGroupNonUniformBallotFindLSB %uint {self.cu(3)} {bal}")
            self.emit(f"{any_} = OpGroupNonUniformAny %bool {self.cu(3)} {act}")
            self.emit(f"{first} = OpSelect %uint {any_} {lsb} {self.cu(0)}")
            self.emit(f"{r} = OpGroupNonUniformShuffle %uint {self.cu(3)} {v} {first}")
            self.dst(a[0], r)
        elif op in ("image_sample_o", "image_sample_lz_o", "image_sample_l_o", "image_sample_b_o",
                    "image_sample_c_lz_o") or                 op in ("image_sample", "image_sample_lz", "image_sample_l", "image_sample_b", "image_sample_c_lz",
                    "image_sample_d", "image_sample_d_cl", "image_load", "image_gather4_lz", "image_gather4_l",
                    "image_gather4_lz_o", "image_gather4_l_o"):
            self.image_op(op, a, mods)
        elif op in ("v_sin_f16", "v_cos_f16"):
            u = self.src_h(a[1])
            v2, f, t, r, w2, pk, lo = (self.tmp() for _ in range(7))
            self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
            self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
            f = self.modify_f(f, a[1])
            self.emit(f"{t} = OpFMul %float {f} {self.fconst(6.283185307179586)}")
            self.emit(f"{r} = OpExtInst %float %glsl {'Sin' if 'sin' in op else 'Cos'} {t}")
            r = self.clamp_mods(r, mods)
            self.write_lo16(a[0], self.f16_bits(r))
        elif op in ("v_pk_lshlrev_b16", "v_pk_lshrrev_b16", "v_pk_ashrrev_i16"):
            # Packed 16-bit shifts (VOP3P): result half h = src1 half (op_sel / op_sel_hi) shifted by src0 half & 15.
            def bits(name, default):
                m = next((m for m in mods if m.startswith(name + ":")), None)
                return [int(v) for v in m.split("[")[1].rstrip("]").split(",")] if m else default
            sel, selhi = bits("op_sel", [0, 0]), bits("op_sel_hi", [1, 1])
            sh_src, val_src = self.src_u(a[1]), self.src_u(a[2])
            halves = []
            for h, sl in ((0, sel), (1, selhi)):
                amt, val, am, r16 = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{amt} = OpBitFieldUExtract %uint {sh_src} {self.cu(16 * sl[0])} {self.cu(4)}")
                ext = "OpBitFieldSExtract" if op.endswith("i16") else "OpBitFieldUExtract"
                self.emit(f"{val} = {ext} %uint {val_src} {self.cu(16 * sl[1])} {self.cu(16)}")
                k = {"v_pk_lshlrev_b16": "OpShiftLeftLogical", "v_pk_lshrrev_b16": "OpShiftRightLogical",
                     "v_pk_ashrrev_i16": "OpShiftRightArithmetic"}[op]
                self.emit(f"{am} = {k} %uint {val} {amt}")
                self.emit(f"{r16} = OpBitwiseAnd %uint {am} {self.cu(0xFFFF)}")
                halves.append(r16)
            hi, r = self.tmp(), self.tmp()
            self.emit(f"{hi} = OpShiftLeftLogical %uint {halves[1]} {self.cu(16)}")
            self.emit(f"{r} = OpBitwiseOr %uint {halves[0]} {hi}")
            self.dst(a[0], r)
        elif op in ("v_sin_f32", "v_cos_f32"):
            # RDNA takes the angle in turns: sin(x * 2*pi).
            x = self.src_f(a[1])
            t, r = self.tmp(), self.tmp()
            self.emit(f"{t} = OpFMul %float {x} {self.fconst(6.283185307179586)}")
            self.emit(f"{r} = OpExtInst %float %glsl {'Sin' if 'sin' in op else 'Cos'} {t}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_rndne_f32":
            x = self.src_f(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl RoundEven {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_madak_f32", "v_madmk_f32"):
            # madak D = S0 * S1 + K, printed `vdst, src0, vsrc1, K`; madmk D = S0 * K + S1, printed
            # `vdst, src0, K, vsrc1`. In print order both are a[1] * a[2] + a[3]. Until 2026-11-21 madmk read
            # a[3] as the factor, so it computed S0 * S1 + K.
            x, y, z = (self.src_f(t) for t in a[1:4])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{r} = OpFAdd %float {m} {z}")
            self.dst(a[0], self.tou(r))
        elif op in ("v_bfi_b32", "v_lshl_or_b32", "v_lshl_add_u32", "v_add_lshl_u32", "v_or3_b32", "v_xor3_b32",
                    "v_add3_u32", "v_and_or_b32_"):
            x, y, z = (self.src_u(t) for t in a[1:4])
            r = self.tmp()
            if op == "v_bfi_b32":
                p, n, q = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{p} = OpBitwiseAnd %uint {x} {y}")
                self.emit(f"{n} = OpNot %uint {x}")
                self.emit(f"{q} = OpBitwiseAnd %uint {n} {z}")
                self.emit(f"{r} = OpBitwiseOr %uint {p} {q}")
            elif op in ("v_lshl_or_b32", "v_lshl_add_u32"):
                s, m = self.tmp(), self.tmp()
                self.emit(f"{m} = OpBitwiseAnd %uint {y} {self.cu(31)}")
                self.emit(f"{s} = OpShiftLeftLogical %uint {x} {m}")
                self.emit(f"{r} = {'OpBitwiseOr' if op == 'v_lshl_or_b32' else 'OpIAdd'} %uint {s} {z}")
            elif op == "v_add_lshl_u32":
                s, m = self.tmp(), self.tmp()
                self.emit(f"{s} = OpIAdd %uint {x} {y}")
                self.emit(f"{m} = OpBitwiseAnd %uint {z} {self.cu(31)}")
                self.emit(f"{r} = OpShiftLeftLogical %uint {s} {m}")
            else:
                k = {"v_or3_b32": "OpBitwiseOr", "v_xor3_b32": "OpBitwiseXor", "v_add3_u32": "OpIAdd"}[op]
                s = self.tmp()
                self.emit(f"{s} = {k} %uint {x} {y}")
                self.emit(f"{r} = {k} %uint {s} {z}")
            self.dst(a[0], r)
        elif op == "v_pack_b32_f16":
            # dst = f16(src0) | f16(src1) << 16, each taken from the low half (op_sel picks high halves).
            sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
            sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")]
            halves = []
            for i, src in enumerate(a[1:3]):
                u = self.src_h(src)
                if sb[i]:
                    s = self.tmp()
                    self.emit(f"{s} = OpShiftRightLogical %uint {u} {self.cu(16)}")
                    u = s
                m = self.tmp()
                self.emit(f"{m} = OpBitwiseAnd %uint {u} {self.cu(0xFFFF)}")
                halves.append(m)
            if any(src.get("neg") or src.get("abs") for src in a[1:3]):
                raise Unsupported("v_pack_b32_f16 modifiers")
            hi, r = self.tmp(), self.tmp()
            self.emit(f"{hi} = OpShiftLeftLogical %uint {halves[1]} {self.cu(16)}")
            self.emit(f"{r} = OpBitwiseOr %uint {halves[0]} {hi}")
            self.dst(a[0], r)
        elif op in ("v_cmp_class_f32", "v_cmpx_class_f32"):
            self.cmp_class(a, exec_only=op == "v_cmpx_class_f32")
        elif op in ("v_ffbl_b32", "v_ffbh_u32", "s_flbit_i32_b32") or \
                (op == "s_ff1_i32_b32" and not self.holds_lane_mask(a[1])):       # of a lane mask: further down
            x = self.src_u(a[1])
            r = self.tmp()
            if op in ("v_ffbl_b32", "s_ff1_i32_b32"):
                # Index of lowest set bit, -1 (0xFFFFFFFF) if none: FindILsb has the same convention.
                ri = self.tmp()
                self.emit(f"{ri} = OpExtInst %int %glsl FindILsb {self.as_int(x)}")
                self.emit(f"{r} = OpBitcast %uint {ri}")
            else:
                # Leading-zero count from the MSB side, -1 if zero: 31 - FindUMsb(x), with FindUMsb(0) = -1.
                m, z, d = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{m} = OpExtInst %int %glsl FindUMsb {x}")
                mu = self.tmp()
                self.emit(f"{mu} = OpBitcast %uint {m}")
                self.emit(f"{d} = OpISub %uint {self.cu(31)} {mu}")
                self.emit(f"{z} = OpIEqual %bool {x} {self.cu(0)}")
                self.emit(f"{r} = OpSelect %uint {z} {self.cu(0xFFFFFFFF)} {d}")
            self.dst(a[0], r)
        elif op in ("image_get_resinfo", "image_load_mip", "image_store", "image_atomic_cmpswap") or op in self.IMG_ATOMICS:
            self.image_misc(op, a, mods)
        elif op in ("v_cubema_f32", "v_cubeid_f32", "v_cubesc_f32", "v_cubetc_f32"):
            # Cube face selection (standard table): major axis Z if |z|>=|x| and |z|>=|y|, else Y if |y|>=|x|,
            # else X.   face: +X0 -X1 +Y2 -Y3 +Z4 -Z5
            #   +X: sc=-z tc=-y   -X: sc=+z tc=-y   +Y: sc=+x tc=+z   -Y: sc=+x tc=-z   +Z: sc=+x tc=-y   -Z: sc=-x tc=-y
            #   cubema = 2 * (signed major-axis coordinate)
            x, y, z = (self.src_f(t) for t in a[1:4])
            ax, ay, az = (self.tmp() for _ in range(3))
            for t, s in ((ax, x), (ay, y), (az, z)):
                self.emit(f"{t} = OpExtInst %float %glsl FAbs {s}")
            zx, zy, zmaj, yx, ymaj = (self.tmp() for _ in range(5))
            self.emit(f"{zx} = OpFOrdGreaterThanEqual %bool {az} {ax}")
            self.emit(f"{zy} = OpFOrdGreaterThanEqual %bool {az} {ay}")
            self.emit(f"{zmaj} = OpLogicalAnd %bool {zx} {zy}")
            self.emit(f"{yx} = OpFOrdGreaterThanEqual %bool {ay} {ax}")
            nz = self.tmp()
            self.emit(f"{nz} = OpLogicalNot %bool {zmaj}")
            self.emit(f"{ymaj} = OpLogicalAnd %bool {nz} {yx}")

            def neg(v):
                n = self.tmp()
                self.emit(f"{n} = OpFNegate %float {v}")
                return n

            def lt0(v):
                b = self.tmp()
                self.emit(f"{b} = OpFOrdLessThan %bool {v} %f0")
                return b

            def pick(zv, yv, xv):
                t1, t2 = self.tmp(), self.tmp()
                self.emit(f"{t1} = OpSelect %float {ymaj} {yv} {xv}")
                self.emit(f"{t2} = OpSelect %float {zmaj} {zv} {t1}")
                return t2

            def sel(c, t, f):
                r = self.tmp()
                self.emit(f"{r} = OpSelect %float {c} {t} {f}")
                return r
            xn, yn, zn = lt0(x), lt0(y), lt0(z)
            if op == "v_cubema_f32":
                m = pick(z, y, x)
                r = self.tmp()
                self.emit(f"{r} = OpFMul %float {m} {self.fconst(2.0)}")
            elif op == "v_cubeid_f32":
                r = pick(sel(zn, self.fconst(5.0), self.fconst(4.0)), sel(yn, self.fconst(3.0), self.fconst(2.0)),
                         sel(xn, self.fconst(1.0), self.fconst(0.0)))
            elif op == "v_cubesc_f32":
                r = pick(sel(zn, neg(x), x), x, sel(xn, z, neg(z)))
            else:
                r = pick(neg(y), sel(yn, neg(z), z), neg(y))
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_mbcnt_lo_u32_b32", "v_mbcnt_hi_u32_b32"):
            # Count of set bits in the mask for lanes below this one (lo: lanes 0-31, hi: lanes 32-63), plus src1.
            self.uses_subgroup = True
            if self.holds_lane_mask(a[1]):              # a real mask: the lanes below this one that are set in it
                mask = self.wave_mask(a[1])[0 if op.endswith("lo_u32_b32") else 1]
                acc = self.src_u(a[2])
            else:
                mask, acc = self.src_u(a[1]), self.src_u(a[2])
            lane, rel, below, inrange, one, m1, cnt, r = (self.tmp() for _ in range(8))
            wl = self.wave_lane()
            self.emit(f"{lane} = OpCopyObject %uint {wl}")
            if op.endswith("lo_u32_b32"):
                self.emit(f"{rel} = OpCopyObject %uint {lane}")
            else:
                self.emit(f"{rel} = OpISub %uint {lane} {self.cu(32)}")
            # bits below `rel`, clamped to [0, 32]
            self.emit(f"{inrange} = OpULessThan %bool {rel} {self.cu(32)}")
            self.emit(f"{one} = OpShiftLeftLogical %uint {self.cu(1)} {rel}")
            self.emit(f"{m1} = OpISub %uint {one} {self.cu(1)}")
            big = self.tmp()
            # lo: lanes >= 32 count all 32 bits; hi: lanes < 32 count none (rel wraps above 32).
            self.emit(f"{big} = OpSelect %uint {inrange} {m1} "
                      f"{self.cu(0xFFFFFFFF) if op.endswith('lo_u32_b32') else self.cu(0)}")
            if op.endswith("hi_u32_b32"):
                lt32, fixed = self.tmp(), self.tmp()
                self.emit(f"{lt32} = OpULessThan %bool {lane} {self.cu(32)}")
                self.emit(f"{fixed} = OpSelect %uint {lt32} {self.cu(0)} {big}")
                big = fixed
            self.emit(f"{below} = OpBitwiseAnd %uint {mask} {big}")
            self.emit(f"{cnt} = OpBitCount %uint {below}")
            self.emit(f"{r} = OpIAdd %uint {cnt} {acc}")
            self.dst(a[0], r)
        elif op == "v_bfrev_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpBitReverse %uint {x}")
            self.dst(a[0], r)
        elif op == "v_bfm_b32":
            w, o = self.src_u(a[1]), self.src_u(a[2])
            wm, om, one, m1, r = (self.tmp() for _ in range(5))
            self.emit(f"{wm} = OpBitwiseAnd %uint {w} {self.cu(31)}")
            self.emit(f"{om} = OpBitwiseAnd %uint {o} {self.cu(31)}")
            self.emit(f"{one} = OpShiftLeftLogical %uint {self.cu(1)} {wm}")
            self.emit(f"{m1} = OpISub %uint {one} {self.cu(1)}")
            self.emit(f"{r} = OpShiftLeftLogical %uint {m1} {om}")
            self.dst(a[0], r)
        elif op in ("s_bcnt1_i32_b64", "s_bcnt1_i32_b32", "s_bcnt0_i32_b64", "s_bcnt0_i32_b32"):
            if self.holds_lane_mask(a[1]):
                lo, hi = self.wave_mask(a[1])
                if op.endswith("b32") and self.wave64:
                    half = self.mask_half(a[1])
                    if half is not None:
                        lo, hi = (hi if half == "hi" else lo), self.cu(0)
            elif op.endswith("b64") and a[1]["k"] == "sgpr" and a[1].get("n") == 2:
                lo, hi = self.load_reg("sgpr", a[1]["r"]), self.load_reg("sgpr", a[1]["r"] + 1)
            else:
                lo, hi = (self.mask_src(a[1]), self.cu(0)) if op.endswith("b64") else (self.src_u(a[1]), self.cu(0))
            if "bcnt0" in op:
                nlo = self.tmp()
                self.emit(f"{nlo} = OpNot %uint {lo}")
                lo = nlo
                if op.endswith("b64"):
                    nhi = self.tmp()
                    self.emit(f"{nhi} = OpNot %uint {hi}")
                    hi = nhi
            c1, c2, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{c1} = OpBitCount %uint {lo}")
            if op.endswith("b64") or "bcnt1" in op:
                # s_bcnt1_i32_b32 counts the high word too: for a lane mask of a wave32 shader running in a 64-lane
                # subgroup (graphics stages can't request 32) the other lanes' bits are there; otherwise hi is 0.
                # s_bcnt0_i32_b32 counts zeros of the low word only (~0 in the high word would add 32).
                self.emit(f"{c2} = OpBitCount %uint {hi}")
                self.emit(f"{r} = OpIAdd %uint {c1} {c2}")
            else:
                r = c1
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op in ("v_add_nc_i32", "v_sub_nc_i32") or \
                (op in ("v_add_nc_u32", "v_sub_nc_u32", "v_subrev_nc_u32") and "clamp" in mods):
            # 32-bit add/sub without carry out. The VOP3 clamp bit saturates an integer result (GFX9+: LLVM selects
            # uaddsat/usubsat to v_add/sub_nc_u32 clamp and saddsat/ssubsat to v_add/sub_nc_i32 clamp): unsigned to
            # [0, 0xFFFFFFFF], signed to [0x80000000, 0x7FFFFFFF]. Without clamp the result wraps.
            x, y = self.src_u(a[1]), self.src_u(a[2])
            if op == "v_subrev_nc_u32":
                x, y = y, x
            sub = "_sub" in op
            r = self.tmp()
            if op.endswith("_u32"):
                s_, w, c = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{s_} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {x} {y}")
                self.emit(f"{w} = OpCompositeExtract %uint {s_} 0")
                self.emit(f"{c} = OpCompositeExtract %uint {s_} 1")
                sat = self.cu(0) if sub else self.cu(0xFFFFFFFF)
                self.emit(f"{r} = OpSelect %uint {self.tmp_bool_nz(c)} {sat} {w}")
            else:
                w = self.tmp()
                self.emit(f"{w} = {'OpISub' if sub else 'OpIAdd'} %uint {x} {y}")
                if "clamp" in mods:
                    # Signed overflow: add when x and y share a sign that w lacks; sub when x and y differ in sign
                    # and w differs from x. The saturated value follows x's sign.
                    t1, t2, ov, neg, lim = (self.tmp() for _ in range(5))
                    if sub:
                        self.emit(f"{t1} = OpBitwiseXor %uint {x} {y}")
                    else:
                        self.emit(f"{t1} = OpBitwiseXor %uint {y} {w}")
                    self.emit(f"{t2} = OpBitwiseXor %uint {x} {w}")
                    ovm = self.tmp()
                    self.emit(f"{ovm} = OpBitwiseAnd %uint {t1} {t2}")
                    self.emit(f"{ov} = OpSLessThan %bool {self.as_int(ovm)} {self.as_int(self.cu(0))}")
                    self.emit(f"{neg} = OpSLessThan %bool {self.as_int(x)} {self.as_int(self.cu(0))}")
                    self.emit(f"{lim} = OpSelect %uint {neg} {self.cu(0x80000000)} {self.cu(0x7FFFFFFF)}")
                    self.emit(f"{r} = OpSelect %uint {ov} {lim} {w}")
                else:
                    r = w
            self.dst(a[0], r)
        elif op == "v_subrev_nc_u32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            self.emit(f"{r} = OpISub %uint {y} {x}")
            self.dst(a[0], r)
        elif op in ("s_flbit_i32_b64", "s_flbit_i32_i64"):
            # Leading-bit count of a 64-bit value, from bit 63 down: b64 counts the zeros before the first 1; i64
            # counts the bits before the first one that differs from the sign bit (b64 of ~S0 when S0 < 0). -1 when
            # there is none (0, and for i64 also -1). SCC is not written. The pair is 64-bit data (as for
            # s_lshl_b64); a lane-mask source stays Unsupported (#210).
            x = a[1]
            if self.holds_lane_mask(x):
                raise Unsupported(op + " of a lane mask")
            if x["k"] == "sgpr" and x.get("n") == 2:
                lo, hi = self.load_reg("sgpr", x["r"]), self.load_reg("sgpr", x["r"] + 1)
            elif x["k"] == "imm":
                v = x["v"] & 0xFFFFFFFFFFFFFFFF
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " source")
            if op == "s_flbit_i32_i64":
                neg, nlo, nhi, slo, shi = (self.tmp() for _ in range(5))
                self.emit(f"{neg} = OpSLessThan %bool {self.as_int(hi)} {self.as_int(self.cu(0))}")
                self.emit(f"{nlo} = OpNot %uint {lo}")
                self.emit(f"{nhi} = OpNot %uint {hi}")
                self.emit(f"{slo} = OpSelect %uint {neg} {nlo} {lo}")
                self.emit(f"{shi} = OpSelect %uint {neg} {nhi} {hi}")
                lo, hi = slo, shi
            mh, ml, mhu, mlu, ch, cl, hz, lz, r0, r = (self.tmp() for _ in range(10))
            self.emit(f"{mh} = OpExtInst %int %glsl FindUMsb {hi}")
            self.emit(f"{ml} = OpExtInst %int %glsl FindUMsb {lo}")
            self.emit(f"{mhu} = OpBitcast %uint {mh}")
            self.emit(f"{mlu} = OpBitcast %uint {ml}")
            self.emit(f"{ch} = OpISub %uint {self.cu(31)} {mhu}")          # zeros above the top 1 of the high half
            self.emit(f"{cl} = OpISub %uint {self.cu(63)} {mlu}")          # 32 + zeros above the top 1 of the low half
            self.emit(f"{hz} = OpIEqual %bool {hi} {self.cu(0)}")
            self.emit(f"{lz} = OpIEqual %bool {lo} {self.cu(0)}")
            self.emit(f"{r0} = OpSelect %uint {lz} {self.cu(0xFFFFFFFF)} {cl}")
            self.emit(f"{r} = OpSelect %uint {hz} {r0} {ch}")
            self.dst(a[0], r)
        elif op in ("s_ff1_i32_b64", "s_ff1_i32_b32", "s_ff0_i32_b64", "s_ff0_i32_b32"):
            # Lowest set (ff1) or zero (ff0) bit of a mask, -1 if none. For a lane mask that is the wave's first lane.
            if self.holds_lane_mask(a[1]):
                lo, hi = self.wave_mask(a[1])
                if op.endswith("b32") and self.wave64:
                    half = self.mask_half(a[1])
                    if half is not None:
                        lo, hi = (hi if half == "hi" else lo), self.cu(0)
            elif op.endswith("b32") or a[1]["k"] == "imm":
                # A b32 source is a plain scalar value, not a lane mask; for b64 an immediate is the low word.
                lo, hi = self.src_u(a[1]), self.cu(0)
            elif a[1]["k"] != "sgpr" or a[1].get("n") != 2:
                lo, hi = self.mask_src(a[1]), self.cu(0)
            else:
                lo, hi = self.load_reg("sgpr", a[1]["r"]), self.load_reg("sgpr", a[1]["r"] + 1)
            if "ff0" in op:
                nlo = self.tmp()
                self.emit(f"{nlo} = OpNot %uint {lo}")
                lo = nlo
                if op.endswith("b64"):
                    nhi = self.tmp()
                    self.emit(f"{nhi} = OpNot %uint {hi}")
                    hi = nhi
            l, h, hp, lz, r = (self.tmp() for _ in range(5))
            self.emit(f"{l} = OpExtInst %int %glsl FindILsb {self.as_int(lo)}")
            if op.endswith("b64"):
                self.emit(f"{h} = OpExtInst %int %glsl FindILsb {self.as_int(hi)}")
                hu = self.tmp()
                self.emit(f"{hu} = OpBitcast %uint {h}")
                hz = self.tmp()
                self.emit(f"{hz} = OpIEqual %bool {hi} {self.cu(0)}")
                self.emit(f"{hp} = OpIAdd %uint {hu} {self.cu(32)}")
                hsel = self.tmp()
                self.emit(f"{hsel} = OpSelect %uint {hz} {self.cu(0xFFFFFFFF)} {hp}")
                self.emit(f"{lz} = OpIEqual %bool {lo} {self.cu(0)}")
                lu = self.tmp()
                self.emit(f"{lu} = OpBitcast %uint {l}")
                self.emit(f"{r} = OpSelect %uint {lz} {hsel} {lu}")
            else:
                lu = self.tmp()
                self.emit(f"{lu} = OpBitcast %uint {l}")
                r = lu
            self.dst(a[0], r)
        elif op == "ds_swizzle_b32":
            self.swizzle(a, mods, ins)
        elif op in ("v_add_f64", "v_cvt_f32_f64"):
            # f64: register pairs as doubles (shaderFloat64). neg/abs modifiers apply to the double.
            def f64(sv):
                if sv["k"] in ("vgpr", "sgpr") and sv.get("n") == 2:
                    lo, hi = self.load_reg(sv["k"], sv["r"]), self.load_reg(sv["k"], sv["r"] + 1)
                elif sv["k"] == "imm" or sv["k"] == "fimm":
                    import struct as _st
                    bits = _st.unpack("<Q", _st.pack("<d", float(sv["v"])))[0] if sv["k"] == "fimm" else sv["v"] & 0xFFFFFFFFFFFFFFFF
                    lo, hi = self.cu(bits & 0xFFFFFFFF), self.cu(bits >> 32)
                else:
                    raise Unsupported(op + " operand")
                v2, d = self.tmp(), self.tmp()
                self.emit(f"{v2} = OpCompositeConstruct %v2uint_d {lo} {hi}")
                self.emit(f"{d} = OpBitcast %double {v2}")
                if sv.get("abs"):
                    t = self.tmp()
                    self.emit(f"{t} = OpExtInst %double %glsl FAbs {d}")
                    d = t
                if sv.get("neg"):
                    t = self.tmp()
                    self.emit(f"{t} = OpFNegate %double {d}")
                    d = t
                return d
            self.uses_f64 = True
            if op == "v_add_f64":
                x, y = f64(a[1]), f64(a[2])
                r, v2, lo, hi = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{r} = OpFAdd %double {x} {y}")
                # VOP3_Profile<VOP_F64_F64_F64>: HasClamp and HasOMod (LLVM VOPProfile, f64 is a modifier type).
                r = self.clamp_mods_f64(r, mods)
                self.emit(f"{v2} = OpBitcast %v2uint_d {r}")
                self.emit(f"{lo} = OpCompositeExtract %uint {v2} 0")
                self.emit(f"{hi} = OpCompositeExtract %uint {v2} 1")
                d = a[0]
                if d["k"] != "vgpr":
                    raise Unsupported(op + " destination")
                self.dst({"k": "vgpr", "r": d["r"], "n": 1}, lo)
                self.dst({"k": "vgpr", "r": d["r"] + 1, "n": 1}, hi)
            else:
                x = f64(a[1])
                r = self.tmp()
                self.emit(f"{r} = OpFConvert %float {x}")
                self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("image_bvh_intersect_ray", "image_bvh64_intersect_ray"):
            self.image_bvh(op, a, mods)
        elif op == "s_bfm_b64":
            # D = ((1 << size) - 1) << offset as a lane mask: in the per-lane model this lane's bit is
            # offset <= lane < offset + size (size/offset taken mod 64).
            size, off = self.src_u(a[1]), self.src_u(a[2])
            lane = self.wave_lane()
            sz, of, end, ge, lt, on, m = (self.tmp() for _ in range(7))
            self.emit(f"{sz} = OpBitwiseAnd %uint {size} {self.cu(63)}")
            self.emit(f"{of} = OpBitwiseAnd %uint {off} {self.cu(63)}")
            self.emit(f"{end} = OpIAdd %uint {of} {sz}")
            self.emit(f"{ge} = OpUGreaterThanEqual %bool {lane} {of}")
            self.emit(f"{lt} = OpULessThan %bool {lane} {end}")
            self.emit(f"{on} = OpLogicalAnd %bool {ge} {lt}")
            self.emit(f"{m} = OpSelect %uint {on} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
            self.mask_dst(a[0], m)
        elif op in ("s_pack_ll_b32_b16", "s_pack_lh_b32_b16", "s_pack_hh_b32_b16"):
            # D = {S1 half, S0 half}: the low result half comes from S0, the high one from S1; the letters name the
            # half taken from each (l = bits 15:0, h = bits 31:16), S0 first. SCC is not written.
            x, y = self.src_u(a[1]), self.src_u(a[2])
            lo, hi, r = self.tmp(), self.tmp(), self.tmp()
            if op == "s_pack_hh_b32_b16":
                self.emit(f"{lo} = OpShiftRightLogical %uint {x} {self.cu(16)}")
            else:
                self.emit(f"{lo} = OpBitwiseAnd %uint {x} {self.cu(0xFFFF)}")
            if op == "s_pack_ll_b32_b16":
                self.emit(f"{hi} = OpShiftLeftLogical %uint {y} {self.cu(16)}")
            else:
                self.emit(f"{hi} = OpBitwiseAnd %uint {y} {self.cu(0xFFFF0000)}")
            self.emit(f"{r} = OpBitwiseOr %uint {lo} {hi}")
            self.dst(a[0], r)
        elif op == "s_bfm_b32":
            # D = ((1 << (S0 & 31)) - 1) << (S1 & 31): an ordinary scalar value
            x, y = self.src_u(a[1]), self.src_u(a[2])
            w, o, one, sh, mm, r = (self.tmp() for _ in range(6))
            self.emit(f"{w} = OpBitwiseAnd %uint {x} {self.cu(31)}")
            self.emit(f"{o} = OpBitwiseAnd %uint {y} {self.cu(31)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {self.cu(1)} {w}")
            self.emit(f"{mm} = OpISub %uint {sh} {self.cu(1)}")
            self.emit(f"{r} = OpShiftLeftLogical %uint {mm} {o}")
            self.dst(a[0], r)
        elif op == "ds_bpermute_b32":
            # vD = vData from lane ((vAddr + offset) / 4) mod 32: a subgroup shuffle (no LDS memory involved).
            d, ad, dat = a[0], self.src_u(a[1]), self.src_u(a[2])
            off = self.mod_int(mods, "offset", 0)
            s1, ln, ln2, r = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s1} = OpIAdd %uint {ad} {self.cu(off)}")
            self.emit(f"{ln} = OpShiftRightLogical %uint {s1} {self.cu(2)}")
            self.emit(f"{ln2} = OpBitwiseAnd %uint {ln} {self.cu(31)}")
            if self.emulate:
                # Wave64 runs it as two wave32s (RDNA ISA 10.4.4): lanes 32-63 read from their own half, and a
                # source lane that is off in EXEC reads as 0.
                half, src = self.tmp(), self.tmp()
                self.emit(f"{half} = OpBitwiseAnd %uint {self.wave_lane()} {self.cu(32)}")
                self.emit(f"{src} = OpBitwiseOr %uint {half} {ln2}")
                self.g_shuffle(r, dat, src, zero_inactive=True)
            elif self.wave64:
                # Same on a 64-wide subgroup: the half's base comes from the subgroup lane (on a 32-wide one it
                # is 0, as before). Until 2026-11-21 lanes 32-63 read lanes 0-31 there, and inactive sources
                # kept their value. GPU-checked with cs_w64x_bpermute. A shuffle from an invocation that isn't
                # running (the missing lanes of a partial wave) is undefined in SPIR-V, and RADV returns stale
                # registers; the ballot makes it read 0 like a lane that is off. GPU-checked with cs_w64x_partial.
                lane, half, src, z = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{lane} = OpLoad %uint %sg_lane")
                self.emit(f"{half} = OpBitwiseAnd %uint {lane} {self.cu(32)}")
                self.emit(f"{src} = OpBitwiseOr %uint {half} {ln2}")
                self.emit(f"{z} = OpSelect %uint {self.active_b()} {dat} {self.cu(0)}")
                bal, ex, sh = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{bal} = OpGroupNonUniformBallot %v4uint_bal %c3 %true")
                self.emit(f"{ex} = OpGroupNonUniformBallotBitExtract %bool %c3 {bal} {src}")
                self.emit(f"{sh} = OpGroupNonUniformShuffle %uint %c3 {z} {src}")
                self.emit(f"{r} = OpSelect %uint {ex} {sh} {self.cu(0)}")
                self.uses_subgroup = self.uses_ballot = True
            else:
                # Wave32: a source lane that is off in EXEC reads as 0 here too (RDNA ISA 10.4.4). Until
                # 2026-11-21 the raw value was shuffled. GPU-checked with cs_bpermute32. A source lane that isn't
                # running (a partial wave) reads 0 as well, through the same ballot as the wave64 path.
                z, bal, ex, sh = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{z} = OpSelect %uint {self.active_b()} {dat} {self.cu(0)}")
                self.emit(f"{bal} = OpGroupNonUniformBallot %v4uint_bal %c3 %true")
                self.emit(f"{ex} = OpGroupNonUniformBallotBitExtract %bool %c3 {bal} {ln2}")
                self.emit(f"{sh} = OpGroupNonUniformShuffle %uint %c3 {z} {ln2}")
                self.emit(f"{r} = OpSelect %uint {ex} {sh} {self.cu(0)}")
                self.uses_subgroup = self.uses_ballot = True
            self.dst(d, r)
        elif op == "ds_permute_b32":
            self.ds_permute(a, mods)
        elif op.startswith("ds_") and self.stage == "gs":
            self.gs_ds(op, a, mods)
        elif op.startswith("ds_"):
            self.lds_op(op, a, mods)
        elif op == "s_barrier":
            # Workgroup execution barrier; AMD orders LDS through s_waitcnt before it, so include workgroup memory.
            if self.stage not in ("cs", "ms"):
                raise Unsupported("s_barrier outside compute")
            if self.emulate:
                # The wave waits in the scheduler until every wave not done waits too (assemble).
                self.emit("OpStore %x_bar %true")
                self.x_split()
                return
            self.emit(f"OpControlBarrier {self.cu(2)} {self.cu(2)} {self.cu(0x108)}")
        elif op == "v_readlane_b32":
            # sD = vS[lane]: a subgroup shuffle from that lane (the result is uniform).
            slot = lane_masks.lane_slot(a[1], a[2])
            if slot in self.cur_masks:
                # A lane mask parked there by v_writelane (a saved EXEC, when the shader ran out of SGPRs): this
                # lane's bit comes back, from the slot the write kept it in. The shuffle would hand every lane
                # the bit of the one lane the mask was parked in.
                self.dst(a[0], self.load_reg("sgpr", self.parked[slot[1:]]))
                return
            self.uses_subgroup = True
            v, ln = self.src_u(a[1]), self.src_u(a[2])
            m, r = self.tmp(), self.tmp()
            # RDNA2: wave64 uses S1[5:0] (mod 64); wave32 uses S1[4:0] (mod 32). Masking wave32 with 63 left
            # lane selects 32..63 pointing past a 32-wide subgroup (SHADER-52 / FUZZ-7).
            self.emit(f"{m} = OpBitwiseAnd %uint {ln} {self.cu(63 if self.wave64 else 31)}")
            self.g_shuffle(r, v, m)
            self.dst(a[0], r)
        elif op == "v_writelane_b32":
            # vD[lane] = sS: only the named lane changes (independent of EXEC, as on hardware).
            self.uses_subgroup = True
            val, ln = self.src_u(a[1]), self.src_u(a[2])
            slot = lane_masks.lane_slot(a[0], a[2])
            if slot and slot[1:] in self.parked:        # somewhere a mask is fetched back from this lane: keep ours
                self.store_reg("sgpr", self.parked[slot[1:]], val)
            me, m, eq, old, r = (self.tmp() for _ in range(5))
            if self.stage in ("cs", "ms"):
                # The hardware lane: lanes 32-63 of a wave64 are lanes 0-31 of a second 32-wide subgroup.
                self.emit(f"{me} = OpCopyObject %uint {self.wave_lane()}")
            else:
                self.emit(f"{me} = OpLoad %uint %sg_lane")
            self.emit(f"{m} = OpBitwiseAnd %uint {ln} {self.cu(63 if self.wave64 else 31)}")
            self.emit(f"{eq} = OpIEqual %bool {me} {m}")
            p = self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_Function_uint %vgpr {self.cu(a[0]['r'])}")
            self.emit(f"{old} = OpLoad %uint {p}")
            self.emit(f"{r} = OpSelect %uint {eq} {val} {old}")
            self.emit(f"OpStore {p} {r}")
        elif op.startswith("buffer_store_dword") or op in ("buffer_store_format_x", "buffer_store_format_xy",
                                                            "buffer_store_format_xyz", "buffer_store_format_xyzw",
                                                            "buffer_store_format_d16_x", "buffer_store_format_d16_xy",
                                                            "buffer_store_format_d16_xyz", "buffer_store_format_d16_xyzw"):
            self.buffer_store(op, a, mods)
        elif op in ("v_permlanex16_b32", "v_permlane16_b32"):
            # Lane l reads lane sel[l & 15] of (x16: the other 16-lane half of its row; plain: the same half).
            # The 4-bit selectors for lanes 0-7 are in src1, for lanes 8-15 in src2.
            self.uses_subgroup = True
            v, s1, s2 = self.src_u(a[1]), self.src_u(a[2]), self.src_u(a[3])
            lane, within, hi8, sh, nib1, nib2, nib, row, src = (self.tmp() for _ in range(9))
            self.sub_lane(lane)
            self.emit(f"{within} = OpBitwiseAnd %uint {lane} {self.cu(15)}")
            self.emit(f"{hi8} = OpUGreaterThanEqual %bool {within} {self.cu(8)}")
            w7 = self.tmp()
            self.emit(f"{w7} = OpBitwiseAnd %uint {within} {self.cu(7)}")
            self.emit(f"{sh} = OpIMul %uint {w7} {self.cu(4)}")
            self.emit(f"{nib1} = OpBitFieldUExtract %uint {s1} {sh} {self.cu(4)}")
            self.emit(f"{nib2} = OpBitFieldUExtract %uint {s2} {sh} {self.cu(4)}")
            self.emit(f"{nib} = OpSelect %uint {hi8} {nib2} {nib1}")
            half = self.tmp()
            self.emit(f"{half} = OpBitwiseAnd %uint {lane} {self.cu(0xFFFFFFF0)}")
            if op == "v_permlanex16_b32":
                self.emit(f"{row} = OpBitwiseXor %uint {half} {self.cu(16)}")
            else:
                self.emit(f"{row} = OpCopyObject %uint {half}")
            self.emit(f"{src} = OpBitwiseOr %uint {row} {nib}")
            r = self.tmp()
            self.g_shuffle(r, v, src)
            self.dst(a[0], r)
        elif op.startswith("buffer_load_dword"):
            self.buffer_load(op, a, mods)
        elif op in ("buffer_load_format_x", "buffer_load_format_xy", "buffer_load_format_xyz",
                    "buffer_load_format_xyzw", "buffer_load_format_d16_x", "buffer_load_format_d16_xy",
                    "buffer_load_format_d16_xyz", "buffer_load_format_d16_xyzw"):
            self.buffer_load_format(op, a, mods)
        elif op in ("tbuffer_load_format_x", "tbuffer_load_format_xy", "tbuffer_load_format_xyz",
                    "tbuffer_load_format_xyzw"):
            # Typed load with the format in the instruction (format:[BUF_FMT_...]) instead of the V#.
            fm = next((m for m in mods if m.startswith("format:")), None)
            if fm is None:
                raise Unsupported(op + " without format")
            name = "GFX10_FORMAT_" + fm.split("BUF_FMT_")[-1].rstrip("]")
            code = next((c for c, n in GFX10_FORMATS.items() if n == name), None)
            if code is None or code not in self.FORMATS:
                raise Unsupported(op + " " + name)
            self.buffer_load_format(op[1:], a, mods, fixed_fmt=code)
        elif op in ("s_bitset0_b64", "s_bitset1_b64"):
            # D[S0 & 63] = 0 / 1 in the SGPR pair D (D is both source and destination); SCC is not written. The pair
            # is 64-bit data (as for s_lshl_b64); a pair that lane_masks.py marks as a lane mask stays Unsupported.
            d = a[0]
            if d["k"] != "sgpr" or d.get("n") != 2 or self.holds_lane_mask(d):
                raise Unsupported(op + " destination")
            b = self.src_u(a[1])
            m, low, bit, sh = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {b} {self.cu(63)}")
            self.emit(f"{low} = OpULessThan %bool {m} {self.cu(32)}")
            self.emit(f"{bit} = OpBitwiseAnd %uint {m} {self.cu(31)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {self.cu(1)} {bit}")
            for half in (0, 1):
                old, want, mk, r = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{old} = OpCopyObject %uint {self.load_reg('sgpr', d['r'] + half)}")
                # The bit lands in this half when (m < 32) matches (half == 0).
                if half == 0:
                    self.emit(f"{mk} = OpSelect %uint {low} {sh} {self.cu(0)}")
                else:
                    self.emit(f"{mk} = OpSelect %uint {low} {self.cu(0)} {sh}")
                if op == "s_bitset1_b64":
                    self.emit(f"{r} = OpBitwiseOr %uint {old} {mk}")
                else:
                    self.emit(f"{want} = OpNot %uint {mk}")
                    self.emit(f"{r} = OpBitwiseAnd %uint {old} {want}")
                self.store_reg("sgpr", d["r"] + half, r)
        elif op == "s_bitset1_b32":
            # sD |= 1 << (src & 31)   (sD is both source and destination)
            b = self.src_u(a[1])
            m, sh, old, r = (self.tmp() for _ in range(4))
            self.emit(f"{m} = OpBitwiseAnd %uint {b} {self.cu(31)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {self.cu(1)} {m}")
            self.emit(f"{old} = OpCopyObject %uint {self.src_u(a[0])}")
            self.emit(f"{r} = OpBitwiseOr %uint {old} {sh}")
            self.dst(a[0], r)
        elif op in ("v_mul_hi_u32_u24", "v_mul_hi_i32_i24"):
            # High 32 bits of the 48-bit product of the low 24 bits of each source: zero-extended for u24,
            # sign-extended from bit 23 for i24 (RDNA2 ISA). The low 32 bits are v_mul_u32_u24 / v_mul_i32_i24.
            x, y = self.src_u(a[1]), self.src_u(a[2])
            signed = op == "v_mul_hi_i32_i24"
            xs, ys, s, r = (self.tmp() for _ in range(4))
            ext = "OpBitFieldSExtract" if signed else "OpBitFieldUExtract"
            self.emit(f"{xs} = {ext} %uint {x} {self.cu(0)} {self.cu(24)}")
            self.emit(f"{ys} = {ext} %uint {y} {self.cu(0)} {self.cu(24)}")
            if signed:
                xi, yi, hi = self.as_int(xs), self.as_int(ys), self.tmp()
                self.emit(f"{s} = OpSMulExtended %smul {xi} {yi}")
                self.emit(f"{hi} = OpCompositeExtract %int {s} 1")
                self.emit(f"{r} = OpBitcast %uint {hi}")
                self.uses_smul = True
            else:
                self.emit(f"{s} = OpUMulExtended %umul {xs} {ys}")
                self.emit(f"{r} = OpCompositeExtract %uint {s} 1")
            self.dst(a[0], r)
        elif op in ("v_mul_i32_i24", "v_mad_i32_i24"):
            # signed 24-bit operands (sign-extended from bit 23), low 32 bits of the product (+ src2 for mad)
            x, y = self.src_u(a[1]), self.src_u(a[2])
            xs, ys, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{xs} = OpBitFieldSExtract %uint {x} {self.cu(0)} {self.cu(24)}")
            self.emit(f"{ys} = OpBitFieldSExtract %uint {y} {self.cu(0)} {self.cu(24)}")
            self.emit(f"{r} = OpIMul %uint {xs} {ys}")
            if op == "v_mad_i32_i24":
                r2 = self.tmp()
                self.emit(f"{r2} = OpIAdd %uint {r} {self.src_u(a[3])}")
                r = r2
            self.dst(a[0], r)
        elif op == "v_lerp_u8":
            # Per byte: D.b = (S0.b + S1.b + (bit 0 of S2.b)) >> 1, an unsigned average whose rounding comes from
            # S2 (RDNA ISA pseudocode). The 9-bit sum never reaches past its own byte after the shift.
            if "clamp" in mods:
                raise Unsupported(op + " clamp")
            x, y, z = (self.src_u(t) for t in a[1:4])
            r = None
            for off in (0, 8, 16, 24):
                xb, yb, zb, t1, t2, avg = (self.tmp() for _ in range(6))
                self.emit(f"{xb} = OpBitFieldUExtract %uint {x} {self.cu(off)} {self.cu(8)}")
                self.emit(f"{yb} = OpBitFieldUExtract %uint {y} {self.cu(off)} {self.cu(8)}")
                self.emit(f"{zb} = OpBitFieldUExtract %uint {z} {self.cu(off)} {self.cu(1)}")
                self.emit(f"{t1} = OpIAdd %uint {xb} {yb}")
                self.emit(f"{t2} = OpIAdd %uint {t1} {zb}")
                self.emit(f"{avg} = OpShiftRightLogical %uint {t2} {self.cu(1)}")
                if r is None:
                    r = avg
                else:
                    sh, o = self.tmp(), self.tmp()
                    self.emit(f"{sh} = OpShiftLeftLogical %uint {avg} {self.cu(off)}")
                    self.emit(f"{o} = OpBitwiseOr %uint {r} {sh}")
                    r = o
            self.dst(a[0], r)
        elif op == "v_xnor_b32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            t, r = self.tmp(), self.tmp()
            self.emit(f"{t} = OpBitwiseXor %uint {x} {y}")
            self.emit(f"{r} = OpNot %uint {t}")
            self.dst(a[0], r)
        elif op == "v_alignbyte_b32":
            # (({src0, src1} as 64 bits) >> (8 * (src2 & 3))) low 32 bits
            # (32 - b) & 31 keeps the left shift below 32 when b == 0; that result is discarded by the OpSelect.
            hi_, lo_, sh = (self.src_u(t) for t in a[1:4])
            m, b, inv0, inv, a1, b1, nz, r0, r = (self.tmp() for _ in range(9))
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(3)}")
            self.emit(f"{b} = OpShiftLeftLogical %uint {m} {self.cu(3)}")
            self.emit(f"{inv0} = OpISub %uint {self.cu(32)} {b}")
            self.emit(f"{inv} = OpBitwiseAnd %uint {inv0} {self.cu(31)}")
            self.emit(f"{a1} = OpShiftRightLogical %uint {lo_} {b}")
            self.emit(f"{b1} = OpShiftLeftLogical %uint {hi_} {inv}")
            self.emit(f"{r0} = OpBitwiseOr %uint {a1} {b1}")
            self.emit(f"{nz} = OpINotEqual %bool {b} {self.cu(0)}")
            self.emit(f"{r} = OpSelect %uint {nz} {r0} {lo_}")
            self.dst(a[0], r)
        elif op == "v_swap_b32":
            # exchange two VGPRs (active lanes only, through dst)
            x, y = self.src_u(a[0]), self.src_u(a[1])
            xc, yc = self.tmp(), self.tmp()
            self.emit(f"{xc} = OpCopyObject %uint {x}")
            self.emit(f"{yc} = OpCopyObject %uint {y}")
            self.dst(a[0], yc)
            self.dst(a[1], xc)
        elif op == "s_abs_i32":
            x = self.src_u(a[1])
            i, ab, r, nz, sc = (self.tmp() for _ in range(5))
            self.emit(f"{i} = OpBitcast %int {x}")
            self.emit(f"{ab} = OpExtInst %int %glsl SAbs {i}")
            self.emit(f"{r} = OpBitcast %uint {ab}")
            self.emit(f"{nz} = OpINotEqual %bool {r} {self.cu(0)}")
            self.emit(f"{sc} = OpSelect %uint {nz} {self.cu(1)} {self.cu(0)}")
            self.dst(a[0], r)
            self.emit(f"OpStore %scc {sc}")
        elif op in ("s_sext_i32_i8", "s_sext_i32_i16"):
            # D = S0's low byte / half, sign-extended to 32 bits; SCC is not written.
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpBitFieldSExtract %uint {x} {self.cu(0)} {self.cu(8 if op.endswith('i8') else 16)}")
            self.dst(a[0], r)
        elif op in ("v_ffbh_i32", "s_flbit_i32"):
            # bit position counted from the MSB of the first bit that differs from the sign bit; -1 if none.
            # s_flbit_i32 is the scalar form (it does not write SCC).
            x = self.as_int(self.src_u(a[1]))
            f, none, sub, r = (self.tmp() for _ in range(4))
            self.emit(f"{f} = OpExtInst %int %glsl FindSMsb {x}")
            self.emit(f"{none} = OpSLessThan %bool {f} {self.as_int(self.cu(0))}")
            self.emit(f"{sub} = OpISub %int {self.as_int(self.cu(31))} {f}")
            subu = self.tmp()
            self.emit(f"{subu} = OpBitcast %uint {sub}")
            self.emit(f"{r} = OpSelect %uint {none} {self.cu(0xFFFFFFFF)} {subu}")
            self.dst(a[0], r)
        elif op.startswith(("v_cmp_", "v_cmpx_")) and op.endswith(("_u64", "_i64")) and \
                op.split("_")[2] in ("lt", "le", "gt", "ge"):
            # 64-bit ordered compare: high words decide (signed for i64), equal high words fall back to the low words.
            # v_cmpx (GFX10) writes EXEC only.
            cmpx = op.startswith("v_cmpx_")
            d, x, y = (None, a[-2], a[-1]) if cmpx else (a[0], a[1], a[2])
            def pair64(sv):
                if sv["k"] in ("vgpr", "sgpr") and sv.get("n") == 2:
                    return self.load_reg(sv["k"], sv["r"]), self.load_reg(sv["k"], sv["r"] + 1)
                if sv["k"] == "imm":
                    v = sv["v"] & 0xFFFFFFFFFFFFFFFF
                    return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                raise Unsupported(op + " operand")
            (x0, x1), (y0, y1) = pair64(x), pair64(y)
            cond, signed = op.split("_")[2], op.endswith("_i64")
            hi_op = {"lt": "LessThan", "le": "LessThan", "gt": "GreaterThan", "ge": "GreaterThan"}[cond]
            lo_op = {"lt": "ULessThan", "le": "ULessThanEqual", "gt": "UGreaterThan", "ge": "UGreaterThanEqual"}[cond]
            hx, hy = (self.as_int(x1), self.as_int(y1)) if signed else (x1, y1)
            hlt, heq, llt, both, res, r = (self.tmp() for _ in range(6))
            self.emit(f"{hlt} = Op{'S' if signed else 'U'}{hi_op} %bool {hx} {hy}")
            self.emit(f"{heq} = OpIEqual %bool {x1} {y1}")
            self.emit(f"{llt} = Op{lo_op} %bool {x0} {y0}")
            self.emit(f"{both} = OpLogicalAnd %bool {heq} {llt}")
            self.emit(f"{res} = OpLogicalOr %bool {hlt} {both}")
            ra = self.tmp()
            self.emit(f"{ra} = OpLogicalAnd %bool {res} {self.active_b()}")   # inactive lanes write 0
            if cmpx:
                self.emit(f"OpStore %active {ra}")
                self.uses_active = True
                return
            self.emit(f"{r} = OpSelect %uint {ra} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
            if d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
                self.emit(f"OpStore %vcc {r}")
            else:
                self.mask_dst(d, r)
        elif op == "v_alignbit_b32":
            # (({src0, src1} as 64 bits) >> (src2 & 31)) low 32 bits
            # (32 - m) & 31 keeps the left shift below 32 when m == 0; that result is discarded by the OpSelect.
            hi_, lo_, sh = (self.src_u(t) for t in a[1:4])
            m, inv0, inv, a1, b1, nz, r0, r = (self.tmp() for _ in range(8))
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{inv0} = OpISub %uint {self.cu(32)} {m}")
            self.emit(f"{inv} = OpBitwiseAnd %uint {inv0} {self.cu(31)}")
            self.emit(f"{a1} = OpShiftRightLogical %uint {lo_} {m}")
            self.emit(f"{b1} = OpShiftLeftLogical %uint {hi_} {inv}")
            self.emit(f"{r0} = OpBitwiseOr %uint {a1} {b1}")
            self.emit(f"{nz} = OpINotEqual %bool {m} {self.cu(0)}")
            self.emit(f"{r} = OpSelect %uint {nz} {r0} {lo_}")
            self.dst(a[0], r)
        elif op in ("v_lshlrev_b64", "v_lshrrev_b64", "v_ashrrev_i64"):
            # v[d:d+1] = v[s:s+1] << / >> (src0 & 63): logical left, logical right, or arithmetic right (the sign
            # bit 63 is copied in).
            sh, src = a[1], a[2]
            if a[0].get("n") != 2:
                raise Unsupported(op + " operands")
            if src["k"] in ("vgpr", "sgpr") and src.get("n") == 2:
                lo, hi = self.load_reg(src["k"], src["r"]), self.load_reg(src["k"], src["r"] + 1)
            elif src["k"] == "imm":
                # 64-bit inline/literal constant: sign-extended from 32 bits.
                v = src["v"] & 0xFFFFFFFFFFFFFFFF
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu((v >> 32) & 0xFFFFFFFF)
            else:
                raise Unsupported(op + " operands")
            s = self.src_u(sh)
            l64, h64, x, sh64, m, res, rlo, rhi, t = (self.tmp() for _ in range(9))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{t} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{x} = OpBitwiseOr %ulong {t} {l64}")
            self.emit(f"{m} = OpBitwiseAnd %uint {s} {self.cu(63)}")
            self.emit(f"{sh64} = OpUConvert %ulong {m}")
            shop = {"v_lshlrev_b64": "OpShiftLeftLogical", "v_lshrrev_b64": "OpShiftRightLogical",
                    "v_ashrrev_i64": "OpShiftRightArithmetic"}[op]
            self.emit(f"{res} = {shop} %ulong {x} {sh64}")
            self.emit(f"{rlo} = OpUConvert %uint {res}")
            hs = self.tmp()
            self.emit(f"{hs} = OpShiftRightLogical %ulong {res} %c64_32")
            self.emit(f"{rhi} = OpUConvert %uint {hs}")
            self.uses_psb = True        # needs Int64 (declared with the PSB types)
            self.dst({"k": "vgpr", "r": a[0]["r"], "n": 1}, rlo)
            self.dst({"k": "vgpr", "r": a[0]["r"] + 1, "n": 1}, rhi)
        elif op == "v_fmac_f16":
            # dst = fma(src0, src1, dst). The accumulator is dst itself: low half, or high half under op_sel's dst bit.
            sel, dhi = self.opsel_bits(mods, 3)
            vals = [self.f16_in(a[1], sel[0]), self.f16_in(a[2], sel[1]), self.f16_in(a[0], sel[2])]
            r = self.fma_f16_odd(*vals)     # one rounding, to f16 (SHADER-55)
            self.f16_out(a[0], r, dhi)  # GFX10: f16 ops keep the other half
        elif op in ("v_add_nc_u16", "v_sub_nc_u16"):
            # VOP3-only GFX10 names of the 16-bit add/sub (no carry): opsel-capable, so the high half is kept.
            self._insn({**ins, "op": ("v_add_u16_e64" if op == "v_add_nc_u16" else "v_sub_u16_e64")})
        elif op in ("v_lshrrev_b16", "v_lshlrev_b16", "v_ashrrev_i16", "v_add_u16", "v_sub_u16", "v_subrev_u16",
                    "v_mul_lo_u16", "v_max_u16", "v_min_u16", "v_max_i16", "v_min_i16", "v_and_b16", "v_or_b16",
                    "v_xor_b16"):
            # 16-bit integer ops on the low halves (op_sel picks high halves in the VOP3 form). Per Mesa ACO
            # (can_use_opsel): the opsel-capable VOP3 (_e64) forms are partial writes that keep the high half,
            # and the VOP2 (_e32) forms define the whole register (high half modelled as zero).
            # VOP3 = an 8-byte encoding. LLVM 21 prints the VOP3 form of these ops without the _e64 suffix (every
            # one in the game's shaders), so the mnemonic alone would make them VOP2 writes and zero the high half.
            enc = ins.get("enc")
            e64 = len(enc) == 16 if enc else ins["op"].endswith("_e64")
            sel = next((m for m in mods if m.startswith("op_sel:")), "op_sel:[0,0,0]")
            sb = [int(x) for x in sel.split("[")[1].rstrip("]").split(",")] + [0, 0, 0]
            if not e64 and sb[:2] != [0, 0]:
                raise Unsupported(op + " op_sel")
            halves = []
            for k, s in enumerate(a[1:3]):
                u = self.src_u(s)
                if sb[k]:
                    t = self.tmp()
                    self.emit(f"{t} = OpShiftRightLogical %uint {u} {self.cu(16)}")
                    u = t
                h = self.tmp()
                self.emit(f"{h} = OpBitwiseAnd %uint {u} {self.cu(0xFFFF)}")
                halves.append(h)
            x, y = halves
            r = self.tmp()
            if op in ("v_lshrrev_b16", "v_lshlrev_b16", "v_ashrrev_i16"):
                sh = self.tmp()                      # shift amount is src0 (reversed operand order), low 4 bits
                self.emit(f"{sh} = OpBitwiseAnd %uint {x} {self.cu(15)}")
                if op == "v_lshrrev_b16":
                    self.emit(f"{r} = OpShiftRightLogical %uint {y} {sh}")
                elif op == "v_lshlrev_b16":
                    self.emit(f"{r} = OpShiftLeftLogical %uint {y} {sh}")
                else:
                    yi, ri = self.tmp(), self.tmp()
                    self.emit(f"{yi} = OpBitFieldSExtract %int {self.as_int(y)} {self.cu(0)} {self.cu(16)}")
                    self.emit(f"{ri} = OpShiftRightArithmetic %int {yi} {sh}")
                    self.emit(f"{r} = OpBitcast %uint {ri}")
            elif op in ("v_max_i16", "v_min_i16"):
                xi, yi, ri = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{xi} = OpBitFieldSExtract %int {self.as_int(x)} {self.cu(0)} {self.cu(16)}")
                self.emit(f"{yi} = OpBitFieldSExtract %int {self.as_int(y)} {self.cu(0)} {self.cu(16)}")
                self.emit(f"{ri} = OpExtInst %int %glsl {'SMax' if 'max' in op else 'SMin'} {xi} {yi}")
                self.emit(f"{r} = OpBitcast %uint {ri}")
            else:
                k = {"v_add_u16": "OpIAdd", "v_sub_u16": "OpISub", "v_mul_lo_u16": "OpIMul", "v_and_b16": "OpBitwiseAnd",
                     "v_or_b16": "OpBitwiseOr", "v_xor_b16": "OpBitwiseXor"}.get(op)
                if op == "v_subrev_u16":
                    self.emit(f"{r} = OpISub %uint {y} {x}")
                elif k:
                    self.emit(f"{r} = {k} %uint {x} {y}")
                else:
                    self.emit(f"{r} = OpExtInst %uint %glsl {'UMax' if 'max' in op else 'UMin'} {x} {y}")
            lo = self.tmp()
            self.emit(f"{lo} = OpBitwiseAnd %uint {r} {self.cu(0xFFFF)}")
            if e64 and sb[2]:
                # op_sel destination bit: the result goes to the high half, the low half is kept.
                old, keep, sh16, v = (self.tmp() for _ in range(4))
                self.emit(f"{old} = OpCopyObject %uint {self.load_reg('vgpr', a[0]['r'])}")
                self.emit(f"{keep} = OpBitwiseAnd %uint {old} {self.cu(0xFFFF)}")
                self.emit(f"{sh16} = OpShiftLeftLogical %uint {lo} {self.cu(16)}")
                self.emit(f"{v} = OpBitwiseOr %uint {keep} {sh16}")
                self.dst(a[0], v)
            else:
                self.write_lo16(a[0], lo, preserve_high=e64)
        elif op == "v_bcnt_u32_b32":
            x, acc = self.src_u(a[1]), self.src_u(a[2])
            c, r = self.tmp(), self.tmp()
            self.emit(f"{c} = OpBitCount %uint {x}")
            self.emit(f"{r} = OpIAdd %uint {c} {acc}")
            self.dst(a[0], r)
        elif op == "s_bitset0_b32":
            b = self.src_u(a[1])
            m, sh, n, r = (self.tmp() for _ in range(4))
            self.emit(f"{m} = OpBitwiseAnd %uint {b} {self.cu(31)}")
            self.emit(f"{sh} = OpShiftLeftLogical %uint {self.cu(1)} {m}")
            self.emit(f"{n} = OpNot %uint {sh}")
            self.emit(f"{r} = OpBitwiseAnd %uint {self.src_u(a[0])} {n}")
            self.dst(a[0], r)
        elif op in ("s_min_i32", "s_max_i32"):
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %uint %glsl {'SMin' if 'min' in op else 'SMax'} {x} {y}")
            self.dst(a[0], r)
            # SCC = 1 when S0 is the one chosen: the ISA's D = S0 < S1 ? S0 : S1 (min) / S0 > S1 ? S0 : S1 (max)
            # picks S1 on a tie, so SCC = S0 < S1 / S0 > S1, compared signed (as s_min_u32/s_max_u32 do unsigned).
            # Until 2026-11-21 equal operands set SCC to 1.
            c, z = self.tmp(), self.tmp()
            xi, yi = self.as_int(x), self.as_int(y)
            self.emit(f"{c} = {'OpSLessThan' if 'min' in op else 'OpSGreaterThan'} %bool {xi} {yi}")
            self.emit(f"{z} = OpSelect %uint {c} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "v_fma_f16":
            sel, dhi = self.opsel_bits(mods, 3)     # op_sel:[src0,src1,src2,dst]
            vals = [self.f16_in(s, sel[i]) for i, s in enumerate(a[1:4])]
            r = self.fma_f16_odd(*vals)     # one rounding, to f16 (SHADER-55); omod/clamp keep round-to-odd valid
            r = self.clamp_mods(r, mods)
            self.f16_out(a[0], r, dhi)  # GFX10: f16 ops keep the other half
        elif op in ("v_pk_add_f16", "v_pk_mul_f16", "v_pk_max_f16", "v_pk_min_f16", "v_pk_fma_f16"):
            # Packed f16x2 (VOP3P). Result lo: source i takes its half from op_sel[i] (0 = lo, 1 = hi), negated by
            # neg_lo[i]. Result hi: half from op_sel_hi[i], negated by neg_hi[i]. Defaults: op_sel [0,0],
            # op_sel_hi [1,1], no negation.
            def bits(name, default):
                m = next((m for m in mods if m.startswith(name + ":")), None)
                return [int(v) for v in m.split("[")[1].rstrip("]").split(",")] if m else default
            ns = 3 if op == "v_pk_fma_f16" else 2
            sel, selhi = bits("op_sel", [0] * ns), bits("op_sel_hi", [1] * ns)
            nlo, nhi = bits("neg_lo", [0] * ns), bits("neg_hi", [0] * ns)
            srcs = []
            for i in range(ns):
                u = self.src_h(a[1 + i])
                v2 = self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                comps = []
                for half_sel, neg in ((sel[i], nlo[i]), (selhi[i], nhi[i])):
                    c = self.tmp()
                    self.emit(f"{c} = OpCompositeExtract %float {v2} {half_sel}")
                    if neg:
                        n2 = self.tmp()
                        self.emit(f"{n2} = OpFNegate %float {c}")
                        c = n2
                    comps.append(c)
                vv = self.tmp()
                self.emit(f"{vv} = OpCompositeConstruct %v2float {comps[0]} {comps[1]}")
                srcs.append(vv)
            vx, vy = srcs[:2]
            r = self.tmp()
            if op == "v_pk_fma_f16":
                # one rounding per half, to f16 (SHADER-55)
                comps = []
                for k in (0, 1):
                    ex = [self.tmp() for _ in range(3)]
                    for e_, v_ in zip(ex, (vx, vy, srcs[2])):
                        self.emit(f"{e_} = OpCompositeExtract %float {v_} {k}")
                    comps.append(self.fma_f16_odd(*ex))
                self.emit(f"{r} = OpCompositeConstruct %v2float {comps[0]} {comps[1]}")
            elif op in ("v_pk_add_f16", "v_pk_mul_f16"):
                self.emit(f"{r} = {'OpFAdd' if 'add' in op else 'OpFMul'} %v2float {vx} {vy}")
            else:
                self.emit(f"{r} = OpExtInst %v2float %glsl {'FMax' if 'max' in op else 'FMin'} {vx} {vy}")
            if "clamp" in mods:
                c = self.tmp()
                z2, o2 = self.tmp(), self.tmp()
                self.emit(f"{z2} = OpCompositeConstruct %v2float %f0 %f0")
                self.emit(f"{o2} = OpCompositeConstruct %v2float %f1 %f1")
                self.emit(f"{c} = OpExtInst %v2float %glsl FClamp {r} {z2} {o2}")
                r = c
            p = self.tmp()
            c0, c1 = self.tmp(), self.tmp()
            self.emit(f"{c0} = OpCompositeExtract %float {r} 0")
            self.emit(f"{c1} = OpCompositeExtract %float {r} 1")
            self.emit(f"{p} = OpCopyObject %uint {self.f16_pack2(c0, c1)}")
            self.dst(a[0], p)
        elif op in ("v_movrels_b32", "v_movreld_b32", "v_movrelsd_b32", "v_movrelsd_2_b32"):
            # Relative VGPR indexing through M0: movrels reads VGPR[src + M0], movreld writes VGPR[dst + M0],
            # movrelsd does both. The register file is an array, so these become dynamic array accesses.
            # movrelsd_2 (GFX10) offsets each side by its own M0 field: VGPR[dst + M0[25:16]] =
            # VGPR[src + M0[9:0]] (RDNA2 ISA, VOP1 opcode 72; the RDNA3 ISA gives the same fields).
            d, s = a[0], a[1]
            if d["k"] != "vgpr" or s["k"] != "vgpr":
                raise Unsupported(op + " operands")
            m0 = self.tmp()
            self.emit(f"{m0} = OpLoad %uint %m0")
            m0_src = m0_dst = m0
            if op == "v_movrelsd_2_b32":
                m0_src, hi, m0_dst = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{m0_src} = OpBitwiseAnd %uint {m0} {self.cu(0x3FF)}")
                self.emit(f"{hi} = OpShiftRightLogical %uint {m0} {self.cu(16)}")
                self.emit(f"{m0_dst} = OpBitwiseAnd %uint {hi} {self.cu(0x3FF)}")
            if op in ("v_movrels_b32", "v_movrelsd_b32", "v_movrelsd_2_b32"):
                si, sp, v = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{si} = OpIAdd %uint {self.cu(s['r'])} {m0_src}")
                self.emit(f"{sp} = OpAccessChain %_ptr_Function_uint %vgpr {si}")
                self.emit(f"{v} = OpLoad %uint {sp}")
            else:
                v = self.load_reg("vgpr", s["r"])
            if op in ("v_movreld_b32", "v_movrelsd_b32", "v_movrelsd_2_b32"):
                di, dp, old, sel = (self.tmp() for _ in range(4))
                self.emit(f"{di} = OpIAdd %uint {self.cu(d['r'])} {m0_dst}")
                self.emit(f"{dp} = OpAccessChain %_ptr_Function_uint %vgpr {di}")
                self.emit(f"{old} = OpLoad %uint {dp}")
                self.emit(f"{sel} = OpSelect %uint {self.active_b()} {v} {old}")
                self.emit(f"OpStore {dp} {sel}")
            else:
                self.dst(d, v)
        elif op == "v_med3_f16":
            sel, dhi = self.opsel_bits(mods, 3)     # op_sel:[src0,src1,src2,dst]
            vals = [self.f16_in(s, sel[i]) for i, s in enumerate(a[1:4])]
            lo_, hi_, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{lo_} = OpExtInst %float %glsl FMin {vals[0]} {vals[1]}")
            self.emit(f"{hi_} = OpExtInst %float %glsl FMax {vals[0]} {vals[1]}")
            self.emit(f"{r} = OpExtInst %float %glsl FClamp {vals[2]} {lo_} {hi_}")
            r = self.clamp_mods(r, mods)
            self.f16_out(a[0], r, dhi)    # opsel-capable VOP3: keeps the other half (Mesa ACO can_use_opsel)
        elif op in ("v_qsad_pk_u16_u8", "v_mqsad_pk_u16_u8"):
            # Four byte-SADs packed as u16 (RDNA ISA pseudocode): for k = 0..3, D.u16[k] = (S2.u16[k] + SAD of the
            # 32-bit window S0[8k+31 : 8k] of the 64-bit S0 against S1) & 0xFFFF. The m form is masked: a byte where
            # S1's byte is 0 is skipped (as v_msad_u8). clamp is not modelled: rejected rather than guessed.
            if a[0].get("n") != 2 or "clamp" in mods:
                raise Unsupported(op + " operands")

            def pair(sv):
                if sv["k"] in ("vgpr", "sgpr") and sv.get("n") == 2:
                    return self.load_reg(sv["k"], sv["r"]), self.load_reg(sv["k"], sv["r"] + 1)
                if sv["k"] == "imm":
                    v = sv["v"] & 0xFFFFFFFFFFFFFFFF          # 64-bit inline/literal constant: sign-extended
                    return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                raise Unsupported(op + " operands")
            (x_lo, x_hi), y, (c_lo, c_hi) = pair(a[1]), self.src_u(a[2]), pair(a[3])
            words = []
            for k in range(4):
                if k == 0:
                    win = x_lo
                else:
                    t1, t2, win = self.tmp(), self.tmp(), self.tmp()
                    self.emit(f"{t1} = OpShiftRightLogical %uint {x_lo} {self.cu(8 * k)}")
                    self.emit(f"{t2} = OpShiftLeftLogical %uint {x_hi} {self.cu(32 - 8 * k)}")
                    self.emit(f"{win} = OpBitwiseOr %uint {t1} {t2}")
                acc_w = c_lo if k < 2 else c_hi
                total = self.tmp()
                self.emit(f"{total} = OpBitFieldUExtract %uint {acc_w} {self.cu(16 * (k % 2))} {self.cu(16)}")
                for off in (0, 8, 16, 24):
                    xb, yb, hi_, lo_, d, t = (self.tmp() for _ in range(6))
                    self.emit(f"{xb} = OpBitFieldUExtract %uint {win} {self.cu(off)} {self.cu(8)}")
                    self.emit(f"{yb} = OpBitFieldUExtract %uint {y} {self.cu(off)} {self.cu(8)}")
                    self.emit(f"{hi_} = OpExtInst %uint %glsl UMax {xb} {yb}")
                    self.emit(f"{lo_} = OpExtInst %uint %glsl UMin {xb} {yb}")
                    self.emit(f"{d} = OpISub %uint {hi_} {lo_}")
                    if op == "v_mqsad_pk_u16_u8":
                        md = self.tmp()
                        self.emit(f"{md} = OpSelect %uint {self.tmp_bool_nz(yb)} {d} {self.cu(0)}")
                        d = md
                    self.emit(f"{t} = OpIAdd %uint {total} {d}")
                    total = t
                w = self.tmp()
                self.emit(f"{w} = OpBitwiseAnd %uint {total} {self.cu(0xFFFF)}")
                words.append(w)
            out = []
            for lo16, hi16 in ((words[0], words[1]), (words[2], words[3])):
                sh, o = self.tmp(), self.tmp()
                self.emit(f"{sh} = OpShiftLeftLogical %uint {hi16} {self.cu(16)}")
                self.emit(f"{o} = OpBitwiseOr %uint {lo16} {sh}")
                out.append(o)
            self.dst({"k": "vgpr", "r": a[0]["r"], "n": 1}, out[0])
            self.dst({"k": "vgpr", "r": a[0]["r"] + 1, "n": 1}, out[1])
        elif op in ("v_sad_u8", "v_sad_hi_u8", "v_sad_u16", "v_sad_u32", "v_msad_u8"):
            # Sums of absolute differences (RDNA ISA pseudocode), all unsigned and wrapping at 32 bits:
            #   v_sad_u8:    S2 + sum over the 4 bytes of |S0.b - S1.b|
            #   v_sad_hi_u8: S2 + (that byte sum << 16)
            #   v_sad_u16:   S2 + sum over the 2 halves of |S0.h - S1.h|
            #   v_sad_u32:   S2 + |S0 - S1|
            #   v_msad_u8:   v_sad_u8, skipping each byte where S1's byte is 0 (masked SAD)
            # clamp is not modelled: rejected rather than guessed.
            if "clamp" in mods:
                raise Unsupported(op + " clamp")
            x, y, acc = (self.src_u(t) for t in a[1:4])
            width = {"v_sad_u16": 16, "v_sad_u32": 32}.get(op, 8)
            total = None
            for off in range(0, 32, width):
                if width == 32:
                    xa, ya = x, y
                else:
                    xa, ya = self.tmp(), self.tmp()
                    self.emit(f"{xa} = OpBitFieldUExtract %uint {x} {self.cu(off)} {self.cu(width)}")
                    self.emit(f"{ya} = OpBitFieldUExtract %uint {y} {self.cu(off)} {self.cu(width)}")
                hi_, lo_, d = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{hi_} = OpExtInst %uint %glsl UMax {xa} {ya}")
                self.emit(f"{lo_} = OpExtInst %uint %glsl UMin {xa} {ya}")
                self.emit(f"{d} = OpISub %uint {hi_} {lo_}")
                if op == "v_msad_u8":
                    md = self.tmp()
                    self.emit(f"{md} = OpSelect %uint {self.tmp_bool_nz(ya)} {d} {self.cu(0)}")
                    d = md
                if total is None:
                    total = d
                else:
                    t = self.tmp()
                    self.emit(f"{t} = OpIAdd %uint {total} {d}")
                    total = t
            if op == "v_sad_hi_u8":
                t = self.tmp()
                self.emit(f"{t} = OpShiftLeftLogical %uint {total} {self.cu(16)}")
                total = t
            r = self.tmp()
            self.emit(f"{r} = OpIAdd %uint {total} {acc}")
            self.dst(a[0], r)
        elif op == "v_mqsad_u32_u8":
            # Four masked byte-SADs with full 32-bit accumulators (RDNA ISA pseudocode): for k = 0..3,
            # D[k] = S2[k] + v_msad_u8 of the 32-bit window S0[8k+31 : 8k] of the 64-bit S0 against S1, wrapping at
            # 32 bits; a byte where S1's byte is 0 adds nothing. S2 and D are 128-bit (four VGPRs).
            # clamp is not modelled: rejected rather than guessed.
            if a[0].get("n") != 4 or "clamp" in mods:
                raise Unsupported(op + " operands")
            s0, s2 = a[1], a[3]
            if s0["k"] in ("vgpr", "sgpr") and s0.get("n") == 2:
                x_lo, x_hi = self.load_reg(s0["k"], s0["r"]), self.load_reg(s0["k"], s0["r"] + 1)
            elif s0["k"] == "imm":
                v = s0["v"] & 0xFFFFFFFFFFFFFFFF                  # 64-bit inline/literal constant: sign-extended
                x_lo, x_hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " operands")
            if s2["k"] == "vgpr" and s2.get("n") == 4:
                accs = [self.load_reg("vgpr", s2["r"] + k) for k in range(4)]
            else:
                raise Unsupported(op + " operands")
            y = self.src_u(a[2])
            outs = []
            for k in range(4):
                if k == 0:
                    win = x_lo
                else:
                    t1, t2, win = self.tmp(), self.tmp(), self.tmp()
                    self.emit(f"{t1} = OpShiftRightLogical %uint {x_lo} {self.cu(8 * k)}")
                    self.emit(f"{t2} = OpShiftLeftLogical %uint {x_hi} {self.cu(32 - 8 * k)}")
                    self.emit(f"{win} = OpBitwiseOr %uint {t1} {t2}")
                total = accs[k]
                for off in (0, 8, 16, 24):
                    xb, yb, hi_, lo_, d, md, t = (self.tmp() for _ in range(7))
                    self.emit(f"{xb} = OpBitFieldUExtract %uint {win} {self.cu(off)} {self.cu(8)}")
                    self.emit(f"{yb} = OpBitFieldUExtract %uint {y} {self.cu(off)} {self.cu(8)}")
                    self.emit(f"{hi_} = OpExtInst %uint %glsl UMax {xb} {yb}")
                    self.emit(f"{lo_} = OpExtInst %uint %glsl UMin {xb} {yb}")
                    self.emit(f"{d} = OpISub %uint {hi_} {lo_}")
                    self.emit(f"{md} = OpSelect %uint {self.tmp_bool_nz(yb)} {d} {self.cu(0)}")
                    self.emit(f"{t} = OpIAdd %uint {total} {md}")
                    total = t
                outs.append(total)
            for k in range(4):               # every source is read before the first write
                self.dst({"k": "vgpr", "r": a[0]["r"] + k, "n": 1}, outs[k])
        elif op == "v_xad_u32":
            x, y, z = (self.src_u(t) for t in a[1:4])
            t, r = self.tmp(), self.tmp()
            self.emit(f"{t} = OpBitwiseXor %uint {x} {y}")
            self.emit(f"{r} = OpIAdd %uint {t} {z}")
            self.dst(a[0], r)
        elif op == "s_not_b64":
            v = self.mask_src(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpNot %uint {v}")
            self.mask_dst(a[0], r)
            self.scc_from_mask(r)
        elif op in ("v_rsq_f16", "v_rcp_f16", "v_sqrt_f16", "v_exp_f16", "v_log_f16", "v_floor_f16", "v_fract_f16",
                    "v_ceil_f16", "v_trunc_f16", "v_rndne_f16"):
            # f16 unary on the low half (op_sel:[src,dst] picks the high half of the source / of the destination);
            # the other half of the destination is kept.
            sel, dhi = self.opsel_bits(mods, 1)
            f = self.f16_in(a[1], sel[0])
            r = self.tmp()
            fn = {"v_rsq_f16": "InverseSqrt", "v_sqrt_f16": "Sqrt", "v_exp_f16": "Exp2", "v_log_f16": "Log2",
                  "v_floor_f16": "Floor", "v_fract_f16": "Fract", "v_ceil_f16": "Ceil", "v_trunc_f16": "Trunc",
                  "v_rndne_f16": "RoundEven"}.get(op)
            if fn:
                self.emit(f"{r} = OpExtInst %float %glsl {fn} {f}")
            else:
                self.emit(f"{r} = OpFDiv %float %f1 {f}")
            r = self.clamp_mods(r, mods)
            self.f16_out(a[0], r, dhi)  # GFX10: f16 ops keep the other half
        elif op == "v_not_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpNot %uint {x}")
            self.dst(a[0], r)
        elif op == "v_mul_hi_i32":
            x, y = self.as_int(self.src_u(a[1])), self.as_int(self.src_u(a[2]))
            s, hi, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s} = OpSMulExtended %smul {x} {y}")
            self.emit(f"{hi} = OpCompositeExtract %int {s} 1")
            self.emit(f"{r} = OpBitcast %uint {hi}")
            self.uses_smul = True
            self.dst(a[0], r)
        elif op in ("v_add_f16", "v_sub_f16", "v_subrev_f16", "v_mul_f16", "v_max_f16", "v_min_f16"):
            # f16 math on the low halves (op_sel:[src0,src1,dst] in the VOP3 form); the other half of the destination
            # is kept. v_subrev_f16 is S1 - S0: op_sel still names src0 and src1, only the subtraction is swapped.
            sel, dhi = self.opsel_bits(mods, 2)
            vals = [self.f16_in(s, sel[i]) for i, s in enumerate(a[1:3])]
            if op == "v_subrev_f16":
                vals.reverse()
            r = self.tmp()
            k = {"v_add_f16": "OpFAdd", "v_sub_f16": "OpFSub", "v_subrev_f16": "OpFSub", "v_mul_f16": "OpFMul"}.get(op)
            if k:
                self.emit(f"{r} = {k} %float {vals[0]} {vals[1]}")
            else:
                self.emit(f"{r} = OpExtInst %float %glsl {'FMax' if 'max' in op else 'FMin'} {vals[0]} {vals[1]}")
            r = self.clamp_mods(r, mods)
            self.f16_out(a[0], r, dhi)  # GFX10: f16 ops keep the other half
        elif op.startswith(("v_cmp_", "v_cmpx_")) and op.endswith(("_u64", "_i64")) and op.split("_")[2] in ("eq", "ne", "lg"):
            # 64-bit equality compare: both halves of the register pairs. GFX10 v_cmpx has no destination operand:
            # it writes EXEC only (inactive lanes stay inactive).
            cmpx = op.startswith("v_cmpx_")
            d, x, y = (None, a[-2], a[-1]) if cmpx else (a[0], a[1], a[2])
            def pair(s):
                if s["k"] in ("vgpr", "sgpr") and s.get("n") == 2:
                    return self.load_reg(s["k"], s["r"]), self.load_reg(s["k"], s["r"] + 1)
                if s["k"] == "imm":
                    v = s["v"] & 0xFFFFFFFFFFFFFFFF
                    return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                raise Unsupported(op + " operand")
            (x0, x1), (y0, y1) = pair(x), pair(y)
            e0, e1, eq, r = (self.tmp() for _ in range(4))
            self.emit(f"{e0} = OpIEqual %bool {x0} {y0}")
            self.emit(f"{e1} = OpIEqual %bool {x1} {y1}")
            self.emit(f"{eq} = OpLogicalAnd %bool {e0} {e1}")
            if op.split("_")[2] != "eq":
                ne = self.tmp()
                self.emit(f"{ne} = OpLogicalNot %bool {eq}")
                eq = ne
            if cmpx:
                both = self.tmp()
                self.emit(f"{both} = OpLogicalAnd %bool {eq} {self.active_b()}")
                self.emit(f"OpStore %active {both}")
                self.uses_active = True
            else:
                eqa = self.tmp()
                self.emit(f"{eqa} = OpLogicalAnd %bool {eq} {self.active_b()}")   # inactive lanes write 0
                self.emit(f"{r} = OpSelect %uint {eqa} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
                if d["k"] == "special" and d["t"] in ("vcc", "vcc_lo"):
                    self.emit(f"OpStore %vcc {r}")
                else:
                    self.mask_dst(d, r)
        elif op == "s_not_b32":
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpNot %uint {x}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op in ("s_cmp_eq_u64", "s_cmp_lg_u64"):
            # 64-bit compare of masks/pointers: equal iff both halves are equal.
            m0, m1 = self.holds_lane_mask(a[0]), self.holds_lane_mask(a[1])

            def _is_u64_zero_or_all_ones(src):
                if src["k"] != "imm":
                    return False
                v = src["v"] & 0xFFFFFFFFFFFFFFFF
                return v == 0 or (self.wave64 and v == 0xFFFFFFFFFFFFFFFF)

            use_vote = (m0 and m1) or (m0 and _is_u64_zero_or_all_ones(a[1])) or (m1 and _is_u64_zero_or_all_ones(a[0]))

            def halves(s, is_mask):
                if is_mask and not use_vote:
                    return self.wave_mask(s)
                if s["k"] == "sgpr" and s.get("n") == 2:
                    return self.load_reg("sgpr", s["r"]), self.load_reg("sgpr", s["r"] + 1)
                if s["k"] == "imm":
                    v = s["v"] & 0xFFFFFFFFFFFFFFFF
                    return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                m = self.mask_src(s)
                return m, m

            (x0, x1), (y0, y1) = halves(a[0], m0), halves(a[1], m1)
            e0, e1, eq, r = (self.tmp() for _ in range(4))
            self.emit(f"{e0} = OpIEqual %bool {x0} {y0}")
            self.emit(f"{e1} = OpIEqual %bool {x1} {y1}")
            self.emit(f"{eq} = OpLogicalAnd %bool {e0} {e1}")
            if use_vote:
                eq = self.wave_equal(eq, True)
            if op == "s_cmp_lg_u64":
                ne = self.tmp()
                self.emit(f"{ne} = OpLogicalNot %bool {eq}")
                eq = ne
            self.emit(f"{r} = OpSelect %uint {eq} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {r}")
        elif op.startswith("buffer_atomic_"):
            self.buffer_atomic(op, a, mods)
        elif op == "image_get_lod":
            self.image_get_lod(a, mods)
        elif op == "s_addk_i32":
            v = a[1]["v"] & 0xFFFF
            k = v - 0x10000 if v & 0x8000 else v
            x = self.src_u(a[0])            # the destination can be m0/vcc_hi as well as an SGPR
            r = self.tmp()
            self.emit(f"{r} = OpIAdd %uint {x} {self.cu(k)}")
            self.dst(a[0], r)
            self.scc_add_overflow(x, self.cu(k & 0xFFFFFFFF), r)
        elif op in ("v_add_co_u32", "v_add_co_ci_u32", "v_sub_co_u32", "v_sub_co_ci_u32", "v_subrev_co_u32",
                    "v_subrev_co_ci_u32"):
            # Lane masks are all-ones/zero: carry in = (vcc or the named mask) != 0, carry out stored the same way.
            vd, cd = a[0], a[1]
            srcs = a[2:]
            x, y = self.src_u(srcs[0]), self.src_u(srcs[1])
            if op.startswith("v_subrev_co"):
                x, y = y, x
            sub = "sub" in op
            if op.endswith("ci_u32"):
                cin_m = self.mask_src(srcs[2]) if len(srcs) > 2 else self.mask_src({"k": "special", "t": "vcc"})
                cin = self.tmp()
                self.emit(f"{cin} = OpSelect %uint {self.tmp_bool_nz(cin_m)} {self.cu(1)} {self.cu(0)}")
            else:
                cin = self.cu(0)
            s1, c1, s2, c2, r, carry = (self.tmp() for _ in range(6))
            if sub:
                self.emit(f"{s1} = OpISubBorrow %umul {x} {y}")
            else:
                self.emit(f"{s1} = OpIAddCarry %umul {x} {y}")
            r1 = self.tmp()
            self.emit(f"{r1} = OpCompositeExtract %uint {s1} 0")
            self.emit(f"{c1} = OpCompositeExtract %uint {s1} 1")
            self.emit(f"{s2} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {r1} {cin}")
            self.emit(f"{r} = OpCompositeExtract %uint {s2} 0")
            self.emit(f"{c2} = OpCompositeExtract %uint {s2} 1")
            self.emit(f"{carry} = OpBitwiseOr %uint {c1} {c2}")
            self.dst(vd, r)
            m = self.tmp()
            self.emit(f"{m} = OpSelect %uint {self.tmp_bool_nz(carry)} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
            if not (cd["k"] == "special" and cd["t"] == "null"):
                self.mask_dst(cd, m)
        elif op == "v_mul_legacy_f32":
            # Legacy multiply treats 0 * anything (inf/NaN included) as 0.
            x, y = self.src_f(a[1]), self.src_f(a[2])
            m, zx, zy, either, r = (self.tmp() for _ in range(5))
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{zx} = OpFOrdEqual %bool {x} %f0")
            self.emit(f"{zy} = OpFOrdEqual %bool {y} %f0")
            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")
            self.emit(f"{r} = OpSelect %float {either} %f0 {m}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_mad_legacy_f32":
            # Legacy multiply treats 0 * anything (inf/NaN included) as 0.
            x, y, z = (self.src_f(t) for t in a[1:4])
            m, zx, zy, either, sel, r = (self.tmp() for _ in range(6))
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{zx} = OpFOrdEqual %bool {x} %f0")
            self.emit(f"{zy} = OpFOrdEqual %bool {y} %f0")
            self.emit(f"{either} = OpLogicalOr %bool {zx} {zy}")
            self.emit(f"{sel} = OpSelect %float {either} %f0 {m}")
            self.emit(f"{r} = OpFAdd %float {sel} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_min3_i32", "v_min3_u32", "v_max3_i32", "v_max3_u32"):
            x, y, z = (self.src_u(t) for t in a[1:4])
            f = ("S" if op.endswith("i32") else "U") + ("Min" if "min" in op else "Max")
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpExtInst %uint %glsl {f} {x} {y}")
            self.emit(f"{r} = OpExtInst %uint %glsl {f} {m} {z}")
            self.dst(a[0], r)
        elif op in ("v_med3_i32", "v_med3_u32"):
            x, y, z = (self.src_u(t) for t in a[1:4])
            s = "S" if op.endswith("i32") else "U"
            lo, hi, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{lo} = OpExtInst %uint %glsl {s}Min {x} {y}")
            self.emit(f"{hi} = OpExtInst %uint %glsl {s}Max {x} {y}")
            self.emit(f"{r} = OpExtInst %uint %glsl {s}Clamp {z} {lo} {hi}")
            self.dst(a[0], r)
        elif op.startswith("s_load_dword") or op.startswith("s_buffer_load_dword"):
            self.scalar_load(op, a, ins["mods"])
        elif op in ("s_lshl_b64", "s_lshr_b64", "s_ashr_i64"):
            # 64-bit shift of an SGPR pair by (src1 & 63); SCC = result != 0.
            d, x = a[0], a[1]
            if d["k"] != "sgpr" or d.get("n") != 2:
                raise Unsupported(op + " destination")
            if x["k"] == "sgpr" and x.get("n") == 2:
                lo, hi = self.load_reg("sgpr", x["r"]), self.load_reg("sgpr", x["r"] + 1)
            elif x["k"] == "imm":
                v = x["v"] & 0xFFFFFFFFFFFFFFFF
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " source")
            sh = self.src_u(a[2])
            l64, h64, t, v64, m, s64, r, rlo, rh, rhi = (self.tmp() for _ in range(10))
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{t} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{v64} = OpBitwiseOr %ulong {t} {l64}")
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(63)}")
            self.emit(f"{s64} = OpUConvert %ulong {m}")
            if op == "s_ashr_i64":
                vs64, rs64 = self.tmp(), self.tmp()
                self.emit(f"{vs64} = OpBitcast %long {v64}")
                self.emit(f"{rs64} = OpShiftRightArithmetic %long {vs64} {s64}")
                self.emit(f"{r} = OpBitcast %ulong {rs64}")
            elif op == "s_lshl_b64":
                self.emit(f"{r} = OpShiftLeftLogical %ulong {v64} {s64}")
            else:
                self.emit(f"{r} = OpShiftRightLogical %ulong {v64} {s64}")
            self.emit(f"{rlo} = OpUConvert %uint {r}")
            self.emit(f"{rh} = OpShiftRightLogical %ulong {r} %c64_32")
            self.emit(f"{rhi} = OpUConvert %uint {rh}")
            self.store_reg("sgpr", d["r"], rlo)
            self.store_reg("sgpr", d["r"] + 1, rhi)
            nz, z = self.tmp(), self.tmp()
            orr = self.tmp()
            self.emit(f"{orr} = OpBitwiseOr %uint {rlo} {rhi}")
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(orr)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
            self.uses_psb = True        # Int64 capability and %ulong come with the PSB declarations
        elif op == "s_ashr_i32":
            x, sh = self.src_u(a[1]), self.src_u(a[2])
            m, xi, ri, r = (self.tmp() for _ in range(4))
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{xi} = OpBitcast %int {x}")
            self.emit(f"{ri} = OpShiftRightArithmetic %int {xi} {m}")
            self.emit(f"{r} = OpBitcast %uint {ri}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op.startswith(("global_load_dword", "global_store_dword")) or \
                op in ("global_load_ushort", "global_load_sshort", "global_load_ubyte", "global_load_sbyte"):
            self.global_mem(op, a, mods)
        elif op in ("s_add_u32", "s_addc_u32", "s_sub_u32", "s_subb_u32"):
            # 32-bit add/sub with carry/borrow through SCC (used for 64-bit address arithmetic, e.g. s_getpc + offset).
            x, y = self.src_u(a[1]), self.src_u(a[2])
            sub = op.startswith("s_sub")
            s1, r1, c1 = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s1} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {x} {y}")
            self.emit(f"{r1} = OpCompositeExtract %uint {s1} 0")
            self.emit(f"{c1} = OpCompositeExtract %uint {s1} 1")
            r, carry = r1, c1
            if op in ("s_addc_u32", "s_subb_u32"):
                cin, s2, r2, c2, cc = (self.tmp() for _ in range(5))
                self.emit(f"{cin} = OpLoad %uint %scc")
                self.emit(f"{s2} = {'OpISubBorrow' if sub else 'OpIAddCarry'} %umul {r1} {cin}")
                self.emit(f"{r2} = OpCompositeExtract %uint {s2} 0")
                self.emit(f"{c2} = OpCompositeExtract %uint {s2} 1")
                self.emit(f"{cc} = OpBitwiseOr %uint {c1} {c2}")
                r, carry = r2, cc
            self.dst(a[0], r)
            self.emit(f"OpStore %scc {carry}")
        elif op in ("s_lshl1_add_u32", "s_lshl2_add_u32", "s_lshl3_add_u32", "s_lshl4_add_u32"):
            # D = (S0 << N) + S1. SCC = unsigned overflow of the sum taken in 64 bits (RDNA ISA pseudocode:
            # tmp = (64'U(S0) << N) + 64'U(S1); SCC = tmp >= 2^32), so bits shifted out of S0 count as a carry too.
            n = int(op[6])
            x, y = self.src_u(a[1]), self.src_u(a[2])
            sh, out, s, r, c, ovf = (self.tmp() for _ in range(6))
            self.emit(f"{sh} = OpShiftLeftLogical %uint {x} {self.cu(n)}")
            self.emit(f"{out} = OpShiftRightLogical %uint {x} {self.cu(32 - n)}")
            self.emit(f"{s} = OpIAddCarry %umul {sh} {y}")
            self.emit(f"{r} = OpCompositeExtract %uint {s} 0")
            self.emit(f"{c} = OpCompositeExtract %uint {s} 1")
            self.emit(f"{ovf} = OpBitwiseOr %uint {out} {c}")
            self.dst(a[0], r)
            self.scc_nonzero(ovf)
        elif op == "s_get_waveid_in_workgroup":
            # D = this wave's index in its workgroup: thread i of a group is in wave i / W (W = 64 or 32), the rule
            # wave_lane uses for the lane. Only compute and mesh shaders have a workgroup. SCC is not written.
            if self.stage not in ("cs", "ms"):
                raise Unsupported(op + " outside compute")
            r = self.tmp()
            self.emit(f"{r} = OpShiftRightLogical %uint {self.local_index()} {self.cu(6 if self.wave64 else 5)}")
            self.dst(a[0], r)
        elif op == "s_getpc_b64":
            # Address of the next instruction: code base (push constants 34-35: where the runtime put this shader's
            # FSSM blob in the Xbox VA window) + offset of the next instruction within the blob.
            if a[0]["k"] != "sgpr" or a[0].get("n") != 2:
                raise Unsupported("s_getpc_b64 destination")
            nxt = ins["pc"] + len(ins["enc"]) // 2
            lo, hi, s, rlo, c, rhi = (self.tmp() for _ in range(6))
            self.emit(f"{lo} = OpLoad %uint %ud_code_lo_p")
            self.emit(f"{hi} = OpLoad %uint %ud_code_hi_p")
            self.emit(f"{s} = OpIAddCarry %umul {lo} {self.cu(nxt)}")
            self.emit(f"{rlo} = OpCompositeExtract %uint {s} 0")
            self.emit(f"{c} = OpCompositeExtract %uint {s} 1")
            self.emit(f"{rhi} = OpIAdd %uint {hi} {c}")
            self.store_reg("sgpr", a[0]["r"], rlo)
            self.store_reg("sgpr", a[0]["r"] + 1, rhi)
            self.uses_codebase = True
            self.cu(3), self.cu(4)
        elif op in ("s_add_i32", "s_sub_i32", "s_min_u32", "s_max_u32", "s_xor_b32", "s_andn2_b32",
                    "s_cselect_b32", "s_orn2_b32", "s_nand_b32", "s_nor_b32", "s_xnor_b32"):
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            if op == "s_cselect_b32":
                c, b = self.tmp(), self.tmp()
                self.emit(f"{c} = OpLoad %uint %scc")
                self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
                self.emit(f"{r} = OpSelect %uint {b} {x} {y}")
            elif op == "s_andn2_b32":
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {y}")
                self.emit(f"{r} = OpBitwiseAnd %uint {x} {n}")
            elif op in ("s_min_u32", "s_max_u32"):
                self.emit(f"{r} = OpExtInst %uint %glsl {'UMin' if 'min' in op else 'UMax'} {x} {y}")
            elif op == "s_orn2_b32":
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {y}")
                self.emit(f"{r} = OpBitwiseOr %uint {x} {n}")
            elif op in ("s_nand_b32", "s_nor_b32", "s_xnor_b32"):
                t = self.tmp()
                k = {"s_nand_b32": "OpBitwiseAnd", "s_nor_b32": "OpBitwiseOr", "s_xnor_b32": "OpBitwiseXor"}[op]
                self.emit(f"{t} = {k} %uint {x} {y}")
                self.emit(f"{r} = OpNot %uint {t}")
            else:
                k = {"s_add_i32": "OpIAdd", "s_sub_i32": "OpISub", "s_sub_u32": "OpISub",
                     "s_xor_b32": "OpBitwiseXor"}[op]
                self.emit(f"{r} = {k} %uint {x} {y}")
            self.dst(a[0], r)
            # SCC (RDNA ISA): bitwise ops -> result != 0; s_add/sub_i32 -> signed overflow;
            # s_min_u32 -> S0 < S1, s_max_u32 -> S0 > S1; s_cselect leaves SCC.
            if op in ("s_xor_b32", "s_andn2_b32", "s_orn2_b32", "s_nand_b32", "s_nor_b32", "s_xnor_b32"):
                self.scc_logic(a, r)
            elif op in ("s_add_i32", "s_sub_i32"):
                self.scc_add_overflow(x, y, r, sub=op == "s_sub_i32")
            elif op in ("s_min_u32", "s_max_u32"):
                b = self.tmp()
                self.emit(f"{b} = {'OpULessThan' if 'min' in op else 'OpUGreaterThan'} %bool {x} {y}")
                self.set_scc(b)
        elif op in ("s_wqm_b64", "s_wqm_b32"):
            self.wqm(op, a)
        elif op == "v_fma_mix_f32":
            # Each source is f32, or (op_sel_hi bit set) an f16 from the low/high half chosen by op_sel.
            vals = self.fma_mix_srcs(a, mods)
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl Fma {vals[0]} {vals[1]} {vals[2]}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_mad_f32", "v_mac_f32"):
            # Unfused: the product is rounded before the add. v_mac_f32 is the VOP2 form with D as the addend
            # (LLVM SIInstrInfo.cpp getNewFMAInst: V_MAC_F32 -> V_MAD_F32_e64, V_FMAC_F32 -> V_FMA_F32_e64).
            if op == "v_mac_f32":
                x, y, z = self.src_f(a[1]), self.src_f(a[2]), self.src_f(a[0])
            else:
                x, y, z = (self.src_f(t) for t in a[1:4])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpFMul %float {x} {y}")
            self.emit(f"{r} = OpFAdd %float {m} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_mul_lo_u32", "v_min_u32", "v_max_u32", "v_min_i32", "v_max_i32"):
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            if op == "v_mul_lo_u32":
                self.emit(f"{r} = OpIMul %uint {x} {y}")
            else:
                f = {"v_min_u32": "UMin", "v_max_u32": "UMax", "v_min_i32": "SMin", "v_max_i32": "SMax"}[op]
                self.emit(f"{r} = OpExtInst %uint %glsl {f} {x} {y}")
            self.dst(a[0], r)
        elif op == "v_cvt_f32_f16":
            u = self.src_h(a[1])
            v2, f = self.tmp(), self.tmp()
            self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
            self.emit(f"{f} = OpCompositeExtract %float {v2} 0")
            self.dst(a[0], self.tou(self.clamp_mods(self.modify_f(f, a[1]), mods)))
        elif op in ("v_interp_p1_f32", "v_interp_p2_f32", "v_interp_mov_f32"):
            # Single-lane model: the barycentric p1/p2 pair yields the interpolated attribute, and the mov of P0
            # the provoking vertex's value. An input only ever read through the mov is declared Flat: such
            # inputs carry integers (instance and material ids, buffer indices), and interpolating their bits
            # as floats does not return them intact. No shader in this build reads one location both ways.
            attr = next((x for x in a if x["k"] == "attr"), None)
            if attr is None or self.stage != "ps":
                raise Unsupported(op + " form")
            f = self.attribute(attr["a"], attr["c"], flat=op == "v_interp_mov_f32")
            if self.pixel_interpolants is not None and (self.pixel_interpolants[1] >> attr["a"]) & 1:
                # 16-bit interpolant (FSSM table flag bit 4) read through a 32-bit op: the stored value only
                # has f16 precision, so round to nearest f16 and back. Reads through the *_f16 opcodes already
                # narrow to f16 and are unchanged.
                f = self.f16_roundtrip(f)
            self.dst(a[0], self.tou(f))
        elif op == "s_movk_i32":
            v = a[1]["v"] & 0xFFFF
            self.dst(a[0], self.cu(v - 0x10000 if v & 0x8000 else v))
        elif op in ("s_cmov_b32", "s_cmovk_i32"):
            # D = SCC ? S0 : D (s_cmovk_i32: S0 is the 16-bit immediate, sign-extended). SCC is read, not written.
            if a[0]["k"] != "sgpr":
                raise Unsupported(op + " destination")
            if op == "s_cmovk_i32":
                v = a[1]["v"] & 0xFFFF
                x = self.cu((v - 0x10000 if v & 0x8000 else v) & 0xFFFFFFFF)
            else:
                x = self.src_u(a[1])
            old, c, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{old} = OpCopyObject %uint {self.load_reg('sgpr', a[0]['r'])}")
            self.emit(f"{c} = OpLoad %uint %scc")
            self.emit(f"{r} = OpSelect %uint {self.tmp_bool_nz(c)} {x} {old}")
            self.dst(a[0], r)
        elif op.startswith("v_cmp_") and not op.startswith("v_cmp_class") and not op.endswith(("_u64", "_i64")):
            self.compare(op, a, scalar=False)
        elif op.startswith("v_cmpx_") and not op.startswith("v_cmpx_class") and not op.endswith(("_u64", "_i64")):
            self.compare_x(op, a)
        elif op in ("s_and_saveexec_b64", "s_or_saveexec_b64", "s_andn2_saveexec_b64", "s_and_saveexec_b32",
                    "s_or_saveexec_b32", "s_andn2_saveexec_b32"):
            saved = self.exec_value()
            src = self.mask_src(a[1])
            r = self.tmp()
            kind = op[2:].split("_saveexec")[0]
            if kind == "andn2":
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {saved}")
                self.emit(f"{r} = OpBitwiseAnd %uint {src} {n}")
            else:
                self.emit(f"{r} = {'OpBitwiseAnd' if kind == 'and' else 'OpBitwiseOr'} %uint {src} {saved}")
            self.mask_dst(a[0], saved)
            self.set_exec(r)
            self.scc_from_mask(r)
        elif op in ("s_andn2_wrexec_b64", "s_andn2_wrexec_b32"):
            # RDNA ISA: EXEC = S0 & ~EXEC; D = EXEC; SCC = (EXEC != 0). Two operands: sdst, ssrc0.
            x, y = self.mask_src(a[1]), self.exec_value()
            n, r = self.tmp(), self.tmp()
            self.emit(f"{n} = OpNot %uint {y}")
            self.emit(f"{r} = OpBitwiseAnd %uint {x} {n}")
            self.set_exec(r)
            self.mask_dst(a[0], r)
            self.scc_from_mask(r)
        elif op in ("s_bfe_u64", "s_bfe_i64"):
            # 64-bit bitfield extract: offset = S1[5:0], width = S1[22:16] (0-127). A width of 0 gives 0; a field
            # running past bit 63 stops there (LLVM constantFoldBFE: Src0 >> Offset, arithmetic for i64), so the
            # width is cut to 64 - offset. SCC = (D != 0). The pair is 64-bit data (as for s_lshl_b64); a lane-mask
            # source stays Unsupported (#210).
            d, x = a[0], a[1]
            if d["k"] != "sgpr" or d.get("n") != 2 or self.holds_lane_mask(x):
                raise Unsupported(op + " operands")
            if x["k"] == "sgpr" and x.get("n") == 2:
                lo, hi = self.load_reg("sgpr", x["r"]), self.load_reg("sgpr", x["r"] + 1)
            elif x["k"] == "imm":
                v = x["v"] & 0xFFFFFFFFFFFFFFFF
                lo, hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " source")
            ctl = self.src_u(a[2])
            off, wid, room, fits, w, l64, h64, t, v64 = (self.tmp() for _ in range(9))
            self.emit(f"{off} = OpBitwiseAnd %uint {ctl} {self.cu(63)}")
            self.emit(f"{wid} = OpBitFieldUExtract %uint {ctl} {self.cu(16)} {self.cu(7)}")
            self.emit(f"{room} = OpISub %uint {self.cu(64)} {off}")
            self.emit(f"{fits} = OpULessThan %bool {wid} {room}")
            self.emit(f"{w} = OpSelect %uint {fits} {wid} {room}")              # 0..64, and off + w <= 64
            self.emit(f"{l64} = OpUConvert %ulong {lo}")
            self.emit(f"{h64} = OpUConvert %ulong {hi}")
            self.emit(f"{t} = OpShiftLeftLogical %ulong {h64} %c64_32")
            self.emit(f"{v64} = OpBitwiseOr %ulong {t} {l64}")
            # Shift the field to the top, then back down: both amounts stay in 0..63 whenever w >= 1; w = 0 is
            # handled by the final select, so its (clamped) shifts never decide the result.
            wz = self.tmp()
            self.emit(f"{wz} = OpIEqual %bool {w} {self.cu(0)}")
            w1, up, down, up64, down64, sh, field = (self.tmp() for _ in range(7))
            self.emit(f"{w1} = OpSelect %uint {wz} {self.cu(1)} {w}")
            ow = self.tmp()
            self.emit(f"{ow} = OpIAdd %uint {off} {w1}")
            self.emit(f"{up} = OpISub %uint {self.cu(64)} {ow}")
            self.emit(f"{down} = OpISub %uint {self.cu(64)} {w1}")
            self.emit(f"{up64} = OpUConvert %ulong {up}")
            self.emit(f"{down64} = OpUConvert %ulong {down}")
            self.emit(f"{sh} = OpShiftLeftLogical %ulong {v64} {up64}")
            if op == "s_bfe_u64":
                self.emit(f"{field} = OpShiftRightLogical %ulong {sh} {down64}")
            else:
                si_, ri = self.tmp(), self.tmp()
                self.emit(f"{si_} = OpBitcast %long {sh}")
                self.emit(f"{ri} = OpShiftRightArithmetic %long {si_} {down64}")
                self.emit(f"{field} = OpBitcast %ulong {ri}")
            flo, fh, fhi, rlo, rhi = (self.tmp() for _ in range(5))
            self.emit(f"{flo} = OpUConvert %uint {field}")
            self.emit(f"{fh} = OpShiftRightLogical %ulong {field} %c64_32")
            self.emit(f"{fhi} = OpUConvert %uint {fh}")
            self.emit(f"{rlo} = OpSelect %uint {wz} {self.cu(0)} {flo}")
            self.emit(f"{rhi} = OpSelect %uint {wz} {self.cu(0)} {fhi}")
            self.store_reg("sgpr", d["r"], rlo)
            self.store_reg("sgpr", d["r"] + 1, rhi)
            orr = self.tmp()
            self.emit(f"{orr} = OpBitwiseOr %uint {rlo} {rhi}")
            self.scc_nonzero(orr)
            self.uses_psb = True        # Int64 capability and %ulong / %long come with the PSB declarations
        elif op in ("s_bfe_u32", "s_bfe_i32"):
            # src1 packs offset in bits [4:0] and width in bits [22:16].
            x, ctl = self.src_u(a[1]), self.src_u(a[2])
            off, wid, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{off} = OpBitwiseAnd %uint {ctl} {self.cu(31)}")
            self.emit(f"{wid} = OpBitFieldUExtract %uint {ctl} {self.cu(16)} {self.cu(7)}")
            wid = self.bfe_width(off, wid)
            if op == "s_bfe_u32":
                self.emit(f"{r} = OpBitFieldUExtract %uint {x} {off} {wid}")
            else:
                xi, ri = self.as_int(x), self.tmp()
                self.emit(f"{ri} = OpBitFieldSExtract %int {xi} {off} {wid}")
                self.emit(f"{r} = OpBitcast %uint {ri}")
            self.dst(a[0], r)
            z = self.tmp()
            self.emit(f"{z} = OpSelect %uint {self.tmp_bool_nz(r)} {self.cu(1)} {self.cu(0)}")
            self.emit(f"OpStore %scc {z}")
        elif op == "v_perm_b32":
            # Byte permute: result byte i = selector byte i picks from the 8 bytes {src0, src1} (src1 is low).
            # Selector values 0-7 pick a byte; 8-11 repeat the sign bit of a 16-bit half (bit 15, 31, 47 or 63 of
            # {src0, src1}, RDNA2 ISA reference, V_PERM_B32); 12 -> 0x00, >=13 -> 0xFF.
            # Both candidate shifts are (k & 3) * 8, so every shift stays below 32 (SPIR-V leaves larger ones
            # undefined, even on the side of the OpSelect that is discarded).
            s0, s1, sel = (self.src_u(t) for t in a[1:4])
            out = self.cu(0)
            for i in range(4):
                sb, lo_pick, hi_pick, k, b, k3, sh, chosen, inrange, val, shifted, acc = \
                    (self.tmp() for _ in range(12))
                self.emit(f"{sb} = OpBitFieldUExtract %uint {sel} {self.cu(8 * i)} {self.cu(8)}")
                self.emit(f"{k} = OpBitwiseAnd %uint {sb} {self.cu(7)}")
                self.emit(f"{b} = OpULessThan %bool {k} {self.cu(4)}")
                self.emit(f"{k3} = OpBitwiseAnd %uint {k} {self.cu(3)}")
                self.emit(f"{sh} = OpIMul %uint {k3} {self.cu(8)}")
                self.emit(f"{lo_pick} = OpShiftRightLogical %uint {s1} {sh}")
                self.emit(f"{hi_pick} = OpShiftRightLogical %uint {s0} {sh}")
                self.emit(f"{chosen} = OpSelect %uint {b} {lo_pick} {hi_pick}")
                self.emit(f"{inrange} = OpULessThan %bool {sb} {self.cu(8)}")
                # 8-11: k3 = sb & 3 picks the half; its sign bit is bit 15 or 31 of src1 (k3 0, 1) or src0 (2, 3).
                k2, in_s1, sgn_src, sgn_sh, sgn_bit, sgn, sb_hi, is_sgn, const, ff, big = (self.tmp() for _ in range(11))
                self.emit(f"{k2} = OpBitwiseAnd %uint {k3} {self.cu(2)}")
                self.emit(f"{in_s1} = OpIEqual %bool {k2} {self.cu(0)}")
                self.emit(f"{sgn_src} = OpSelect %uint {in_s1} {s1} {s0}")
                self.emit(f"{sgn_sh} = OpBitFieldInsert %uint {self.cu(15)} {k3} {self.cu(4)} {self.cu(1)}")
                self.emit(f"{sgn_bit} = OpBitFieldUExtract %uint {sgn_src} {sgn_sh} {self.cu(1)}")
                self.emit(f"{sgn} = OpIMul %uint {sgn_bit} {self.cu(0xFF)}")
                self.emit(f"{sb_hi} = OpBitwiseAnd %uint {sb} {self.cu(0xFC)}")
                self.emit(f"{is_sgn} = OpIEqual %bool {sb_hi} {self.cu(8)}")
                self.emit(f"{big} = OpUGreaterThanEqual %bool {sb} {self.cu(13)}")
                self.emit(f"{const} = OpSelect %uint {big} {self.cu(0xFF)} {self.cu(0)}")
                self.emit(f"{ff} = OpSelect %uint {is_sgn} {sgn} {const}")
                self.emit(f"{val} = OpSelect %uint {inrange} {chosen} {ff}")
                m8 = self.tmp()
                self.emit(f"{m8} = OpBitwiseAnd %uint {val} {self.cu(0xFF)}")
                self.emit(f"{shifted} = OpShiftLeftLogical %uint {m8} {self.cu(8 * i)}")
                self.emit(f"{acc} = OpBitwiseOr %uint {out} {shifted}")
                out = acc
            self.dst(a[0], out)
        elif op in ("s_and_b64", "s_or_b64", "s_xor_b64", "s_andn2_b64", "s_orn2_b64",
                    "s_and_b32_mask", "s_or_b32_mask"):
            x, y = self.mask_src(a[1]), self.mask_src(a[2])
            r = self.tmp()
            if op.endswith("n2_b64"):
                n = self.tmp()
                self.emit(f"{n} = OpNot %uint {y}")
                y = n
            k = {"and": "OpBitwiseAnd", "or": "OpBitwiseOr", "xor": "OpBitwiseXor", "andn2": "OpBitwiseAnd",
                 "orn2": "OpBitwiseOr"}[op[2:].rsplit("_", 1)[0]]
            self.emit(f"{r} = {k} %uint {x} {y}")
            self.mask_dst(a[0], r)
            self.scc_from_mask(r)
        elif op == "s_cselect_b64":
            def is_mask_op(x):
                return self.holds_lane_mask(x) or (
                    x["k"] == "special" and x.get("t", "").startswith(("exec", "vcc"))
                )
            if (a[0]["k"] == "special" and a[0].get("t", "").startswith(("exec", "vcc"))) or \
                    is_mask_op(a[1]) or is_mask_op(a[2]):
                x, y = self.mask_src(a[1]), self.mask_src(a[2])
                c, b, r = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{c} = OpLoad %uint %scc")
                self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
                self.emit(f"{r} = OpSelect %uint {b} {x} {y}")
                self.mask_dst(a[0], r)
            else:
                if a[0]["k"] != "sgpr" or a[0].get("n") != 2:
                    raise Unsupported("s_cselect_b64 destination")
                def src64(s):
                    if s["k"] == "sgpr" and s.get("n") == 2:
                        return self.load_reg("sgpr", s["r"]), self.load_reg("sgpr", s["r"] + 1)
                    if s["k"] == "imm":
                        v = s["v"] & 0xFFFFFFFFFFFFFFFF
                        return self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
                    if s["k"] == "special" and s.get("t") == "null":
                        return self.cu(0), self.cu(0)
                    raise Unsupported("s_cselect_b64 source")
                # Load all source dwords before storing to a[0] so destination aliasing stays exact;
                # lane_masks.py already clears a[0]'s mask state when either source is non-mask.
                (lo1, hi1), (lo2, hi2) = src64(a[1]), src64(a[2])
                c, b, rlo, rhi = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{c} = OpLoad %uint %scc")
                self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
                self.emit(f"{rlo} = OpSelect %uint {b} {lo1} {lo2}")
                self.emit(f"{rhi} = OpSelect %uint {b} {hi1} {hi2}")
                self.store_reg("sgpr", a[0]["r"], rlo)
                self.store_reg("sgpr", a[0]["r"] + 1, rhi)
        elif op in ("v_interp_p1ll_f16", "v_interp_p1lv_f16", "v_interp_p2_f16"):
            attr = next((x for x in a if x["k"] == "attr"), None)
            if attr is None or self.stage != "ps":
                raise Unsupported(op + " form")
            f = self.attribute(attr["a"], attr["c"])
            if op == "v_interp_p2_f16":
                # VOP3 clamp applies to the f16 result (ACO: dst(F16); llvm-mc takes clamp but no omod on this op).
                # Until 2026-11-22 it was dropped.
                self.dst(a[0], self.f16_bits(self.clamp_mods(f, mods)))
            else:   # p1ll/p1lv produce the f32 intermediate consumed by p2_f16
                self.dst(a[0], self.tou(f))
        elif op.startswith("s_cmp_"):
            self.compare(op, a, scalar=True)
        elif op.startswith("s_cmpk_"):
            # s_cmpk_<cond>_<ty> sX, imm16: zero-extended for u32, sign-extended for i32.
            ty = op.rsplit("_", 1)[1]
            v = a[1]["v"] & 0xFFFF
            if ty == "i32" and v & 0x8000:
                v -= 0x10000
            self.compare("s_cmp_" + op[len("s_cmpk_"):], [a[0], {"k": "imm", "v": v}], scalar=True)
        elif op.startswith("s_bitcmp"):
            self.bitcmp(op, a)
        elif op == "s_mul_hi_u32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            s, r = self.tmp(), self.tmp()
            self.emit(f"{s} = OpUMulExtended %umul {x} {y}")
            self.emit(f"{r} = OpCompositeExtract %uint {s} 1")
            self.dst(a[0], r)
        elif op == "s_mul_hi_i32":
            # High 32 bits of the signed 64-bit product; SCC is not written.
            x, y = self.as_int(self.src_u(a[1])), self.as_int(self.src_u(a[2]))
            s, hi, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s} = OpSMulExtended %smul {x} {y}")
            self.emit(f"{hi} = OpCompositeExtract %int {s} 1")
            self.emit(f"{r} = OpBitcast %uint {hi}")
            self.uses_smul = True
            self.dst(a[0], r)
        elif op == "s_absdiff_i32":
            # RDNA2 ISA: D = S0 - S1 as a 32-bit signed value, negated if negative; SCC = (D != 0). The difference
            # wraps before the absolute value (0x7FFFFFFF - 0x80000000 = -1 -> 1), and SAbs keeps 0x80000000 as is.
            x, y = self.src_u(a[1]), self.src_u(a[2])
            d, ab, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{d} = OpISub %uint {x} {y}")
            self.emit(f"{ab} = OpExtInst %int %glsl SAbs {self.as_int(d)}")
            self.emit(f"{r} = OpBitcast %uint {ab}")
            self.dst(a[0], r)
            self.scc_nonzero(r)
        elif op in ("v_fma_mixlo_f16", "v_fma_mixhi_f16"):
            vals = self.fma_mix_srcs(a, mods)
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl Fma {vals[0]} {vals[1]} {vals[2]}")
            r = self.clamp_mods(r, mods)
            self.f16_out(a[0], r, hi=(op == "v_fma_mixhi_f16"))
        elif op == "v_cvt_pk_u8_f32":
            # Convert src0 to u8 (round to nearest even, clamp) and insert it at byte src1 of src2.
            x, byte, base = self.src_f(a[1]), self.src_u(a[2]), self.src_u(a[3])
            c, rn, u, sh, mask, nmask, keep, ins_, r = (self.tmp() for _ in range(9))
            self.emit(f"{c} = OpExtInst %float %glsl FClamp {x} %f0 {self.fconst(255.0)}")
            self.emit(f"{rn} = OpExtInst %float %glsl RoundEven {c}")
            self.emit(f"{u} = OpConvertFToU %uint {rn}")
            b3 = self.tmp()
            self.emit(f"{b3} = OpBitwiseAnd %uint {byte} {self.cu(3)}")
            self.emit(f"{sh} = OpIMul %uint {b3} {self.cu(8)}")
            self.emit(f"{mask} = OpShiftLeftLogical %uint {self.cu(0xFF)} {sh}")
            self.emit(f"{nmask} = OpNot %uint {mask}")
            self.emit(f"{keep} = OpBitwiseAnd %uint {base} {nmask}")
            self.emit(f"{ins_} = OpShiftLeftLogical %uint {u} {sh}")
            self.emit(f"{r} = OpBitwiseOr %uint {keep} {ins_}")
            self.dst(a[0], r)
        elif op == "v_mul_hi_u32":
            x, y = self.src_u(a[1]), self.src_u(a[2])
            s, r = self.tmp(), self.tmp()
            self.emit(f"{s} = OpUMulExtended %umul {x} {y}")
            self.emit(f"{r} = OpCompositeExtract %uint {s} 1")
            self.dst(a[0], r)
        elif op in ("v_cvt_pk_u16_u32", "v_cvt_pk_i16_i32"):
            # D = {sat16(S1), sat16(S0)} (RDNA ISA pseudocode: u32_to_u16 / i32_to_i16 saturate): the low half from
            # S0, the high half from S1; u16 clamps to [0, 0xFFFF], i16 to [-32768, 32767]. The whole dword is
            # written. clamp is not modelled: rejected rather than guessed.
            if "clamp" in mods:
                raise Unsupported(op + " clamp")
            halves = []
            for src in a[1:3]:
                x, h = self.src_u(src), self.tmp()
                if op == "v_cvt_pk_u16_u32":
                    self.emit(f"{h} = OpExtInst %uint %glsl UMin {x} {self.cu(0xFFFF)}")
                else:
                    xi, c, cu_ = self.as_int(x), self.tmp(), self.tmp()
                    self.emit(f"{c} = OpExtInst %int %glsl SClamp {xi} {self.as_int(self.cu(0xFFFF8000))} "
                              f"{self.as_int(self.cu(0x7FFF))}")
                    self.emit(f"{cu_} = OpBitcast %uint {c}")
                    self.emit(f"{h} = OpBitwiseAnd %uint {cu_} {self.cu(0xFFFF)}")
                halves.append(h)
            sh, r = self.tmp(), self.tmp()
            self.emit(f"{sh} = OpShiftLeftLogical %uint {halves[1]} {self.cu(16)}")
            self.emit(f"{r} = OpBitwiseOr %uint {halves[0]} {sh}")
            self.dst(a[0], r)
        elif op == "v_cvt_pkrtz_f16_f32":
            # Packs two f32 as f16 into one dword with round-toward-zero (exact, see f16_rtz). The earlier
            # round-to-nearest version produced inf where hardware gives 65504 (found by the GPU smoke test).
            x, y = self.src_f(a[1]), self.src_f(a[2])
            lo, hi = self.f16_rtz(x), self.f16_rtz(y)
            sh, r = self.tmp(), self.tmp()
            self.emit(f"{sh} = OpShiftLeftLogical %uint {hi} {self.cu(16)}")
            self.emit(f"{r} = OpBitwiseOr %uint {lo} {sh}")
            self.dst(a[0], r)
        elif op == "v_cvt_off_f32_i4":
            # RDNA ISA: D = CVT_OFF_TABLE[S0[3:0]], i.e. the 4-bit signed value / 16 (-0.5 .. 0.4375), exact in f32.
            # Output modifiers (mul/div, clamp) apply to the result.
            x = self.src_u(a[1])
            i4, fi, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{i4} = OpBitFieldSExtract %uint {x} {self.cu(0)} {self.cu(4)}")
            self.emit(f"{fi} = OpConvertSToF %float {self.as_int(i4)}")
            self.emit(f"{r} = OpFMul %float {fi} {self.fconst(0.0625)}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_rcp_iflag_f32":
            x = self.src_f(a[1])
            r = self.tmp()
            self.emit(f"{r} = OpFDiv %float %f1 {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in F2:
            x, y = self.src_f(a[1]), self.src_f(a[2])
            r = self.tmp()
            if F2[op] in ("max", "min"):
                self.emit(f"{r} = OpExtInst %float %glsl {'FMax' if F2[op] == 'max' else 'FMin'} {x} {y}")
            else:
                self.emit(f"{r} = {F2[op]} %float {x} {y}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op == "v_subrev_f32":
            x, y = self.src_f(a[1]), self.src_f(a[2])
            r = self.tmp()
            self.emit(f"{r} = OpFSub %float {y} {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_fma_f32", "v_fmac_f32", "v_fmaak_f32", "v_fmamk_f32", "v_med3_f32"):
            if op in ("v_fma_f32", "v_fmaak_f32", "v_fmamk_f32"):
                # fmaak D = S0 * S1 + K, printed `vdst, src0, vsrc1, K`; fmamk D = S0 * K + S1, printed
                # `vdst, src0, K, vsrc1`. In print order both are fma(a[1], a[2], a[3]), like v_fma_f32. Until
                # 2026-11-21 fmamk read a[3] as the factor, so it computed S0 * S1 + K.
                x, y, z = (self.src_f(t) for t in a[1:4])
            elif op == "v_fmac_f32":
                x, y, z = self.src_f(a[1]), self.src_f(a[2]), self.src_f(a[0])
            else:
                x, y, z = (self.src_f(t) for t in a[1:4])
                lo, hi, r = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{lo} = OpExtInst %float %glsl FMin {x} {y}")
                self.emit(f"{hi} = OpExtInst %float %glsl FMax {x} {y}")
                self.emit(f"{r} = OpExtInst %float %glsl FClamp {z} {lo} {hi}")
                self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
                return
            r = self.tmp()
            self.emit(f"{r} = OpExtInst %float %glsl Fma {x} {y} {z}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in F1:
            x = self.src_f(a[1])
            r = self.tmp()
            if F1[op] == "rcp":
                self.emit(f"{r} = OpFDiv %float %f1 {x}")
            else:
                self.emit(f"{r} = OpExtInst %float %glsl {F1[op]} {x}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_cvt_f32_u32", "v_cvt_f32_i32"):
            x = self.src_u(a[1])
            r = self.tmp()
            self.emit(f"{r} = {'OpConvertUToF' if op.endswith('u32') else 'OpConvertSToF'} %float "
                      f"{x if op.endswith('u32') else self.as_int(x)}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))   # VOP3: clamp, omod
        elif op.startswith("v_cvt_f32_ubyte"):
            n = int(op[-1])
            x = self.src_u(a[1])
            s, m, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{s} = OpShiftRightLogical %uint {x} {self.cu(8 * n)}")
            self.emit(f"{m} = OpBitwiseAnd %uint {s} {self.cu(0xFF)}")
            self.emit(f"{r} = OpConvertUToF %float {m}")
            self.dst(a[0], self.tou(self.clamp_mods(r, mods)))
        elif op in ("v_cvt_flr_i32_f32", "v_cvt_rpi_i32_f32"):
            # RDNA ISA pseudocode: flr = f32_to_i32(floor(S0)), rpi = f32_to_i32(floor(S0 + 0.5)), the + 0.5 rounded
            # in binary32 like any f32 add. f32_to_i32 saturates as v_cvt_i32_f32 does (#300): +-2^31 and beyond go to
            # INT_MAX / INT_MIN, NaN to 0. Source neg/abs apply before the rounding.
            x = self.src_f(a[1])
            if op == "v_cvt_rpi_i32_f32":
                t = self.tmp()
                self.emit(f"{t} = OpFAdd %float {x} {self.fconst(0.5)}")
                x = t
            fl = self.tmp()
            self.emit(f"{fl} = OpExtInst %float %glsl Floor {x}")
            bits = self.tou(fl)
            magnitude, negative, not_nan, bound, in_range, saturated, safe, signed, converted, result = \
                (self.tmp() for _ in range(10))
            self.emit(f"{magnitude} = OpBitwiseAnd %uint {bits} {self.cu(0x7FFFFFFF)}")
            self.emit(f"{in_range} = OpULessThan %bool {magnitude} {self.cu(0x4F000000)}")  # |x| < 2^31
            self.emit(f"{negative} = OpINotEqual %bool {bits} {magnitude}")
            self.emit(f"{not_nan} = OpULessThan %bool {magnitude} {self.cu(0x7F800001)}")
            self.emit(f"{bound} = OpSelect %uint {negative} {self.cu(0x80000000)} {self.cu(0x7FFFFFFF)}")
            self.emit(f"{saturated} = OpSelect %uint {not_nan} {bound} {self.cu(0)}")
            # Convert only an in-range operand: SPIR-V leaves the out-of-range conversion undefined.
            self.emit(f"{safe} = OpSelect %uint {in_range} {bits} {self.cu(0)}")
            self.emit(f"{signed} = OpConvertFToS %int {self.tof(safe)}")
            self.emit(f"{converted} = OpBitcast %uint {signed}")
            self.emit(f"{result} = OpSelect %uint {in_range} {converted} {saturated}")
            self.dst(a[0], result)
        elif op in ("v_cvt_u32_f32", "v_cvt_i32_f32"):
            # Compare binary32 words before converting: the raw SPIR-V conversion
            # must only see a representable operand, even when its result is discarded.
            bits = self.src_u(a[1])
            zero = self.cu(0)
            past_inf = self.cu(0x7F800001)
            in_range, saturated, safe, converted, result = (self.tmp() for _ in range(5))
            if op == "v_cvt_u32_f32":
                positive = self.tmp()
                self.emit(f"{in_range} = OpULessThan %bool {bits} {self.cu(0x4F800000)}")  # +2^32
                # Positive overflow through +inf saturates. Negative values and NaNs give zero.
                self.emit(f"{positive} = OpULessThan %bool {bits} {past_inf}")
                self.emit(f"{saturated} = OpSelect %uint {positive} {self.cu(0xFFFFFFFF)} {zero}")
            else:
                magnitude, negative, not_nan, bound = (self.tmp() for _ in range(4))
                self.emit(f"{magnitude} = OpBitwiseAnd %uint {bits} {self.cu(0x7FFFFFFF)}")
                self.emit(f"{in_range} = OpULessThan %bool {magnitude} {self.cu(0x4F000000)}")  # 2^31
                self.emit(f"{negative} = OpINotEqual %bool {bits} {magnitude}")
                self.emit(f"{not_nan} = OpULessThan %bool {magnitude} {past_inf}")
                self.emit(f"{bound} = OpSelect %uint {negative} {self.cu(0x80000000)} {self.cu(0x7FFFFFFF)}")
                self.emit(f"{saturated} = OpSelect %uint {not_nan} {bound} {zero}")
            self.emit(f"{safe} = OpSelect %uint {in_range} {bits} {zero}")
            x = self.tof(safe)
            if op == "v_cvt_u32_f32":
                self.emit(f"{converted} = OpConvertFToU %uint {x}")
            else:
                signed = self.tmp()
                self.emit(f"{signed} = OpConvertFToS %int {x}")
                self.emit(f"{converted} = OpBitcast %uint {signed}")
            self.emit(f"{result} = OpSelect %uint {in_range} {converted} {saturated}")
            self.dst(a[0], result)
        elif op in U2 or op in S2:
            x, y = self.src_u(a[1]), self.src_u(a[2])
            r = self.tmp()
            kind = U2.get(op) or S2.get(op)
            if kind == "mul24":
                xm, ym = self.tmp(), self.tmp()
                self.emit(f"{xm} = OpBitwiseAnd %uint {x} {self.cu(0xFFFFFF)}")
                self.emit(f"{ym} = OpBitwiseAnd %uint {y} {self.cu(0xFFFFFF)}")
                self.emit(f"{r} = OpIMul %uint {xm} {ym}")
            else:
                if op in ("s_lshl_b32", "s_lshr_b32"):
                    m = self.tmp()
                    self.emit(f"{m} = OpBitwiseAnd %uint {y} {self.cu(31)}")
                    y = m
                self.emit(f"{r} = {kind} %uint {x} {y}")
            self.dst(a[0], r)
            if op in ("s_and_b32", "s_or_b32"):
                self.scc_logic(a, r)
            elif op in ("s_lshl_b32", "s_lshr_b32"):
                self.scc_nonzero(r)
        elif op in SHIFT_REV:
            sh, x = self.src_u(a[1]), self.src_u(a[2])
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {sh} {self.cu(31)}")
            self.emit(f"{r} = {SHIFT_REV[op]} %uint {x} {m}")
            self.dst(a[0], r)
        elif op in ("v_bfe_u32", "v_bfe_i32"):
            # offset and width are taken mod 32 (a width of 0 gives 0)
            x, o, w = (self.src_u(t) for t in a[1:4])
            om, wm, r = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{om} = OpBitwiseAnd %uint {o} {self.cu(31)}")
            self.emit(f"{wm} = OpBitwiseAnd %uint {w} {self.cu(31)}")
            wc = self.bfe_width(om, wm)
            f = "OpBitFieldSExtract" if op.endswith("i32") else "OpBitFieldUExtract"
            self.emit(f"{r} = {f} %uint {x} {om} {wc}")
            self.dst(a[0], r)
        elif op in ("v_mad_u64_u32", "v_mad_i64_i32"):
            # {sdst, v[d:d+1]} = S0 * S1 + S2 as a 65-bit result (RDNA ISA pseudocode): 32 x 32 -> 64-bit product,
            # unsigned or signed, plus the 64-bit S2. sdst (a lane mask) gets bit 64: the carry out for u64; for i64
            # the sign of the exact sum, which a 65-bit two's-complement value always holds.
            if a[0].get("n") != 2 or "clamp" in mods:
                raise Unsupported(op + " operands")
            vd, sd, s0, s1, s2 = a[0], a[1], a[2], a[3], a[4]
            signed = op == "v_mad_i64_i32"
            if s2["k"] in ("vgpr", "sgpr") and s2.get("n") == 2:
                c_lo, c_hi = self.load_reg(s2["k"], s2["r"]), self.load_reg(s2["k"], s2["r"] + 1)
            elif s2["k"] == "imm":
                v = s2["v"] & 0xFFFFFFFFFFFFFFFF          # 64-bit inline/literal constant: sign-extended from 32 bits
                c_lo, c_hi = self.cu(v & 0xFFFFFFFF), self.cu(v >> 32)
            else:
                raise Unsupported(op + " operands")
            x, y = self.src_u(s0), self.src_u(s1)
            pm, p_lo, p_hi = self.tmp(), self.tmp(), self.tmp()
            if signed:
                ph = self.tmp()
                self.emit(f"{pm} = OpSMulExtended %smul {self.as_int(x)} {self.as_int(y)}")
                pl = self.tmp()
                self.emit(f"{pl} = OpCompositeExtract %int {pm} 0")
                self.emit(f"{ph} = OpCompositeExtract %int {pm} 1")
                self.emit(f"{p_lo} = OpBitcast %uint {pl}")
                self.emit(f"{p_hi} = OpBitcast %uint {ph}")
                self.uses_smul = True
            else:
                self.emit(f"{pm} = OpUMulExtended %umul {x} {y}")
                self.emit(f"{p_lo} = OpCompositeExtract %uint {pm} 0")
                self.emit(f"{p_hi} = OpCompositeExtract %uint {pm} 1")
            s_lo, r_lo, c0, s_h1, h1, c1, s_h2, r_hi, c2 = (self.tmp() for _ in range(9))
            self.emit(f"{s_lo} = OpIAddCarry %umul {p_lo} {c_lo}")
            self.emit(f"{r_lo} = OpCompositeExtract %uint {s_lo} 0")
            self.emit(f"{c0} = OpCompositeExtract %uint {s_lo} 1")
            self.emit(f"{s_h1} = OpIAddCarry %umul {p_hi} {c_hi}")
            self.emit(f"{h1} = OpCompositeExtract %uint {s_h1} 0")
            self.emit(f"{c1} = OpCompositeExtract %uint {s_h1} 1")
            self.emit(f"{s_h2} = OpIAddCarry %umul {h1} {c0}")
            self.emit(f"{r_hi} = OpCompositeExtract %uint {s_h2} 0")
            self.emit(f"{c2} = OpCompositeExtract %uint {s_h2} 1")
            if signed:
                # Sign of the exact sum: the operands' common sign if they agree, else the wrapped sum's sign.
                sp, sc, sr, same, neg = (self.tmp() for _ in range(5))
                self.emit(f"{sp} = OpShiftRightLogical %uint {p_hi} {self.cu(31)}")
                self.emit(f"{sc} = OpShiftRightLogical %uint {c_hi} {self.cu(31)}")
                self.emit(f"{sr} = OpShiftRightLogical %uint {r_hi} {self.cu(31)}")
                self.emit(f"{same} = OpIEqual %bool {sp} {sc}")
                self.emit(f"{neg} = OpSelect %uint {same} {sp} {sr}")
                bit64 = neg
            else:
                bit64 = self.tmp()
                self.emit(f"{bit64} = OpBitwiseOr %uint {c1} {c2}")
            self.dst({"k": "vgpr", "r": vd["r"], "n": 1}, r_lo)
            self.dst({"k": "vgpr", "r": vd["r"] + 1, "n": 1}, r_hi)
            if not (sd["k"] == "special" and sd["t"] == "null"):
                m = self.tmp()
                self.emit(f"{m} = OpSelect %uint {self.tmp_bool_nz(bit64)} {self.cu(0xFFFFFFFF)} {self.cu(0)}")
                self.mask_dst(sd, m)
        elif op in ("v_mad_u32_u16", "v_mad_i32_i16"):
            # D = S0.half * S1.half + S2 (RDNA ISA pseudocode), wrapping at 32 bits: the 16-bit halves zero-extended
            # (u16) or sign-extended (i16); op_sel bits 0 / 1 read the high half of S0 / S1. S2 and D are whole
            # dwords, so an op_sel bit on them, and clamp, are not modelled: rejected rather than guessed.
            sel, dhi = self.opsel_bits(mods, 3)
            if sel[2] or dhi or "clamp" in mods:
                raise Unsupported(op + " op_sel/clamp")
            ext = "OpBitFieldSExtract" if op == "v_mad_i32_i16" else "OpBitFieldUExtract"
            hs = []
            for i, src in enumerate(a[1:3]):
                h = self.tmp()
                self.emit(f"{h} = {ext} %uint {self.src_u(src)} {self.cu(16 * sel[i])} {self.cu(16)}")
                hs.append(h)
            m, r = self.tmp(), self.tmp()
            self.emit(f"{m} = OpIMul %uint {hs[0]} {hs[1]}")
            self.emit(f"{r} = OpIAdd %uint {m} {self.src_u(a[3])}")
            self.dst(a[0], r)
        elif op == "v_mad_u32_u24":
            x, y, z = (self.src_u(t) for t in a[1:4])
            xm, ym, p, r = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{xm} = OpBitwiseAnd %uint {x} {self.cu(0xFFFFFF)}")
            self.emit(f"{ym} = OpBitwiseAnd %uint {y} {self.cu(0xFFFFFF)}")
            self.emit(f"{p} = OpIMul %uint {xm} {ym}")
            self.emit(f"{r} = OpIAdd %uint {p} {z}")
            self.dst(a[0], r)
        elif op == "v_cndmask_b32":
            # e32: dst, src0, src1 (vcc) ; e64: dst, src0, src1, sN[:N+1]/vcc
            x, y = self.src_u(a[1]), self.src_u(a[2])
            c = self.mask_src(a[3]) if len(a) > 3 else self.mask_src({"k": "special", "t": "vcc"})
            b, r = self.tmp(), self.tmp()
            self.emit(f"{b} = OpINotEqual %bool {c} {self.cu(0)}")
            self.emit(f"{r} = OpSelect %uint {b} {y} {x}")
            self.dst(a[0], r)
        elif op == "exp":
            self.export(ins)
        else:
            raise Unsupported(op)

    def scalar_load(self, op, a, mods):
        """s_load_dwordxN sD, s[p:p+1], off   -> N dwords at (64-bit pointer + off)
        s_buffer_load_dwordxN sD, s[v:v+3], off -> N dwords at (V# base address + off)
        Both become PhysicalStorageBuffer loads (Vulkan buffer device address). V# base = dword0 | (dword1 & 0xFFFF) << 32
        (AMD RDNA ISA, buffer resource descriptor).
        off is an immediate or a scalar register. An SMEM load can carry both, which llvm-mc prints as
        "sOFF offset:IMM" (the immediate arrives in mods); the address is then base + sOFF + IMM. Measured on an
        RX 6800 (gfx1030) for s_buffer_load_dword/x2/x4/x8: soffset 8 + imm 4 reads +12. s_load_dword* shares the
        encoding and gets the same rule (not measured). The immediate is zero-extended from 32 bits exactly as the
        immediate-only form is.
        Not modelled: for s_buffer_load* the hardware was measured to truncate base, sOFF and IMM each to a multiple
        of 4 and to return 0 past num_records (docs/design.md, Scalar/buffer memory)."""
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        d, base = a[0], a[1]
        # A single dword may also land in vcc_lo / m0 / vcc_hi (seen: s_buffer_load_dword vcc_lo, ...).
        single_special = (n == 1 and d["k"] == "special" and d["t"] in ("vcc_lo", "vcc", "m0", "vcc_hi")) or \
            (n == 2 and d["k"] == "special" and d["t"] == "vcc")
        if (d["k"] != "sgpr" and not single_special) or base["k"] != "sgpr":
            raise Unsupported(f"{op} operands {d.get('t', d['k'])} {base.get('t', base['k'])}")
        lo, hi = self.load_reg("sgpr", base["r"]), self.load_reg("sgpr", base["r"] + 1)
        if op.startswith("s_buffer"):
            if base.get("n") != 4:
                raise Unsupported("s_buffer_load without V#")
            m = self.tmp()
            self.emit(f"{m} = OpBitwiseAnd %uint {hi} {self.cu(0xFFFF)}")
            hi = m
        else:
            if base.get("n") != 2:
                raise Unsupported("s_load without 64-bit pointer")
        off = a[2] if len(a) > 2 else {"k": "imm", "v": 0}
        inst = [m.split(":", 1)[1] for m in mods if m.startswith("offset:")]
        if len(a) > 3 or len(inst) > 1 or (inst and off["k"] == "imm"):
            raise Unsupported(f"{op} offset form")      # not something llvm-mc prints: don't guess an address
        imm = None
        if inst:
            try:
                imm = int(inst[0], 0)
            except ValueError:
                raise Unsupported(f"{op} offset:{inst[0]}") from None
        offv = self.src_u(off)
        l64, h64, sh, addr, o64, full = (self.tmp() for _ in range(6))
        self.emit(f"{l64} = OpUConvert %ulong {lo}")
        self.emit(f"{h64} = OpUConvert %ulong {hi}")
        self.emit(f"{sh} = OpShiftLeftLogical %ulong {h64} %c64_32")
        self.emit(f"{addr} = OpBitwiseOr %ulong {sh} {l64}")
        self.emit(f"{o64} = OpUConvert %ulong {offv}")
        self.emit(f"{full} = OpIAdd %ulong {addr} {o64}")
        if imm is not None:                             # register offset and immediate: both are added
            i64, both = self.tmp(), self.tmp()
            self.emit(f"{i64} = OpUConvert %ulong {self.cu(imm)}")
            self.emit(f"{both} = OpIAdd %ulong {full} {i64}")
            full = both
        ptr = self.tmp()
        self.emit(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(full)}")
        self.uses_psb = True
        for i in range(n):
            p, v = self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
            self.emit(f"{v} = OpLoad %uint {p} Aligned 4")
            if single_special:
                if n == 2:          # vcc pair: low dword -> vcc_lo, high dword -> vcc_hi
                    self.dst({"k": "special", "t": "vcc_lo" if i == 0 else "vcc_hi"}, v)
                else:
                    self.dst(d, v)
            else:
                self.store_reg("sgpr", d["r"] + i, v)

    # ---- images: T#/S# -> bindless index through a runtime-filled hash table ----
    # Table layout (storage buffer, set 0 binding 2): uint mask; then entries of 10 uints:
    #   key[0..7] (descriptor dwords; S# uses key[0..3] and key[4..7] = 0xFFFFFFFF), index, pad.
    # Probes 8 slots from hash(key) & mask; a miss yields index 0 (the runtime puts a fallback there).
    PROBES = 8

    def desc_lookup(self, words):
        """words: 8 uint ids. Returns a uint id with the bindless index.

        The probe is about 400 instructions, so it is one function in the module (%desc_lookup_fn) that every
        image instruction calls. Inlined at each use it was most of the code of texture-heavy shaders: 194 image
        instructions made over half of a 7 MB module."""
        self.uses_tex = True
        if self.desc_fn is None:
            params = [f"%dl_w{i}" for i in range(8)]
            saved, self.body = self.body, []
            result = self.desc_probe(params)
            lines, self.body = self.body, saved
            self.desc_fn = (["%desc_lookup_fn = OpFunction %uint None %fn_desc_lookup"] +
                            [f"{p} = OpFunctionParameter %uint" for p in params] + ["%dl_entry = OpLabel"] +
                            lines + [f"         OpReturnValue {result}", "OpFunctionEnd"])
        r = self.tmp()
        self.emit(f"{r} = OpFunctionCall %uint %desc_lookup_fn " + " ".join(words))
        return r

    def desc_probe(self, words):
        """The lookup itself: FNV-1a of the 8 dwords, then PROBES slots compared in turn."""
        h = self.cu(2166136261)
        for w in words:                      # FNV-1a over the 8 dwords
            x, m = self.tmp(), self.tmp()
            self.emit(f"{x} = OpBitwiseXor %uint {h} {w}")
            self.emit(f"{m} = OpIMul %uint {x} {self.cu(16777619)}")
            h = m
        mp, mask = self.tmp(), self.tmp()
        self.emit(f"{mp} = OpAccessChain %_ptr_SB_uint %desc_map %c0 %c0")
        self.emit(f"{mask} = OpLoad %uint {mp}")
        # One slot per turn of a loop that stops at the first match, not PROBES copies of the comparison. A shader
        # compiler inlines this function at every use, and the unrolled copies were 95% of its work on a shader
        # with 330 lookups (117 s to compile with 8 slots unrolled, 14 s with one, 6 s with none).
        kv, iv = "%dl_k", "%dl_idx"                  # slot number, and the result (0 until a slot matches)
        self.body[0:0] = [f"         {kv} = OpVariable %_ptr_Function_uint Function %c0",
                          f"         {iv} = OpVariable %_ptr_Function_uint Function %c0"]
        self.emit("OpBranch %dl_head")
        self.body.append("%dl_head = OpLabel")
        self.emit("OpLoopMerge %dl_end %dl_cont None")
        self.emit("OpBranch %dl_body")
        self.body.append("%dl_body = OpLabel")
        k, s, slot, base = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{k} = OpLoad %uint {kv}")
        self.emit(f"{s} = OpIAdd %uint {h} {k}")
        self.emit(f"{slot} = OpBitwiseAnd %uint {s} {mask}")
        self.emit(f"{base} = OpIMul %uint {slot} {self.cu(10)}")
        match = "%true"
        for j, w in enumerate(words):
            o, p, v, e, m2 = self.tmp(), self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{o} = OpIAdd %uint {base} {self.cu(1 + j)}")
            self.emit(f"{p} = OpAccessChain %_ptr_SB_uint %desc_map %c0 {o}")
            self.emit(f"{v} = OpLoad %uint {p}")
            self.emit(f"{e} = OpIEqual %bool {v} {w}")
            self.emit(f"{m2} = OpLogicalAnd %bool {match} {e}")
            match = m2
        o, p, v = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{o} = OpIAdd %uint {base} {self.cu(9)}")
        self.emit(f"{p} = OpAccessChain %_ptr_SB_uint %desc_map %c0 {o}")
        self.emit(f"{v} = OpLoad %uint {p}")
        self.emit("OpSelectionMerge %dl_next None")
        self.emit(f"OpBranchConditional {match} %dl_hit %dl_next")
        self.body.append("%dl_hit = OpLabel")
        self.emit(f"OpStore {iv} {v}")
        self.emit(f"OpStore {kv} {self.cu(self.PROBES)}")          # the first match ends the search
        self.emit("OpBranch %dl_next")
        self.body.append("%dl_next = OpLabel")
        self.emit("OpBranch %dl_cont")
        self.body.append("%dl_cont = OpLabel")
        k2, k3, more, r = self.tmp(), self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{k2} = OpLoad %uint {kv}")
        self.emit(f"{k3} = OpIAdd %uint {k2} {self.cu(1)}")
        self.emit(f"OpStore {kv} {k3}")
        self.emit(f"{more} = OpULessThan %bool {k3} {self.cu(self.PROBES)}")
        self.emit(f"OpBranchConditional {more} %dl_head %dl_end")
        self.body.append("%dl_end = OpLabel")
        self.emit(f"{r} = OpLoad %uint {iv}")
        return r

    def desc_cache_ok(self):
        """True when the next instruction is emitted at the top level of the basic block, outside any selection or
        loop construct an earlier instruction opened and has not yet reached the merge label of. An id defined
        there dominates everything after it in the block. One defined inside a construct would not dominate the
        code after the merge, so lookups made there are not cached. The body is append-only within a block, so
        this only reads the lines added since the last call."""
        body, seen, open_ = self._scan
        if body is not self.body:
            body, seen, open_ = self.body, 0, []
        for line in body[seen:]:
            s = line.strip()
            if s.startswith(("OpSelectionMerge ", "OpLoopMerge ")):
                open_.append(s.split()[1])
            elif s.endswith("= OpLabel"):
                name = s.split()[0]
                if name in open_:
                    del open_[open_.index(name):]
        self._scan = (body, len(body), open_)
        return not open_

    # SQ_IMG_SAMP_WORD0 address modes (Mesa src/amd/registers/gfx10-rsrc.json: CLAMP_X = bits [2:0], CLAMP_Y = [5:3];
    # values: SQ_TEX_* in Mesa's gfx6.json, as RADV's radv_tex_wrap maps VkSamplerAddressMode onto them): 0 wrap, 1 mirror, 2 clamp last texel, 3 mirror once last texel,
    # 4 clamp half border, 5 mirror once half border, 6 clamp border, 7 mirror once border.
    def gather_addr_modes(self, s_op):
        """(x, y) address modes of an S# for the gather emulation, each either a Python int (the S# word 0 is a
        literal an s_mov_b32 wrote earlier in this basic block) or the id of a uint computed from the S# at run time.

        A literal S# whose mode needs a border colour (4..7), or that sets FORCE_UNNORMALIZED, is Unsupported: the
        four fetches can't reproduce it. A run-time S# can't be judged at translation time, so modes 4..7 are
        treated as clamp-to-edge there, and FORCE_UNNORMALIZED (bit 15) is not applied (the two remaining
        approximations)."""
        if s_op["k"] != "sgpr" or s_op.get("n", 1) < 4:
            raise Unsupported("descriptor operand")
        lit = self.sgpr_imm.get(s_op["r"])
        if lit is not None:
            if lit & 0x8000:
                raise Unsupported("image_gather4_l with FORCE_UNNORMALIZED")
            m = (lit & 7, lit >> 3 & 7)
            if max(m) > 3:
                raise Unsupported(f"image_gather4_l address mode {max(m)} (border)")
            return m
        w0 = self.load_reg("sgpr", s_op["r"])
        out = []
        for k in range(2):
            m = self.tmp()
            self.emit(f"{m} = OpBitFieldUExtract %uint {w0} {self.cu(3 * k)} {self.cu(3)}")
            out.append(m)
        return tuple(out)

    def gather_addr(self, c, n, mode):
        """Texel coordinate c (int id) after the address mode (int, or a uint id) for an axis of n texels (int id):
        wrap: c mod n; mirror: c mod 2n, reflected in the upper half; mirror once: the negative side reflected
        (-1 - c), then clamped; otherwise (clamp, and border modes at run time) clamped to [0, n-1]. As in the
        Vulkan address-mode definitions (and RADV's radv_tex_wrap mapping of them onto SQ_TEX_*). A literal mode
        emits only the code it needs; a run-time one emits all four and selects."""
        zero, one = self.as_int(self.cu(0)), self.as_int(self.cu(1))
        static = isinstance(mode, int)
        want = {mode if mode <= 3 else 2} if static else {0, 1, 2, 3}
        nm1 = self.tmp()
        self.emit(f"{nm1} = OpISub %int {n} {one}")
        res = {}
        if 2 in want:
            res[2] = self.tmp()
            self.emit(f"{res[2]} = OpExtInst %int %glsl SClamp {c} {zero} {nm1}")
        if 0 in want:
            res[0] = self.tmp()
            self.emit(f"{res[0]} = OpSMod %int {c} {n}")
        if 1 in want:
            n2, p, hi, rf, lt, res[1] = (self.tmp() for _ in range(6))
            self.emit(f"{n2} = OpIAdd %int {n} {n}")
            self.emit(f"{p} = OpSMod %int {c} {n2}")
            self.emit(f"{hi} = OpISub %int {n2} {one}")
            self.emit(f"{rf} = OpISub %int {hi} {p}")
            self.emit(f"{lt} = OpSLessThan %bool {p} {n}")
            self.emit(f"{res[1]} = OpSelect %int {lt} {p} {rf}")
        if 3 in want:
            neg, refl, ge, mo, res[3] = (self.tmp() for _ in range(5))
            self.emit(f"{neg} = OpISub %int {zero} {c}")
            self.emit(f"{refl} = OpISub %int {neg} {one}")
            self.emit(f"{ge} = OpSGreaterThanEqual %bool {c} {zero}")
            self.emit(f"{mo} = OpSelect %int {ge} {c} {refl}")
            self.emit(f"{res[3]} = OpExtInst %int %glsl SClamp {mo} {zero} {nm1}")
        if static:
            return res[mode if mode <= 3 else 2]
        r = res[2]
        for val in (3, 1, 0):
            eq, nr = self.tmp(), self.tmp()
            self.emit(f"{eq} = OpIEqual %bool {mode} {self.cu(val)}")
            self.emit(f"{nr} = OpSelect %int {eq} {res[val]} {r}")
            r = nr
        return r

    def desc_lookup_sgpr(self, a, n):
        """desc_lookup for an SGPR descriptor range (n=8: T#, n=4: S#), cached for the rest of the basic block.

        The cache holds the id of the call's result per (first SGPR, n). It is dropped by store_reg when any
        SGPR of the range is written (all writes to the %sgpr array go through store_reg, including the 64-bit
        ones, which are two calls, s_load results, v_readlane/v_readfirstlane results and the spare slots lane
        masks are parked in) and cleared when a basic block starts. Only lookups made at the block's top level
        are remembered, see desc_cache_ok. Returns the bindless index (a uint id)."""
        if a["k"] != "sgpr" or a.get("n", 1) < n:
            raise Unsupported("descriptor operand")
        key = (a["r"], n)
        hit = self.desc_cache.get(key)
        if hit is not None:
            return hit
        top = self.desc_cache_ok()
        words = self.sgpr_words(a, n)
        if n < 8:
            words = words + [self.cu(0xFFFFFFFF)] * (8 - n)
        idx = self.desc_lookup(words)
        if top:
            self.desc_cache[key] = idx
        return idx

    def sgpr_words(self, a, n):
        if a["k"] != "sgpr" or a.get("n", 1) < n:
            raise Unsupported("descriptor operand")
        return [self.load_reg("sgpr", a["r"] + i) for i in range(n)]

    def addr_regs(self, a):
        """Image address operand: contiguous v[x:y] or an NSA list."""
        if a["k"] == "vgpr":
            return [a["r"] + i for i in range(a.get("n", 1))]
        if a["k"] == "vlist":
            return a["r"]
        raise Unsupported("image address operand")

    def cube_coords(self, f):
        """(s, t, face) of a cube sample -> the 6-layer 2D-array view. The hardware takes s and t in [1, 2]
        (sc / |cubema| + 1.5) and the third coordinate as 8 * slice + face."""
        s, t, q, fl, d, layer = (self.tmp() for _ in range(6))
        self.emit(f"{s} = OpFSub %float {f[0]} %f1")
        self.emit(f"{t} = OpFSub %float {f[1]} %f1")
        self.emit(f"{q} = OpFMul %float {f[2]} {self.fconst(0.125)}")
        self.emit(f"{fl} = OpExtInst %float %glsl Floor {q}")
        self.emit(f"{d} = OpFMul %float {fl} {self.fconst(2.0)}")
        self.emit(f"{layer} = OpFSub %float {f[2]} {d}")
        return [s, t, layer] + f[3:]

    def image_op(self, op, a, mods):
        a16 = "a16" in mods
        d16 = "d16" in mods
        dim = next((m for m in mods if m.startswith("dim:")), "dim:SQ_RSRC_IMG_2D")
        kinds = {"dim:SQ_RSRC_IMG_2D": ("img2d", "%images", "simg2d", 2, "_ptr_UC_img"),
                 "dim:SQ_RSRC_IMG_2D_ARRAY": ("img2da", "%images_arr", "simg2da", 3, "_ptr_UC_imga"),
                 # Cube: the shader already did face selection (v_cube*), and cube_coords maps (s, t, face) onto a
                 # 6-layer 2D array view of the cube. The runtime binds cube textures as 2D-array views.
                 "dim:SQ_RSRC_IMG_CUBE": ("img2da", "%images_arr", "simg2da", 3, "_ptr_UC_imga"),
                 "dim:SQ_RSRC_IMG_1D": ("img1d", "%images_1d", "simg1d", 1, "_ptr_UC_img1"),
                 "dim:SQ_RSRC_IMG_3D": ("img3d", "%images_3d", "simg3d", -3, "_ptr_UC_img3")}
        if dim not in kinds:
            raise Unsupported(op + " " + dim.split("SQ_RSRC_IMG_")[-1])
        itype, iarr, stype, ncoord, iptr = kinds[dim]
        is3d = ncoord == -3          # 3D: three real coordinates (and 3-component gradients)
        if is3d:
            ncoord = 3
            self.uses_tex_3d = True
        elif ncoord == 3:
            self.uses_tex_array = True
        if ncoord == 1:
            self.uses_tex_1d = True
        if is3d and op.endswith("_d"):
            raise Unsupported(op + " 3D gradients")
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x1")
        dmask = int(dm.split(":")[1], 0)
        vdata, addr = a[0], self.addr_regs(a[1])
        img_idx = self.desc_lookup_sgpr(a[2], 8)
        ip, img = self.tmp(), self.tmp()
        self.emit(f"{ip} = OpAccessChain %{iptr} {iarr} {img_idx}")
        self.emit(f"{img} = OpLoad %{itype} {ip}")
        coords = [self.load_reg("vgpr", r) for r in addr]
        if a16:
            # GFX10 A16 packs only the coordinates, and the LOD or clamp after them, as f16 pairs (low half first).
            # The extra arguments before them keep a whole VGPR each, and so do non-G16 gradients (LLVM AMDGPU
            # getAddrSizeMIMGOp: the extra arguments' words, the gradients, then the packed halves; llvm-mc wants
            # v[0:2] for a 2D image_sample_c_b ... a16). A bias is an f16 in the low half of its VGPR, a z-compare
            # an f32. Until 2026-11-21 those were unpacked as pairs too, and the coordinates after them were read a
            # half early. A texel offset comes first; samples with one stop under a16 further down.
            extras = op[len("image_sample"):] if op.startswith("image_sample") else ""
            w_off = 1 if extras.endswith("_o") else 0
            extras = extras[:-2] if w_off else extras
            w_bias = 1 if extras in ("_b", "_c_b") else 0
            w_grad = 2 * min(ncoord, 2) if extras in ("_d", "_d_cl") else 0
            whole = w_off + w_bias + (1 if extras.startswith("_c") else 0) + w_grad
            unpacked = []
            for i, c in enumerate(coords):
                if i < whole and not w_off <= i < w_off + w_bias:
                    unpacked.append(c)            # offset, z-compare or gradient: the whole word
                    continue
                v2 = self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {c}")
                for k in ((0,) if i < whole else (0, 1)):   # a bias is the low half only
                    f = self.tmp()
                    self.emit(f"{f} = OpCompositeExtract %float {v2} {k}")
                    unpacked.append(self.tou(f))
            coords = unpacked
            if op == "image_load":
                # 16-bit integer coordinates: each half is an unsigned 16-bit int, not an f16.
                ints = []
                for c in [self.load_reg("vgpr", r) for r in addr]:
                    lo, hi = self.tmp(), self.tmp()
                    self.emit(f"{lo} = OpBitwiseAnd %uint {c} {self.cu(0xFFFF)}")
                    self.emit(f"{hi} = OpShiftRightLogical %uint {c} {self.cu(16)}")
                    ints += [lo, hi]
                coords = ints
        if op == "image_load":
            if len(coords) < ncoord:
                raise Unsupported("image_load coords")
            ints = [self.as_int(c) for c in coords[:ncoord]]
            if ncoord == 1:
                iv = ints[0]
            else:
                iv = self.tmp()
                self.emit(f"{iv} = OpCompositeConstruct {'%v2int' if ncoord == 2 else '%v3int'} {' '.join(ints)}")
            if dim in self.STORE_DIMS and not d16:
                self.image_load_typed(dim, self.load_reg("sgpr", a[2]["r"] + 1), iv, img, img_idx, dmask, vdata)
                return
            res = self.tmp()
            self.emit(f"{res} = OpImageFetch %v4float {img} {iv} Lod {self.cu(0)}")
            ncomp = bin(dmask).count("1")
        else:
            smp_idx = self.desc_lookup_sgpr(a[3], 4)
            sp, smp, si = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{sp} = OpAccessChain %_ptr_UC_smp %samplers {smp_idx}")
            self.emit(f"{smp} = OpLoad %smp {sp}")
            self.emit(f"{si} = OpSampledImage %{stype} {img} {smp}")
            f = [self.tof(c) for c in coords]
            goff = None
            if op in ("image_gather4_lz_o", "image_gather4_l_o"):
                # first address dword: signed 6-bit texel offsets x [5:0], y [13:8]
                if a16:
                    raise Unsupported(op + " a16")
                parts = []
                for k in range(2):
                    u, i = self.tmp(), self.tmp()
                    self.emit(f"{u} = OpBitFieldSExtract %uint {coords[0]} {self.cu(8 * k)} {self.cu(6)}")
                    self.emit(f"{i} = OpBitcast %int {u}")
                    parts.append(i)
                goff = self.tmp()
                self.emit(f"{goff} = OpCompositeConstruct %v2int {parts[0]} {parts[1]}")
                f = f[1:]
                op = op[:-2]
                self.uses_img_offset = True
            if op == "image_gather4_l":
                # Gather at an explicit LOD: Vulkan gathers have no Lod operand without an AMD extension, so fetch
                # the 2x2 footprint at mip round(lod) (clamped to the mip range) in gather order
                # (u0,v1) (u1,v1) (u1,v0) (u0,v0). Each texel coordinate goes through the S#'s own address mode
                # (CLAMP_X/CLAMP_Y, see gather_addr_modes / gather_addr).
                if bin(dmask).count("1") != 1 or ncoord != 2 or len(f) < 3 or d16:
                    raise Unsupported("image_gather4_l form")
                modes = self.gather_addr_modes(a[3])
                self.uses_query = True
                ch = dmask.bit_length() - 1
                lv, lvm, rl, mi, mc, mz, sz, uvf, szf, st, half, flo, i0, one2, i1 = (self.tmp() for _ in range(15))
                self.emit(f"{lv} = OpImageQueryLevels %int {img}")
                self.emit(f"{lvm} = OpISub %int {lv} {self.as_int(self.cu(1))}")
                self.emit(f"{rl} = OpExtInst %float %glsl RoundEven {f[2]}")
                self.emit(f"{mi} = OpConvertFToS %int {rl}")
                self.emit(f"{mz} = OpExtInst %int %glsl SMax {mi} {self.as_int(self.cu(0))}")
                self.emit(f"{mc} = OpExtInst %int %glsl SMin {mz} {lvm}")
                self.emit(f"{sz} = OpImageQuerySizeLod %v2int {img} {mc}")
                self.emit(f"{uvf} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
                self.emit(f"{szf} = OpConvertSToF %v2float {sz}")
                self.emit(f"{st} = OpFMul %v2float {uvf} {szf}")
                self.emit(f"{half} = OpCompositeConstruct %v2float %fhalf_g %fhalf_g")
                sub = self.tmp()
                self.emit(f"{sub} = OpFSub %v2float {st} {half}")
                self.emit(f"{flo} = OpExtInst %v2float %glsl Floor {sub}")
                self.emit(f"{i0} = OpConvertFToS %v2int {flo}")
                if goff is not None:
                    i0o = self.tmp()
                    self.emit(f"{i0o} = OpIAdd %v2int {i0} {goff}")
                    i0 = i0o
                c1 = self.as_int(self.cu(1))
                self.emit(f"{one2} = OpCompositeConstruct %v2int {c1} {c1}")
                self.emit(f"{i1} = OpIAdd %v2int {i0} {one2}")
                x0, y0, x1, y1 = (self.tmp() for _ in range(4))
                self.emit(f"{x0} = OpCompositeExtract %int {i0} 0")
                self.emit(f"{y0} = OpCompositeExtract %int {i0} 1")
                self.emit(f"{x1} = OpCompositeExtract %int {i1} 0")
                self.emit(f"{y1} = OpCompositeExtract %int {i1} 1")
                szx, szy = self.tmp(), self.tmp()
                self.emit(f"{szx} = OpCompositeExtract %int {sz} 0")
                self.emit(f"{szy} = OpCompositeExtract %int {sz} 1")
                ax = [self.gather_addr(c, szx, modes[0]) for c in (x0, x1)]
                ay = [self.gather_addr(c, szy, modes[1]) for c in (y0, y1)]
                comps = []
                for xx, yy in ((ax[0], ay[1]), (ax[1], ay[1]), (ax[1], ay[0]), (ax[0], ay[0])):
                    cc, tx, comp = self.tmp(), self.tmp(), self.tmp()
                    self.emit(f"{cc} = OpCompositeConstruct %v2int {xx} {yy}")
                    self.emit(f"{tx} = OpImageFetch %v4float {img} {cc} Lod {mc}")
                    self.emit(f"{comp} = OpCompositeExtract %float {tx} {ch}")
                    comps.append(comp)
                for i, comp in enumerate(comps):
                    self.dst({"k": "vgpr", "r": vdata["r"] + i, "n": 1}, self.tou(comp))
                self.uses_fhalf = True
                return
            if op == "image_gather4_lz":
                # Gather the 2x2 footprint of one channel (dmask picks it) at mip 0 -> 4 results.
                if bin(dmask).count("1") != 1 or ncoord not in (2, 3) or is3d or len(f) < ncoord:
                    raise Unsupported("image_gather4_lz form")
                if dim == "dim:SQ_RSRC_IMG_CUBE":
                    f = self.cube_coords(f)
                uv, res = self.tmp(), self.tmp()
                if ncoord == 3:          # 2D array (or cube as a 6-layer array): layer is the 3rd coordinate
                    self.emit(f"{uv} = OpCompositeConstruct %v3float {f[0]} {f[1]} {f[2]}")
                else:
                    self.emit(f"{uv} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
                self.emit(f"{res} = OpImageGather %v4float {si} {uv} {self.cu(dmask.bit_length() - 1)}"
                          + (f" Offset {goff}" if goff is not None else ""))
                comps = []
                for c in range(4):
                    comp = self.tmp()
                    self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                    comps.append(comp)
                if d16:
                    for i in range(0, 4, 2):
                        p = self.pack_f16_pair_int(comps[i], comps[i + 1])
                        self.dst({"k": "vgpr", "r": vdata["r"] + i // 2, "n": 1}, p)
                else:
                    for i, comp in enumerate(comps):
                        self.dst({"k": "vgpr", "r": vdata["r"] + i, "n": 1}, self.tou(comp))
                return
            kind = op[len("image_sample"):]          # '', '_lz', '_l', '_b', '_c_lz', '_d', '_d_cl' (+ '_o')
            off = None
            if kind.endswith("_o"):
                # _o: the first address dword packs signed 6-bit texel offsets, x = [5:0], y = [13:8], z = [21:16]
                # (GFX10 MIMG address order: offset, bias, z-compare, derivatives, coordinates). Mapped to the
                # SPIR-V Offset operand; Vulkan only guarantees offsets within minTexelOffset..maxTexelOffset.
                if a16:
                    raise Unsupported(op + " a16")
                kind = kind[:-2]
                raw, f = coords[0], f[1:]
                n_off = 3 if is3d else (1 if ncoord == 1 else 2)
                parts = []
                for k in range(n_off):
                    u, i = self.tmp(), self.tmp()
                    self.emit(f"{u} = OpBitFieldSExtract %uint {raw} {self.cu(8 * k)} {self.cu(6)}")
                    self.emit(f"{i} = OpBitcast %int {u}")
                    parts.append(i)
                if n_off == 1:
                    off = parts[0]
                else:
                    off = self.tmp()
                    self.emit(f"{off} = OpCompositeConstruct {'%v3int' if n_off == 3 else '%v2int'} {' '.join(parts)}")
                self.uses_img_offset = True
            minlod = None
            clamp = kind == "_d_cl"        # _d plus a min-LOD clamp, read after the gradients
            if clamp:
                kind = "_d"
                self.uses_minlod = True
            ref = None
            if kind.startswith("_c"):
                ref, f = f[0], f[1:]
            bias = None
            if kind in ("_b", "_c_b"):
                bias, f = f[0], f[1:]
            grads = None
            if kind == "_d":
                # Address: gradients first, then coords. 2D and 2D-array have 2-component gradients (the array
                # layer has none): ds/dx, dt/dx, ds/dy, dt/dy. 1D has scalar ds/dx, ds/dy.
                if ncoord == 1:
                    if len(f) < 3:
                        raise Unsupported(op + " coords")
                    grads, f = (f[0], f[1]), f[2:]
                else:
                    if len(f) < 4 + ncoord:
                        raise Unsupported(op + " coords")
                    gx, gy = self.tmp(), self.tmp()
                    self.emit(f"{gx} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
                    self.emit(f"{gy} = OpCompositeConstruct %v2float {f[2]} {f[3]}")
                    grads, f = (gx, gy), f[4:]
            if clamp:
                # The clamp is the component right after the coordinates. Under a16 an f16 pad may follow it, so
                # it isn't the last one there (taking the last one read that pad until 2026-11-21).
                if len(f) <= ncoord:
                    raise Unsupported(op + " coords")
                minlod, f = f[ncoord], f[:ncoord]
            if len(f) < ncoord:
                raise Unsupported(op + " coords")
            if dim == "dim:SQ_RSRC_IMG_CUBE":
                f = self.cube_coords(f)
            if ncoord == 1:
                uv = f[0]
            elif ncoord == 3:
                # The array layer arrives as a float coordinate; Vulkan rounds it the same way.
                uv = self.tmp()
                self.emit(f"{uv} = OpCompositeConstruct %v3float {f[0]} {f[1]} {f[2]}")
            else:
                uv = self.tmp()
                self.emit(f"{uv} = OpCompositeConstruct %v2float {f[0]} {f[1]}")
            res = self.tmp()
            if off is not None and kind not in ("", "_lz", "_l", "_b", "_c_lz"):
                raise Unsupported(op)

            def opnd(mask, args):
                """Image operands with the texel offset appended (Offset = 0x10 sorts after Lod/Bias)."""
                if off is None:
                    return f" {mask} {args}" if mask else ""
                return f" {mask + '|' if mask else ''}Offset {args + ' ' if args else ''}{off}"
            if kind == "_d":
                if minlod is not None:
                    self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv} Grad|MinLod {grads[0]} {grads[1]} {minlod}")
                else:
                    self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv} Grad {grads[0]} {grads[1]}")
            elif kind == "":
                self.emit(f"{res} = OpImageSampleImplicitLod %v4float {si} {uv}{opnd('', '')}" if self.stage == "ps" else
                          f"{res} = OpImageSampleExplicitLod %v4float {si} {uv}{opnd('Lod', '%f0')}")
            elif kind == "_lz":
                self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv}{opnd('Lod', '%f0')}")
            elif kind == "_l":
                self.emit(f"{res} = OpImageSampleExplicitLod %v4float {si} {uv}{opnd('Lod', f[ncoord] if len(f) > ncoord else '%f0')}")
            elif kind == "_b" and self.stage == "ps":
                self.emit(f"{res} = OpImageSampleImplicitLod %v4float {si} {uv}{opnd('Bias', bias)}")
            elif kind == "_c_lz":
                d = self.tmp()
                self.emit(f"{d} = OpImageSampleDrefExplicitLod %float {si} {uv} {ref}{opnd('Lod', '%f0')}")
                if d16:
                    # d16: the compare result is an f16 in the low half, as in the other d16 paths below. Until
                    # 2026-11-21 it was written as f32 bits, whose low half is 0 for every result (1.0 = 0x3f800000).
                    r = self.f16_bits(d)
                else:
                    r = self.tou(d)
                self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, r)
                return
            else:
                raise Unsupported(op)
            ncomp = bin(dmask).count("1")
        comps = []
        for c in range(4):
            if dmask >> c & 1:
                comp = self.tmp()
                self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                comps.append(comp)
        if len(comps) != ncomp:
            raise Unsupported("dmask")
        if d16:
            # d16: components are packed two per register as f16 (x|y<<16, z|w<<16).
            for i in range(0, len(comps), 2):
                second = comps[i + 1] if i + 1 < len(comps) else "%f0"
                p = self.pack_f16_pair_int(comps[i], second)
                self.dst({"k": "vgpr", "r": vdata["r"] + i // 2, "n": 1}, p)
        else:
            for i, comp in enumerate(comps):
                self.dst({"k": "vgpr", "r": vdata["r"] + i, "n": 1}, self.tou(comp))

    # image atomic -> (SPIR-V op, value that leaves memory unchanged, used for inactive lanes)
    IMG_ATOMICS = {"image_atomic_umin": ("OpAtomicUMin", 0xFFFFFFFF), "image_atomic_umax": ("OpAtomicUMax", 0),
                   "image_atomic_smin": ("OpAtomicSMin", 0x7FFFFFFF), "image_atomic_smax": ("OpAtomicSMax", 0x80000000),
                   "image_atomic_or": ("OpAtomicOr", 0), "image_atomic_xor": ("OpAtomicXor", 0),
                   "image_atomic_and": ("OpAtomicAnd", 0xFFFFFFFF), "image_atomic_add": ("OpAtomicIAdd", 0),
                   "image_atomic_sub": ("OpAtomicISub", 0)}

    # image_store views: dim -> (index, SPIR-V Dim, arrayed, coordinate count). Bindings 8 + 3 * index + class,
    # class 0 = float, 1 = uint, 2 = sint (Vulkan needs the shader's component type to match the image format).
    STORE_DIMS = {"dim:SQ_RSRC_IMG_2D": (0, "2D", 0, 2), "dim:SQ_RSRC_IMG_2D_ARRAY": (1, "2D", 1, 3),
                  "dim:SQ_RSRC_IMG_3D": (2, "3D", 0, 3), "dim:SQ_RSRC_IMG_1D": (3, "1D", 0, 1)}
    INT_FORMATS = None          # (uint mask words, sint mask words) over the 512 GFX10 formats

    @classmethod
    def int_format_masks(cls):
        if cls.INT_FORMATS is None:
            u, si = [0] * 16, [0] * 16
            for k, name in GFX10_FORMATS.items():
                if name.endswith("_UINT"):
                    u[k >> 5] |= 1 << (k & 31)
                elif name.endswith("_SINT"):
                    si[k >> 5] |= 1 << (k & 31)
            cls.INT_FORMATS = (u, si)
        return cls.INT_FORMATS

    def fmt_int_flags(self, word1):
        """(is UINT format, is SINT format) for a T# word1 (FORMAT [28:20]), from the 512-bit tables."""
        for w in [16] + sum(self.int_format_masks(), []):
            self.cu(w)
        fmt, word, bit = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{fmt} = OpBitFieldUExtract %uint {word1} {self.cu(20)} {self.cu(9)}")
        self.emit(f"{word} = OpShiftRightLogical %uint {fmt} {self.cu(5)}")
        self.emit(f"{bit} = OpBitwiseAnd %uint {fmt} {self.cu(31)}")
        flags = []
        for mask in ("%fmt_uint_mask", "%fmt_sint_mask"):
            p, w, b, nz = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{p} = OpAccessChain %_ptr_Private_uint {mask} {word}")
            self.emit(f"{w} = OpLoad %uint {p}")
            self.emit(f"{b} = OpBitFieldUExtract %uint {w} {bit} {self.cu(1)}")
            self.emit(f"{nz} = OpINotEqual %bool {b} {self.cu(0)}")
            flags.append(nz)
        return flags

    def image_load_typed(self, dim, t1, iv, img, idx, dmask, vdata):
        """image_load with the image's numeric type decided at run time (T# FORMAT): UINT/SINT formats are read
        through the uint/sint storage views (OpImageRead, read-without-format), everything else through the float
        sampled view (OpImageFetch, as before). Vulkan requires the view's component type to match the format, so a
        float view of an integer image would be undefined. Results are the raw bits of the dmask components.
        t1 is the T#'s second dword (FORMAT), idx the bindless index image_load already looked up."""
        di = self.STORE_DIMS[dim][0]
        self.uses_store_views.add(di)
        self.uses_store_reads = True
        is_u, is_s = self.fmt_int_flags(t1)
        act = self.active_b()
        L = {k: self.tmp() + "_" + k for k in ("s", "rest", "e2", "u", "f", "e3")}

        def write(v4u):
            k = 0
            for c in range(4):
                if dmask >> c & 1:
                    x = self.tmp()
                    self.emit(f"{x} = OpCompositeExtract %uint {v4u} {c}")
                    self.dst({"k": "vgpr", "r": vdata["r"] + k, "n": 1}, x)
                    k += 1

        def read(cls, ty):
            ip, im, r, b = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_stv{cls}_{di} %stv{cls}_{di} {idx}")
            self.emit(f"{im} = OpLoad %stimg{cls}_{di} {ip}")
            self.emit(f"{r} = OpImageRead {ty} {im} {iv}")
            self.emit(f"{b} = OpBitcast %v4uint_st {r}")
            write(b)

        _ = act
        self.emit(f"OpSelectionMerge {L['e2']} None")
        self.emit(f"OpBranchConditional {is_s} {L['s']} {L['rest']}")
        self.emit(f"{L['s']} = OpLabel")
        read(2, "%v4int_st")
        self.emit(f"OpBranch {L['e2']}")
        self.emit(f"{L['rest']} = OpLabel")
        self.emit(f"OpSelectionMerge {L['e3']} None")
        self.emit(f"OpBranchConditional {is_u} {L['u']} {L['f']}")
        self.emit(f"{L['u']} = OpLabel")
        read(1, "%v4uint_st")
        self.emit(f"OpBranch {L['e3']}")
        self.emit(f"{L['f']} = OpLabel")
        res, b = self.tmp(), self.tmp()
        self.emit(f"{res} = OpImageFetch %v4float {img} {iv} Lod {self.cu(0)}")
        self.emit(f"{b} = OpBitcast %v4uint_st {res}")
        write(b)
        self.emit(f"OpBranch {L['e3']}")
        self.emit(f"{L['e3']} = OpLabel")
        self.emit(f"OpBranch {L['e2']}")
        self.emit(f"{L['e2']} = OpLabel")

    def image_store(self, a, mods, dim):
        """image_store vData, vAddr, T#: write the dmask components to a storage image. The view is picked at run
        time from T# word1 FORMAT [28:20]: *_UINT -> uint view, *_SINT -> sint view, anything else -> float view
        (UNORM/SNORM/FLOAT convert from the float bits the shader wrote). Missing components are 0."""
        if dim not in self.STORE_DIMS or "d16" in mods:
            raise Unsupported("image_store " + dim.split("SQ_RSRC_IMG_")[-1] + (" a16" if "a16" in mods else "")
                              + (" d16" if "d16" in mods else ""))
        di, _, _, nc = self.STORE_DIMS[dim]
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0xf")
        ncomp = bin(int(dm.split(":")[1], 0)).count("1")
        vdata, addr = a[0], self.addr_regs(a[1])
        raw = [self.load_reg("vgpr", r) for r in addr]
        if "a16" in mods:
            # 16-bit coordinates: two unsigned 16-bit values per register, low half first.
            halves = []
            for c in raw:
                lo, hi = self.tmp(), self.tmp()
                self.emit(f"{lo} = OpBitwiseAnd %uint {c} {self.cu(0xFFFF)}")
                self.emit(f"{hi} = OpShiftRightLogical %uint {c} {self.cu(16)}")
                halves += [lo, hi]
            raw = halves
        if len(raw) < nc:
            raise Unsupported("image_store coords")
        vals = [self.load_reg("vgpr", vdata["r"] + i) for i in range(ncomp)] + [self.cu(0)] * (4 - ncomp)
        coords = [self.as_int(c) for c in raw[:nc]]
        idx = self.desc_lookup_sgpr(a[2], 8)
        t1 = self.load_reg("sgpr", a[2]["r"] + 1)
        self.uses_store_views.add(di)
        for w in [16] + sum(self.int_format_masks(), []):      # constants used by assemble's format masks
            self.cu(w)
        if nc == 1:
            iv = coords[0]
        else:
            iv = self.tmp()
            self.emit(f"{iv} = OpCompositeConstruct {'%v2int' if nc == 2 else '%v3int'} {' '.join(coords)}")
        is_u, is_s = self.fmt_int_flags(t1)
        fv = [self.tof(v) for v in vals]
        sv = [self.as_int(v) for v in vals]
        v4f, v4u, v4s = self.tmp(), self.tmp(), self.tmp()
        self.emit(f"{v4f} = OpCompositeConstruct %v4float {' '.join(fv)}")
        self.emit(f"{v4u} = OpCompositeConstruct %v4uint_st {' '.join(vals)}")
        self.emit(f"{v4s} = OpCompositeConstruct %v4int_st {' '.join(sv)}")
        act = self.active_b()
        L = {k: self.tmp() + "_" + k for k in ("do", "end", "s", "rest", "e2", "u", "f", "e3")}

        def write(cls, texel):
            ip, img = self.tmp(), self.tmp()
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_stv{cls}_{di} %stv{cls}_{di} {idx}")
            self.emit(f"{img} = OpLoad %stimg{cls}_{di} {ip}")
            self.emit(f"OpImageWrite {img} {iv} {texel}")

        self.emit(f"OpSelectionMerge {L['end']} None")
        self.emit(f"OpBranchConditional {act} {L['do']} {L['end']}")
        self.emit(f"{L['do']} = OpLabel")
        self.emit(f"OpSelectionMerge {L['e2']} None")
        self.emit(f"OpBranchConditional {is_s} {L['s']} {L['rest']}")
        self.emit(f"{L['s']} = OpLabel")
        write(2, v4s)
        self.emit(f"OpBranch {L['e2']}")
        self.emit(f"{L['rest']} = OpLabel")
        self.emit(f"OpSelectionMerge {L['e3']} None")
        self.emit(f"OpBranchConditional {is_u} {L['u']} {L['f']}")
        self.emit(f"{L['u']} = OpLabel")
        write(1, v4u)
        self.emit(f"OpBranch {L['e3']}")
        self.emit(f"{L['f']} = OpLabel")
        write(0, v4f)
        self.emit(f"OpBranch {L['e3']}")
        self.emit(f"{L['e3']} = OpLabel")
        self.emit(f"OpBranch {L['e2']}")
        self.emit(f"{L['e2']} = OpLabel")
        self.emit(f"OpBranch {L['end']}")
        self.emit(f"{L['end']} = OpLabel")

    def image_misc(self, op, a, mods):
        """image_get_resinfo / image_load_mip on 2D sampled images, image_atomic_umin on 2D r32ui storage images."""
        dim = next((m for m in mods if m.startswith("dim:")), "dim:SQ_RSRC_IMG_2D")
        is3d_atomic = dim == "dim:SQ_RSRC_IMG_3D" and (op in self.IMG_ATOMICS or op == "image_atomic_cmpswap")
        if op == "image_store":
            self.image_store(a, mods, dim)
            return
        if op == "image_load_mip" and dim == "dim:SQ_RSRC_IMG_3D" and "d16" not in mods and "a16" not in mods:
            # x, y, z, mip from a 3D sampled image (binding 6)
            vdata, addr = a[0], self.addr_regs(a[1])
            if len(addr) < 4:
                raise Unsupported("image_load_mip coords")
            idx = self.desc_lookup_sgpr(a[2], 8)
            self.uses_tex_3d = True
            ip, img, iv, res = self.tmp(), self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_img3 %images_3d {idx}")
            self.emit(f"{img} = OpLoad %img3d {ip}")
            xs = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:4]]
            self.emit(f"{iv} = OpCompositeConstruct %v3int {xs[0]} {xs[1]} {xs[2]}")
            self.emit(f"{res} = OpImageFetch %v4float {img} {iv} Lod {xs[3]}")
            dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x1")
            dmask, out = int(dm.split(":")[1], 0), 0
            for c in range(4):
                if dmask >> c & 1:
                    comp = self.tmp()
                    self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                    self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, self.tou(comp))
                    out += 1
            return
        if (dim != "dim:SQ_RSRC_IMG_2D" and not is3d_atomic) or "d16" in mods or "a16" in mods:
            raise Unsupported(op + " " + dim.split("SQ_RSRC_IMG_")[-1])
        vdata, addr = a[0], self.addr_regs(a[1])
        idx = self.desc_lookup_sgpr(a[2], 8)
        if is3d_atomic:
            # 3D r32ui storage image atomics (binding 7).
            self.uses_storage3d = True
            coords = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:3]]
            if len(coords) < 3:
                raise Unsupported(op + " coords")
            iv, ip, tp, old, safe = (self.tmp() for _ in range(5))
            self.emit(f"{iv} = OpCompositeConstruct %v3int {coords[0]} {coords[1]} {coords[2]}")
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_simg3 %simages3d {idx}")
            val = self.load_reg("vgpr", vdata["r"])
            if op == "image_atomic_cmpswap":
                cmpv = self.load_reg("vgpr", vdata["r"] + 1)
                act = self.active_b()
                do, end = self.tmp() + "_do", self.tmp() + "_end"
                self.emit(f"OpSelectionMerge {end} None")
                self.emit(f"OpBranchConditional {act} {do} {end}")
                self.emit(f"{do} = OpLabel")
                self.emit(f"{tp} = OpImageTexelPointer %_ptr_Image_uint {ip} {iv} {self.cu(0)}")
                self.emit(f"{old} = OpAtomicCompareExchange %uint {tp} {self.cu(1)} {self.cu(0)} {self.cu(0)} {val} {cmpv}")
                if "glc" in mods:
                    self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, old)
                self.emit(f"OpBranch {end}")
                self.emit(f"{end} = OpLabel")
                return
            spv, neutral = self.IMG_ATOMICS[op]
            self.emit(f"{tp} = OpImageTexelPointer %_ptr_Image_uint {ip} {iv} {self.cu(0)}")
            self.emit(f"{safe} = OpSelect %uint {self.active_b()} {val} {self.cu(neutral)}")
            self.emit(f"{old} = {spv} %uint {tp} {self.cu(1)} {self.cu(0)} {safe}")
            if "glc" in mods:
                self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, old)
            return
        if op == "image_store":
            self.image_store(a, mods, dim)
            return
        if op in self.IMG_ATOMICS or op == "image_atomic_cmpswap":
            # Storage image (2D r32ui). Atomics: vdata holds the operand; with 'glc' the pre-op value comes back.
            self.uses_storage = True
            coords = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:2]]
            iv = self.tmp()
            self.emit(f"{iv} = OpCompositeConstruct %v2int {coords[0]} {coords[1]}")
            ip = self.tmp()
            self.emit(f"{ip} = OpAccessChain %_ptr_UC_simg %simages {idx}")
            val = self.load_reg("vgpr", vdata["r"])
            if op == "image_store":
                # Writes dmask component 0 as r32ui; only active lanes store.
                img, lbl_do, lbl_end = self.tmp(), self.tmp() + "_do", self.tmp() + "_end"
                self.emit(f"{img} = OpLoad %simg2d_u {ip}")
                v4 = self.tmp()
                self.emit(f"{v4} = OpCompositeConstruct %v4uint {val} {self.cu(0)} {self.cu(0)} {self.cu(0)}")
                act = self.active_b()      # must be computed before OpSelectionMerge (merge directly precedes branch)
                self.emit(f"OpSelectionMerge {lbl_end} None")
                self.emit(f"OpBranchConditional {act} {lbl_do} {lbl_end}")
                self.emit(f"{lbl_do} = OpLabel")
                self.emit(f"OpImageWrite {img} {iv} {v4}")
                self.emit(f"OpBranch {lbl_end}")
                self.emit(f"{lbl_end} = OpLabel")
                self.uses_v4uint = True
                return
            if op == "image_atomic_cmpswap":
                # vdata = (src, cmp): memory = (memory == cmp) ? src : memory; glc returns the old value.
                cmpv = self.load_reg("vgpr", vdata["r"] + 1)
                act = self.active_b()
                do, end = self.tmp() + "_do", self.tmp() + "_end"
                self.emit(f"OpSelectionMerge {end} None")
                self.emit(f"OpBranchConditional {act} {do} {end}")
                self.emit(f"{do} = OpLabel")               # inactive lanes don't touch memory
                tp, old = self.tmp(), self.tmp()
                self.emit(f"{tp} = OpImageTexelPointer %_ptr_Image_uint {ip} {iv} {self.cu(0)}")
                self.emit(f"{old} = OpAtomicCompareExchange %uint {tp} {self.cu(1)} {self.cu(0)} {self.cu(0)} {val} {cmpv}")
                if "glc" in mods:
                    self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, old)
                self.emit(f"OpBranch {end}")
                self.emit(f"{end} = OpLabel")
                return
            spv, neutral = self.IMG_ATOMICS[op]
            tp, old, safe = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{tp} = OpImageTexelPointer %_ptr_Image_uint {ip} {iv} {self.cu(0)}")
            self.emit(f"{safe} = OpSelect %uint {self.active_b()} {val} {self.cu(neutral)}")
            self.emit(f"{old} = {spv} %uint {tp} {self.cu(1)} {self.cu(0)} {safe}")
            if "glc" in mods:
                self.dst({"k": "vgpr", "r": vdata["r"], "n": 1}, old)
            return
        ip, img = self.tmp(), self.tmp()
        self.emit(f"{ip} = OpAccessChain %_ptr_UC_img %images {idx}")
        self.emit(f"{img} = OpLoad %img2d {ip}")
        dm = next((m for m in mods if m.startswith("dmask:")), "dmask:0x1")
        dmask = int(dm.split(":")[1], 0)
        self.uses_query = True
        if op == "image_get_resinfo":
            lod = self.as_int(self.load_reg("vgpr", addr[0]))
            sz, lv = self.tmp(), self.tmp()
            self.emit(f"{sz} = OpImageQuerySizeLod %v2int {img} {lod}")
            self.emit(f"{lv} = OpImageQueryLevels %int {img}")
            w, h = self.tmp(), self.tmp()
            self.emit(f"{w} = OpCompositeExtract %int {sz} 0")
            self.emit(f"{h} = OpCompositeExtract %int {sz} 1")
            vals = [w, h, self.as_int(self.cu(1)), lv]          # x=width, y=height, z=depth(1), w=mip levels
            out = 0
            for c in range(4):
                if dmask >> c & 1:
                    u = self.tmp()
                    self.emit(f"{u} = OpBitcast %uint {vals[c]}")
                    self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, u)
                    out += 1
            return
        # image_load_mip: x, y, mip
        if len(addr) < 3:
            raise Unsupported("image_load_mip coords")
        xs = [self.as_int(self.load_reg("vgpr", r)) for r in addr[:3]]
        iv, res = self.tmp(), self.tmp()
        self.emit(f"{iv} = OpCompositeConstruct %v2int {xs[0]} {xs[1]}")
        self.emit(f"{res} = OpImageFetch %v4float {img} {iv} Lod {xs[2]}")
        out = 0
        for c in range(4):
            if dmask >> c & 1:
                comp = self.tmp()
                self.emit(f"{comp} = OpCompositeExtract %float {res} {c}")
                self.dst({"k": "vgpr", "r": vdata["r"] + out, "n": 1}, self.tou(comp))
                out += 1

    def attribute(self, n, comp, flat=False):
        """Component `comp` of fragment input attribute n (declared as a vec4 Input at Location n)."""
        name = f"%in_attr{n}"
        self.inputs[name] = n
        if not flat:
            self.interpolated.add(name)
        p, v = self.tmp(), self.tmp()
        self.emit(f"{p} = OpAccessChain %_ptr_Input_float {name} {self.cu(comp)}")
        self.emit(f"{v} = OpLoad %float {p}")
        return v

    def as_int(self, u):
        i = self.tmp()
        self.emit(f"{i} = OpBitcast %int {u}")
        return i

    # ---- legacy (merged ES/GS) geometry shader on LDS rings ----
    # Inputs: input vertex i sits at the dword offset the system VGPRs give (v0 lo/hi, v1 lo); the entry sets them
    # to 64*i, so an LDS read at byte address A + offset is vertex A >> 8, dword ((offset/4) mod 64) = 4*reg + comp
    # of the ES output (= the GS input signature register). Outputs: every LDS write goes to the per-lane GSVS
    # area: emitted vertex k is dwords [k*4*nreg, (k+1)*4*nreg), dword = 4*reg + comp of the GS output signature
    # (checked on $GS_VoxelizeGeneric: 20 dwords = 5 output registers, writes match every declared mask).
    # EMIT copies slot <emit count> to the outputs and emits; CUT ends the strip.
    GS_VTX_DW = 64

    def gs_when_active(self, instr):
        act = self.active_b()
        do, end = self.tmp() + "_do", self.tmp() + "_end"
        self.emit(f"OpSelectionMerge {end} None")
        self.emit(f"OpBranchConditional {act} {do} {end}")
        self.emit(f"{do} = OpLabel")
        self.emit(instr)
        self.emit(f"OpBranch {end}")
        self.emit(f"{end} = OpLabel")

    def gs_ds(self, op, a, mods):
        nreg = self.gs["nreg"]
        if op in ("ds_read_b32", "ds_read_b64", "ds_read2_b32", "ds_read_b128", "ds_read_b96"):
            if op == "ds_read2_b32":
                dws = [self.mod_int(mods, "offset0", 0), self.mod_int(mods, "offset1", 0)]
            else:
                n = {"ds_read_b32": 1, "ds_read_b64": 2, "ds_read_b96": 3, "ds_read_b128": 4}[op]
                base = self.mod_int(mods, "offset", 0)
                if base % 4:
                    raise Unsupported("gs lds read alignment")
                dws = [base // 4 + k for k in range(n)]
            addr = self.src_u(a[1])
            vtx = self.tmp()
            self.emit(f"{vtx} = OpShiftRightLogical %uint {addr} {self.cu(8)}")
            for k, dw in enumerate(dws):
                dw %= self.GS_VTX_DW
                reg, comp = dw // 4, dw % 4
                regs = {r: sv for r, sv in self.gs["in"]}
                if reg in regs:
                    var = "%gs_in_pos" if regs[reg] == 1 else f"%gs_in{reg}"
                    pp, v4, c = self.tmp(), self.tmp(), self.tmp()
                    self.emit(f"{pp} = OpAccessChain %_ptr_Input_v4 {var} {vtx}")
                    self.emit(f"{v4} = OpLoad %v4float {pp}")
                    self.emit(f"{c} = OpCompositeExtract %float {v4} {comp}")
                    val = self.tou(c)
                else:
                    val = self.cu(0)
                self.dst({"k": "vgpr", "r": a[0]["r"] + k, "n": 1}, val)
            return
        if op in ("ds_write_b32", "ds_write2_b32", "ds_write_b64"):
            if op == "ds_write2_b32":
                items = [(self.mod_int(mods, "offset0", 0), a[1]), (self.mod_int(mods, "offset1", 0), a[2])]
            elif op == "ds_write_b64":
                base = self.mod_int(mods, "offset", 0) // 4
                items = [(base, {"k": "vgpr", "r": a[1]["r"], "n": 1}), (base + 1, {"k": "vgpr", "r": a[1]["r"] + 1, "n": 1})]
            else:
                items = [(self.mod_int(mods, "offset", 0) // 4, a[1])]
            for dw, src in items:
                slot, rem = divmod(dw, 4 * nreg)
                if slot >= self.gs["maxv"]:
                    raise Unsupported("gs output slot")
                reg, comp = rem // 4, rem % 4
                v = self.tof(self.src_u(src))
                pp, old, new, sel = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{pp} = OpAccessChain %_ptr_Function_v4_gs %gs_stage {self.cu(slot * nreg + reg)}")
                self.emit(f"{old} = OpLoad %v4float {pp}")
                self.emit(f"{new} = OpCompositeInsert %v4float {v} {old} {comp}")
                self.emit(f"{sel} = OpSelect %v4float {self.active_b()} {new} {old}")
                self.emit(f"OpStore {pp} {sel}")
            return
        raise Unsupported(op + " in gs")

    def gs_emit(self):
        nreg = self.gs["nreg"]
        act = self.active_b()
        do, end = self.tmp() + "_do", self.tmp() + "_end"
        self.emit(f"OpSelectionMerge {end} None")
        self.emit(f"OpBranchConditional {act} {do} {end}")
        self.emit(f"{do} = OpLabel")
        k, base = self.tmp(), self.tmp()
        self.emit(f"{k} = OpLoad %uint %gs_emitted")
        self.emit(f"{base} = OpIMul %uint {k} {self.cu(nreg)}")
        for reg, sv in self.gs["out"]:
            i, pp, v = self.tmp(), self.tmp(), self.tmp()
            self.emit(f"{i} = OpIAdd %uint {base} {self.cu(reg)}")
            self.emit(f"{pp} = OpAccessChain %_ptr_Function_v4_gs %gs_stage {i}")
            self.emit(f"{v} = OpLoad %v4float {pp}")
            self.emit(f"OpStore {'%gs_out_pos' if sv == 1 else f'%gs_out{reg}'} {v}")
        self.emit("OpEmitVertex")
        k2 = self.tmp()
        self.emit(f"{k2} = OpIAdd %uint {k} {self.cu(1)}")
        self.emit(f"OpStore %gs_emitted {k2}")
        self.emit(f"OpBranch {end}")
        self.emit(f"{end} = OpLabel")

    # ---- ray tracing: image_bvh(64)_intersect_ray ----
    def image_bvh(self, op, a, mods):
        """One BVH traversal step against AMD's hardware node format, emulated over buffer reads. Semantics follow
        Mesa RADV's software path (radv_nir_rt_common.c: intersect_ray_amd_software_box / _tri), which RADV uses
        in place of this instruction. Node id = byte address >> 3 with the node type in bits [2:0]: 0..3 triangle,
        4 box16, 5 box32. Address = descriptor base + ((node & ~7) << 3); the base is BVH descriptor bits [39:0]
        taken in 256-byte units (ASSUMPTION: RADV only ever uses base 0). Box nodes return the four child ids
        (misses and inactive boxes = 0xFFFFFFFF), sorted by entry distance when the descriptor's box_sort_enable
        (bit 63) is set. Triangle nodes return (t/|det|, sign(det), v/|det|, w/|det|), or (inf, 1, 0, 0) on a miss:
        consumers divide by .y, which gives the same values as the hardware's numerator/denominator form."""
        if "a16" in mods:
            raise Unsupported(op + " a16")
        is64 = op.startswith("image_bvh64")
        self.uses_bvh = True
        vd = a[0]
        regs = self.addr_regs(a[1])
        need = 12 if is64 else 11
        if len(regs) < need or vd["k"] != "vgpr":
            raise Unsupported(op + " operands")
        desc = self.sgpr_words(a[2], 4)
        ld = lambda r: self.load_reg("vgpr", r)
        self.uses_psb = True
        self.uses_long = True
        e = self.emit
        T = self.tmp

        def u64(lo, hi):
            l, h, sh, r = T(), T(), T(), T()
            e(f"{l} = OpUConvert %ulong {lo}")
            e(f"{h} = OpUConvert %ulong {hi}")
            e(f"{sh} = OpShiftLeftLogical %ulong {h} %c64_32")
            e(f"{r} = OpBitwiseOr %ulong {sh} {l}")
            return r

        # descriptor base (bits [39:0], 256-byte units) and sort flag
        bhi, base40, base = T(), T(), T()
        e(f"{bhi} = OpBitwiseAnd %uint {desc[1]} {self.cu(0xFF)}")
        base40 = u64(desc[0], bhi)
        e(f"{base} = OpShiftLeftLogical %ulong {base40} {self.cl(8)}")
        sortb, sort = T(), T()
        e(f"{sortb} = OpShiftRightLogical %uint {desc[1]} {self.cu(31)}")
        e(f"{sort} = OpINotEqual %bool {sortb} {self.cu(0)}")
        # node id
        if is64:
            node = u64(ld(regs[0]), ld(regs[1]))
            rest = regs[2:]
        else:
            n32 = T()
            e(f"{n32} = OpUConvert %ulong {ld(regs[0])}")
            node = n32
            rest = regs[1:]
        nlo, ntype = T(), T()
        e(f"{nlo} = OpUConvert %uint {node}")
        e(f"{ntype} = OpBitwiseAnd %uint {nlo} {self.cu(7)}")
        nm, off, addr = T(), T(), T()
        e(f"{nm} = OpBitwiseAnd %ulong {node} {self.cl(0xFFFFFFFFFFFFFFF8)}")
        e(f"{off} = OpShiftLeftLogical %ulong {nm} {self.cl(3)}")
        e(f"{addr} = OpIAdd %ulong {base} {off}")
        ptr = T()
        e(f"{ptr} = OpConvertUToPtr %_ptr_PSB_arr {self.host_addr(addr)}")

        def ldw(i):
            pp, v = T(), T()
            e(f"{pp} = OpAccessChain %_ptr_PSB_uint {ptr} {self.cu(i)}")
            e(f"{v} = OpLoad %uint {pp} Aligned 4")
            return v

        f = lambda r: self.tof(ld(r))
        tmax = f(rest[0])
        org = [f(r) for r in rest[1:4]]
        dirv = [f(r) for r in rest[4:7]]
        inv = [f(r) for r in rest[7:10]]
        INF, ONE, NEGMAX, POSMAX = (self.tof(self.cu(x)) for x in (0x7F800000, 0x3F800000, 0xFF7FFFFF, 0x7F7FFFFF))
        out = lambda k, v: self.dst({"k": "vgpr", "r": vd["r"] + k, "n": 1}, v)

        def fop(o, x, y):
            r = T()
            e(f"{r} = {o} %float {x} {y}")
            return r

        def fext(name, *xs):
            r = T()
            e(f"{r} = OpExtInst %float %glsl {name} {' '.join(xs)}")
            return r

        def bop(o, x, y):
            r = T()
            e(f"{r} = {o} %bool {x} {y}")
            return r

        def sel(c, x, y, ty="%float"):
            r = T()
            e(f"{r} = OpSelect {ty} {c} {x} {y}")
            return r

        is_tri, is_b16, is_box = T(), T(), T()
        e(f"{is_tri} = OpULessThan %bool {ntype} {self.cu(4)}")
        e(f"{is_b16} = OpIEqual %bool {ntype} {self.cu(4)}")
        bx16 = T()
        e(f"{bx16} = OpIEqual %bool {ntype} {self.cu(5)}")
        e(f"{is_box} = OpLogicalOr %bool {is_b16} {bx16}")
        L = {k: T() + "_" + k for k in ("tri", "nt", "box", "oth", "end1", "end2")}
        e(f"OpSelectionMerge {L['end1']} None")
        e(f"OpBranchConditional {is_tri} {L['tri']} {L['nt']}")

        # ---------------- triangle ----------------
        e(f"{L['tri']} = OpLabel")
        V = [[self.tof(ldw(3 * j + c)) for c in range(3)] for j in range(3)]
        ad = [fext("FAbs", d) for d in dirv]
        c01, c02, c12 = bop("OpFOrdGreaterThanEqual", ad[0], ad[1]), bop("OpFOrdGreaterThanEqual", ad[0], ad[2]), \
            bop("OpFOrdGreaterThanEqual", ad[1], ad[2])
        k_a = sel(c02, self.cu((0 << 4) | (2 << 2) | 1), self.cu((2 << 4) | (1 << 2) | 0), "%uint")
        k_b = sel(c12, self.cu((1 << 4) | (0 << 2) | 2), self.cu((2 << 4) | (1 << 2) | 0), "%uint")
        pk = sel(c01, k_a, k_b, "%uint")
        kx0, ky0, kz = T(), T(), T()
        e(f"{kx0} = OpBitwiseAnd %uint {pk} {self.cu(3)}")
        e(f"{ky0} = OpBitFieldUExtract %uint {pk} {self.cu(2)} {self.cu(2)}")
        e(f"{kz} = OpShiftRightLogical %uint {pk} {self.cu(4)}")
        dvec, ivec = T(), T()
        e(f"{dvec} = OpCompositeConstruct %v3float {' '.join(dirv)}")
        e(f"{ivec} = OpCompositeConstruct %v3float {' '.join(inv)}")

        def vx(vec, i):
            r = T()
            e(f"{r} = OpVectorExtractDynamic %float {vec} {i}")
            return r
        dkz = vx(dvec, kz)
        neg = bop("OpFOrdLessThan", dkz, "%f0")
        kx = sel(neg, ky0, kx0, "%uint")
        ky = sel(neg, kx0, ky0, "%uint")
        sz = vx(ivec, kz)
        sx = fop("OpFMul", vx(dvec, kx), sz)
        sy = fop("OpFMul", vx(dvec, ky), sz)
        rel = []
        for j in range(3):
            vv = T()
            comps = [fop("OpFSub", V[j][c], org[c]) for c in range(3)]
            e(f"{vv} = OpCompositeConstruct %v3float {' '.join(comps)}")
            rel.append(vv)
        def shear(vv):
            zk = vx(vv, kz)
            return (fop("OpFSub", vx(vv, kx), fop("OpFMul", sx, zk)),
                    fop("OpFSub", vx(vv, ky), fop("OpFMul", sy, zk)), zk)
        (ax, ay, az0), (bx, by, bz0), (cx, cy, cz0) = (shear(r) for r in rel)
        uu = fop("OpFSub", fop("OpFMul", cx, by), fop("OpFMul", cy, bx))
        vv_ = fop("OpFSub", fop("OpFMul", ax, cy), fop("OpFMul", ay, cx))
        ww = fop("OpFSub", fop("OpFMul", bx, ay), fop("OpFMul", by, ax))
        lt0 = lambda x: bop("OpFOrdLessThan", x, "%f0")
        gt0 = lambda x: bop("OpFOrdGreaterThan", x, "%f0")
        eq0 = lambda x: bop("OpFOrdEqual", x, "%f0")
        le0 = lambda x: bop("OpFOrdLessThanEqual", x, "%f0")
        orr = lambda x, y: bop("OpLogicalOr", x, y)
        andd = lambda x, y: bop("OpLogicalAnd", x, y)
        cback = orr(orr(lt0(uu), lt0(vv_)), lt0(ww))
        cfront = orr(orr(gt0(uu), gt0(vv_)), gt0(ww))
        cond = T()
        e(f"{cond} = OpLogicalNot %bool {andd(cback, cfront)}")
        ea, eb, ec = eq0(uu), eq0(vv_), eq0(ww)

        def edge(v0x, v0y, v1x, v1y):
            tx = fop("OpFSub", v1x, v0x)
            ty = fop("OpFSub", v1y, v0y)
            return sel(eq0(tx), lt0(ty), lt0(tx), "%bool")

        def vert(v0x, v1x, v2x):
            return andd(le0(fop("OpFSub", v1x, v0x)), gt0(fop("OpFSub", v2x, v0x)))
        iea = andd(ea, edge(bx, by, cx, cy))
        ieb = andd(eb, edge(cx, cy, ax, ay))
        iec = andd(ec, edge(ax, ay, bx, by))
        ie = andd(cond, orr(orr(iea, ieb), iec))
        isva, isvb, isvc = andd(eb, ec), andd(ea, ec), andd(ea, eb)
        iva = andd(isva, vert(ax, bx, cx))
        ivb = andd(isvb, vert(bx, cx, ax))
        ivc = andd(isvc, vert(cx, ax, bx))
        isv = orr(orr(isva, isvb), isvc)
        nisv = T()
        e(f"{nisv} = OpLogicalNot %bool {isv}")
        iv = orr(nisv, orr(orr(iva, ivb), ivc))
        ie2 = andd(ie, iv)
        cedge = orr(orr(ea, eb), ec)
        hitc = sel(cedge, ie2, cond, "%bool")
        det = fop("OpFAdd", uu, fop("OpFAdd", vv_, ww))
        azz, bzz, czz = (fop("OpFMul", sz, z) for z in (az0, bz0, cz0))
        tt = fop("OpFAdd", fop("OpFAdd", fop("OpFMul", uu, azz), fop("OpFMul", vv_, bzz)), fop("OpFMul", ww, czz))
        sg = fext("FSign", det)
        ts = fop("OpFMul", sg, tt)
        front = T()
        e(f"{front} = OpLogicalNot %bool {lt0(ts)}")
        hit = andd(hitc, front)
        dab = fext("FAbs", det)
        rt = sel(hit, fop("OpFDiv", tt, dab), INF)
        rs = sel(hit, sg, ONE)
        rv = sel(hit, fop("OpFDiv", vv_, dab), "%f0")
        rw = sel(hit, fop("OpFDiv", ww, dab), "%f0")
        for k, v in enumerate((rt, rs, rv, rw)):
            out(k, self.tou(v))
        e(f"OpBranch {L['end1']}")

        # ---------------- not a triangle ----------------
        e(f"{L['nt']} = OpLabel")
        e(f"OpSelectionMerge {L['end2']} None")
        e(f"OpBranchConditional {is_box} {L['box']} {L['oth']}")
        e(f"{L['box']} = OpLabel")
        invc = [fext("FClamp", d, NEGMAX, POSMAX) for d in inv]
        dist, idx = [], []
        for i in range(4):
            child = ldw(i)
            # box32: min/max f32 at 16 + 24 i; box16: 6 halves at 16 + 12 i (min x,y,z, max x,y,z)
            m32 = [self.tof(ldw(4 + 6 * i + c)) for c in range(6)]
            h = [ldw(4 + 3 * i + c) for c in range(3)]
            m16 = []
            for w in h:
                v2 = T()
                e(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {w}")
                for c in range(2):
                    x = T()
                    e(f"{x} = OpCompositeExtract %float {v2} {c}")
                    m16.append(x)
            mm = [sel(is_b16, m16[c], m32[c]) for c in range(6)]
            b0 = [fop("OpFMul", fop("OpFSub", mm[c], org[c]), invc[c]) for c in range(3)]
            b1 = [fop("OpFMul", fop("OpFSub", mm[3 + c], org[c]), invc[c]) for c in range(3)]
            tmin = fext("FMax", fext("FMax", fext("FMin", b0[0], b1[0]), fext("FMin", b0[1], b1[1])),
                        fext("FMin", b0[2], b1[2]))
            tmx = fext("FMin", fext("FMin", fext("FMax", b0[0], b1[0]), fext("FMax", b0[1], b1[1])),
                       fext("FMax", b0[2], b1[2]))
            notnan = T()
            e(f"{notnan} = OpFOrdEqual %bool {mm[0]} {mm[0]}")
            h1 = bop("OpFOrdGreaterThanEqual", tmx, fext("FMax", "%f0", tmin))
            h2 = bop("OpFOrdLessThan", tmin, tmax)
            hb = andd(notnan, andd(h1, h2))
            dist.append(sel(hb, tmin, INF))
            idx.append(sel(hb, child, self.cu(0xFFFFFFFF), "%uint"))
        for p_, q_ in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
            sw = andd(sort, bop("OpFOrdLessThan", dist[q_], dist[p_]))
            dist[p_], dist[q_] = sel(sw, dist[q_], dist[p_]), sel(sw, dist[p_], dist[q_])
            idx[p_], idx[q_] = sel(sw, idx[q_], idx[p_], "%uint"), sel(sw, idx[p_], idx[q_], "%uint")
        for k in range(4):
            out(k, idx[k])
        e(f"OpBranch {L['end2']}")
        e(f"{L['oth']} = OpLabel")
        for k in range(4):            # instance / aabb nodes are handled by the shader, not the instruction
            out(k, self.cu(0xFFFFFFFF))
        e(f"OpBranch {L['end2']}")
        e(f"{L['end2']} = OpLabel")
        e(f"OpBranch {L['end1']}")
        e(f"{L['end1']} = OpLabel")

    def ms_export(self, tgt, ins):
        """NGG export in a mesh shader: thread t exports vertex t (pos/param) and primitive t (prim). The values are
        staged in per-invocation variables and written to the mesh outputs at shader end, after
        OpSetMeshOutputsEXT (see assemble). Exports whose operands are all 'off' are the dummy exports a wave with no
        vertices must still issue: they write nothing."""
        a = ins["args"]
        srcs = a[1:5]
        if all(x["k"] == "special" and x["t"] == "off" for x in srcs):
            return
        act = self.active_b()
        if tgt == "prim":
            # Packed: v0 [8:0], v1 [18:10], v2 [28:20], null primitive [31].
            u = self.src_u(srcs[0])
            old, sel = self.tmp(), self.tmp()
            self.emit(f"{old} = OpLoad %uint %ms_prim")
            self.emit(f"{sel} = OpSelect %uint {act} {u} {old}")
            self.emit("OpStore %ms_prim " + sel)
            hp, hs = self.tmp(), self.tmp()
            self.emit(f"{hp} = OpLoad %bool %ms_has_prim")
            self.emit(f"{hs} = OpLogicalOr %bool {hp} {act}")
            self.emit("OpStore %ms_has_prim " + hs)
            return
        if tgt.startswith("pos") and tgt != "pos0":
            if "compr" in ins["mods"]:
                raise Unsupported("mesh compressed export")
            pidx = int(tgt[3:])
            mapping = self.pos_map.get(pidx)
            if not mapping:
                return
            wrote = False
            for comp, (kind, idx) in sorted(mapping.items()):
                if comp >= len(srcs):
                    continue
                x = srcs[comp]
                if x["k"] == "special" and x["t"] == "off":
                    continue
                fval = self.src_f(x)
                if kind == "clip":
                    self.uses_clip = True
                    var = "%ms_clip"
                else:
                    self.uses_cull = True
                    var = "%ms_cull_dist"
                ptr, old, sel = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{ptr} = OpAccessChain %_ptr_Function_float {var} {self.cu(idx)}")
                self.emit(f"{old} = OpLoad %float {ptr}")
                self.emit(f"{sel} = OpSelect %float {act} {fval} {old}")
                self.emit(f"OpStore {ptr} {sel}")
                wrote = True
            if wrote:
                hv, hs = self.tmp(), self.tmp()
                self.emit(f"{hv} = OpLoad %bool %ms_has_vtx")
                self.emit(f"{hs} = OpLogicalOr %bool {hv} {act}")
                self.emit("OpStore %ms_has_vtx " + hs)
            return
        if not (tgt == "pos0" or tgt.startswith("param")):
            raise Unsupported(f"mesh export {tgt}")
        if "compr" in ins["mods"]:
            raise Unsupported("mesh compressed export")
        comps = [("%f0" if x["k"] == "special" and x["t"] == "off" else self.src_f(x)) for x in srcs]
        comps += ["%f0"] * (4 - len(comps))
        v = self.tmp()
        self.emit(f"{v} = OpCompositeConstruct %v4float {' '.join(comps[:4])}")
        var = "%ms_pos" if tgt == "pos0" else f"%ms_param{int(tgt[5:])}"
        if tgt != "pos0":
            self.ms_params.add(int(tgt[5:]))
        old, sel = self.tmp(), self.tmp()
        self.emit(f"{old} = OpLoad %v4float {var}")
        self.emit(f"{sel} = OpSelect %v4float {act} {v} {old}")
        self.emit(f"OpStore {var} {sel}")
        hv, hs = self.tmp(), self.tmp()
        self.emit(f"{hv} = OpLoad %bool %ms_has_vtx")
        self.emit(f"{hs} = OpLogicalOr %bool {hv} {act}")
        self.emit("OpStore %ms_has_vtx " + hs)

    def export(self, ins):
        a = ins["args"]
        tgt = a[0]["t"] if a and a[0]["k"] in ("target", "special") else None
        if tgt is None or tgt == "null":
            return
        if self.stage == "ms":
            return self.ms_export(tgt, ins)
        if tgt.startswith("pos") and tgt != "pos0" and self.stage == "vs":
            srcs = a[1:5]
            if all(x["k"] == "special" and x["t"] == "off" for x in srcs):
                return
            pidx = int(tgt[3:])
            act = self.active_b()
            if pidx == 1 and self.pos_misc:
                # GFX9+ misc vector at pos1: z[10:0] = Layer, z[19:16] = ViewportIndex.
                if len(srcs) >= 3 and not (srcs[2]["k"] == "special" and srcs[2]["t"] == "off"):
                    z = self.src_u(srcs[2])
                    if 4 in self.pos_misc:
                        self.uses_layer = True
                        lu, li, old, sel = self.tmp(), self.tmp(), self.tmp(), self.tmp()
                        self.emit(f"{lu} = OpBitwiseAnd %uint {z} {self.cu(0x7FF)}")
                        self.emit(f"{li} = OpBitcast %int {lu}")
                        self.emit(f"{old} = OpLoad %int %out_layer")
                        self.emit(f"{sel} = OpSelect %int {act} {li} {old}")
                        self.emit(f"OpStore %out_layer {sel}")
                    if 5 in self.pos_misc:
                        self.uses_viewport = True
                        vs_, vu, vi, old, sel = self.tmp(), self.tmp(), self.tmp(), self.tmp(), self.tmp()
                        self.emit(f"{vs_} = OpShiftRightLogical %uint {z} {self.cu(16)}")
                        self.emit(f"{vu} = OpBitwiseAnd %uint {vs_} {self.cu(0xF)}")
                        self.emit(f"{vi} = OpBitcast %int {vu}")
                        self.emit(f"{old} = OpLoad %int %out_viewport")
                        self.emit(f"{sel} = OpSelect %int {act} {vi} {old}")
                        self.emit(f"OpStore %out_viewport {sel}")
                return
            mapping = self.pos_map.get(pidx)
            if not mapping:
                return
            for comp, (kind, idx) in sorted(mapping.items()):
                if comp >= len(srcs):
                    continue
                x = srcs[comp]
                if x["k"] == "special" and x["t"] == "off":
                    continue
                fval = self.src_f(x)
                if kind == "clip":
                    self.uses_clip = True
                    var = "%out_clip"
                else:
                    self.uses_cull = True
                    var = "%out_cull"
                ptr, old, sel = self.tmp(), self.tmp(), self.tmp()
                self.emit(f"{ptr} = OpAccessChain %_ptr_Output_float {var} {self.cu(idx)}")
                self.emit(f"{old} = OpLoad %float {ptr}")
                self.emit(f"{sel} = OpSelect %float {act} {fval} {old}")
                self.emit(f"OpStore {ptr} {sel}")
            return
        if tgt == "mrtz" and self.stage == "ps":
            # Depth export: x is the fragment depth when it's on in an uncompressed export (RGBA = Z, stencil,
            # sample mask, alpha; Mesa ac_shader_util.c ac_get_spi_shader_z_format). A compressed mrtz is the 16-bit
            # stencil and sample mask format, which never holds Z (ac_nir_lower_ps_late.c emit_ps_mrtz_export), and
            # with x off no depth is exported. Until 2026-11-22 both wrote one anyway: 0.0, or an f16 half of the
            # stencil word under compr. Stencil, sample mask and alpha aren't modelled.
            x = a[1] if len(a) > 1 else None
            if "compr" in ins["mods"] or x is None or (x["k"] == "special" and x["t"] == "off"):
                return
            self.writes_depth = True
            old, sel = self.tmp(), self.tmp()
            self.emit(f"{old} = OpLoad %float %out_depth")
            self.emit(f"{sel} = OpSelect %float {self.active_b()} {self.src_f(x)} {old}")
            self.emit(f"OpStore %out_depth {sel}")
            return
        comps = []
        if "compr" in ins["mods"]:
            # Compressed export: two registers, each holding two f16 values (x,y) and (z,w). LLVM lists each register
            # twice (one slot per half): 'exp mrt0 v4, v4, v5, v5 compr', so the pair is slots 1 and 3.
            # (Found by the GPU texture test: taking the first two operands exported (r,g,r,g).)
            srcs = a[1:5]
            regs = [srcs[0], srcs[2]] if len(srcs) >= 3 else srcs[:1]
            for x in regs:
                if x["k"] == "special" and x["t"] == "off":
                    comps += ["%f0", "%f0"]
                    continue
                u = self.src_u(x)
                v2 = self.tmp()
                self.emit(f"{v2} = OpExtInst %v2float %glsl UnpackHalf2x16 {u}")
                for c in (0, 1):
                    f = self.tmp()
                    self.emit(f"{f} = OpCompositeExtract %float {v2} {c}")
                    comps.append(f)
        else:
            for x in a[1:5]:
                if x["k"] == "special" and x["t"] == "off":
                    comps.append("%f0")
                else:
                    comps.append(self.src_f(x))
        while len(comps) < 4:
            comps.append("%f0")
        comps = comps[:4]
        v = self.tmp()
        self.emit(f"{v} = OpCompositeConstruct %v4float {' '.join(comps)}")
        name = {"pos0": "%out_pos"}.get(tgt)
        if name is None:
            if tgt.startswith("param") and self.stage == "vs":
                name = f"%out_param{int(tgt[5:])}"
            elif tgt.startswith("mrt") and tgt != "mrtz" and self.stage == "ps":
                name = f"%out_mrt{int(tgt[3:])}"
            elif tgt.startswith("pos"):
                return  # pos1+ (clip/cull/point size): ignored in v0
            else:
                raise Unsupported(f"export {tgt}")
        self.outputs[name] = tgt
        # Exports from inactive lanes don't happen: keep the previous output value.
        old, sel = self.tmp(), self.tmp()
        self.emit(f"{old} = OpLoad %v4float {name}")
        self.emit(f"{sel} = OpSelect %v4float {self.active_b()} {v} {old}")
        self.emit(f"OpStore {name} {sel}")


_TYPE_DECL = re.compile(r"\s*(%\w+) = (OpTypeVector|OpTypePointer) (.*)$")
_ID_REF = re.compile(r"%\w+")


def dedupe_types(text):
    """SPIR-V forbids duplicate non-aggregate types. Separate features may each declare e.g. a uint4 vector
    (%v4uint for r32ui storage, %v4uint_st for typed stores, %v4uint_in for vertex inputs): keep the first
    declaration of each OpTypeVector / OpTypePointer and rename the others everywhere."""
    alias, seen, out = {}, {}, []
    for line in text.split("\n"):
        m = _TYPE_DECL.match(line)
        if m:
            name, opc, rest = m.groups()
            key = (opc, _ID_REF.sub(lambda t: alias.get(t.group(0), t.group(0)), rest))
            if key in seen:
                alias[name] = seen[key]
                continue
            seen[key] = name
        out.append(line)
    if not alias:
        return text
    return "\n".join(_ID_REF.sub(lambda t: alias.get(t.group(0), t.group(0)), l) for l in out)


def assemble(gen):
    vs = gen.stage == "vs"
    cs = gen.stage == "cs"
    ms = gen.stage == "ms"
    gsx = gen.stage == "gs"
    if gen.emulate:
        # No subgroup operation is left (x_round does them through %x_buf), so none of their capabilities.
        if any("OpGroupNonUniform" in l or "%sg_lane" in l for b in gen.blocks + list(gen.x_segs.values()) for l in b):
            raise Unsupported("wave64 emulation: a subgroup operation is left")
        gen.uses_subgroup = gen.uses_vote = gen.uses_ballot = gen.uses_wave_bits = gen.uses_quad = False
        x_waves, x_mask, x_sched, x_sink = gen.x_sizes()
        x_zero = x_sink - x_mask           # mask and schedule words: zeroed before the first round
        x_threads = (gen.cs or gen.ms)["size"]
        x_threads = x_threads[0] * x_threads[1] * x_threads[2]
        sx, sy, _ = (gen.cs or gen.ms)["size"]
        for v in (0, 1, 2, 3, 6, 31, 63, 0x948, x_mask, x_sched, x_sink, x_sink + 1, x_zero - 1, x_threads,
                  x_waves, sx, sx * sy, 0xFFFFFFC0):
            gen.cu(v)
        x_missing = gen.x_lanes - x_threads          # lanes of a partial last wave that don't exist
        for k in range(-(-max(x_zero, x_missing) // x_threads)):
            gen.cu(k * x_threads)
        if x_missing:
            for v in (x_missing - 1, gen.x_lanes + x_threads):
                gen.cu(v)
    model = "Vertex" if vs else "GLCompute" if cs else "MeshEXT" if ms else "Geometry" if gsx else "Fragment"
    ms_blk = ms and (gen.uses_clip or gen.uses_cull)
    iface = list(gen.outputs) + list(gen.inputs) + (["%vertex_index", "%instance_index"] if vs else []) +        (["%wg_id", "%local_id"] if cs else []) + (["%lds"] if gen.uses_lds else []) + (
        (["%wg_id", "%num_wg", "%local_index", "%ms_counts", ("%ms_out_vtx" if ms_blk else "%ms_out_pos"), "%ms_prim_idx", "%ms_cull"] +
         [f"%ms_out_param{n}" for n in sorted(gen.ms_params)]) if ms else []) + (
        ["%x_buf"] if gen.emulate else []) + (
        ([("%gs_in_pos" if sv == 1 else f"%gs_in{r}") for r, sv in gen.gs["in"]] +
         [("%gs_out_pos" if sv == 1 else f"%gs_out{r}") for r, sv in gen.gs["out"]]) if gsx else []) +         ((["%fmt_uint_mask", "%fmt_sint_mask"] +
          [f"%stv{c}_{d}" for d in sorted(gen.uses_store_views) for c in range(3)]) if gen.uses_store_views else [])
    psb = gen.uses_psb
    tex = gen.uses_tex
    sg = gen.uses_subgroup
    if tex:
        iface += ["%images", "%samplers", "%desc_map"] + (["%images_arr"] if gen.uses_tex_array else []) + \
                 (["%simages"] if gen.uses_storage else []) + (["%images_1d"] if gen.uses_tex_1d else []) + \
                 (["%images_3d"] if gen.uses_tex_3d else []) + (["%simages3d"] if gen.uses_storage3d else [])
    if sg:
        iface += ["%sg_lane"]
    if gen.writes_depth:
        iface += ["%out_depth"]
    if vs:
        if gen.uses_clip:
            iface += ["%out_clip"]
        if gen.uses_cull:
            iface += ["%out_cull"]
        if gen.uses_layer:
            iface += ["%out_layer"]
        if gen.uses_viewport:
            iface += ["%out_viewport"]
    iface += [f"%vin{reg}" for reg, _ in gen.vertex_inputs] + ["%user_data"]
    bary = any(str(c).startswith("bary") for c in gen.frag_inputs.values())
    if gen.frag_inputs:
        iface += ["%frag_coord"] + (["%front_facing"] if "front" in gen.frag_inputs.values() else []) +             (["%bary"] if bary else [])
    for i in range(USER_SGPRS):          # index constants for the user-data copy at entry
        gen.cu(i)
    # SPIR-V module layout: every OpCapability, then every OpExtension, then imports.
    caps = ["Shader"] + (["Int64", "PhysicalStorageBufferAddresses"] if psb else []) + \
        (["RuntimeDescriptorArray"] if tex else []) + (["GroupNonUniform", "GroupNonUniformShuffle"] if sg else []) + \
        (["ImageQuery"] if gen.uses_query else []) + (["Sampled1D"] if gen.uses_tex_1d else []) + \
        (["MinLod"] if gen.uses_minlod else []) + (["ImageGatherExtended"] if gen.uses_img_offset else []) +         (["FragmentBarycentricKHR"] if bary else [])
    if gen.uses_vote:
        caps += [c for c in ("GroupNonUniform", "GroupNonUniformVote") if c not in caps]
    if gen.uses_ballot:
        caps += [c for c in ("GroupNonUniform", "GroupNonUniformBallot") if c not in caps]
    if gen.uses_wave_bits:
        caps += [c for c in ("GroupNonUniform", "GroupNonUniformArithmetic") if c not in caps]
    if gen.uses_quad:
        caps += [c for c in ("GroupNonUniform", "GroupNonUniformQuad") if c not in caps]
    if gen.uses_f64:
        caps.append("Float64")
    if gen.uses_store_views:
        caps.append("StorageImageWriteWithoutFormat")     # shaderStorageImageWriteWithoutFormat
        if 3 in gen.uses_store_views:
            caps.append("Image1D")
        if gen.uses_store_reads:
            caps.append("StorageImageReadWithoutFormat")     # shaderStorageImageReadWithoutFormat
    if gsx:
        caps.append("Geometry")
    if ms:
        caps.append("MeshShadingEXT")
    if gen.uses_clip:
        caps.append("ClipDistance")
    if gen.uses_cull:
        caps.append("CullDistance")
    if gen.uses_layer:
        caps.append("ShaderLayer")
    if gen.uses_viewport:
        caps.append("ShaderViewportIndex")
    pops = gen.uses_pops and gen.stage == "ps"
    exts = (["SPV_KHR_physical_storage_buffer"] if psb else []) + (["SPV_EXT_descriptor_indexing"] if tex else [])
    if bary:
        exts.append("SPV_KHR_fragment_shader_barycentric")
    if ms:
        exts.append("SPV_EXT_mesh_shader")
    if pops:
        caps.append("FragmentShaderPixelInterlockEXT")
        exts.append("SPV_EXT_fragment_shader_interlock")
    hdr = [f"OpCapability {c}" for c in caps] + [f'OpExtension "{e}"' for e in exts] + \
          ['%glsl = OpExtInstImport "GLSL.std.450"',
           f"OpMemoryModel {'PhysicalStorageBuffer64' if psb else 'Logical'} GLSL450",
           f"OpEntryPoint {model} %main \"main\" {' '.join(iface)}"]
    if cs:
        hdr.append("OpExecutionMode %main LocalSize {} {} {}".format(*gen.cs["size"]))
    if gsx:
        hdr += ["OpExecutionMode %main Triangles", "OpExecutionMode %main Invocations 1",
                "OpExecutionMode %main OutputTriangleStrip", f"OpExecutionMode %main OutputVertices {gen.gs['maxv']}"]
    if ms:
        hdr += ["OpExecutionMode %main LocalSize {} {} {}".format(*gen.ms["size"]),
                f"OpExecutionMode %main OutputVertices {gen.ms['maxv']}",
                f"OpExecutionMode %main OutputPrimitivesEXT {gen.ms['maxp']}",
                "OpExecutionMode %main OutputTrianglesEXT"]
    if gen.stage == "ps":
        hdr.append("OpExecutionMode %main OriginUpperLeft")
        if pops:
            hdr.append("OpExecutionMode %main PixelInterlockOrderedEXT")
        if gen.writes_depth:
            hdr.append("OpExecutionMode %main DepthReplacing")
    dec = []
    if vs:
        dec.append("OpDecorate %vertex_index BuiltIn VertexIndex")
        dec.append("OpDecorate %instance_index BuiltIn InstanceIndex")
    if cs:
        dec += ["OpDecorate %wg_id BuiltIn WorkgroupId", "OpDecorate %local_id BuiltIn LocalInvocationId"]
    if gsx:
        for r, sv in gen.gs["in"]:
            dec.append("OpDecorate %gs_in_pos BuiltIn Position" if sv == 1 else f"OpDecorate %gs_in{r} Location {r}")
        for r, sv in gen.gs["out"]:
            dec.append("OpDecorate %gs_out_pos BuiltIn Position" if sv == 1 else f"OpDecorate %gs_out{r} Location {r}")
    ms_clip_member = ms_cull_member = None
    if ms:
        dec += ["OpDecorate %wg_id BuiltIn WorkgroupId", "OpDecorate %num_wg BuiltIn NumWorkgroups",
                "OpDecorate %local_index BuiltIn LocalInvocationIndex"]
        if ms_blk:
            dec += ["OpDecorate %ms_vtx_blk Block", "OpMemberDecorate %ms_vtx_blk 0 BuiltIn Position"]
            mi = 1
            if gen.uses_clip:
                dec.append(f"OpMemberDecorate %ms_vtx_blk {mi} BuiltIn ClipDistance")
                ms_clip_member = mi
                mi += 1
            if gen.uses_cull:
                dec.append(f"OpMemberDecorate %ms_vtx_blk {mi} BuiltIn CullDistance")
                ms_cull_member = mi
                mi += 1
        else:
            dec.append("OpDecorate %ms_out_pos BuiltIn Position")
        # CullPrimitiveEXT as Block+bool member (VUID-07036): array-of-bool is also legal, but
        # SPIRV-Tools ≤2025.1 falsely rejects it (#5980); the Block form passes both old and new validators.
        # PerPrimitiveEXT is a MemberDecorate on the bool (glslang / SPIRV-Tools MeshEXT tests).
        dec += ["OpDecorate %ms_prim_idx BuiltIn PrimitiveTriangleIndicesEXT",
                "OpDecorate %ms_cull_blk Block",
                "OpMemberDecorate %ms_cull_blk 0 BuiltIn CullPrimitiveEXT",
                "OpMemberDecorate %ms_cull_blk 0 PerPrimitiveEXT"]
        dec += [f"OpDecorate %ms_out_param{n} Location {n}" for n in sorted(gen.ms_params)]
    if vs:
        if gen.uses_clip:
            dec.append("OpDecorate %out_clip BuiltIn ClipDistance")
        if gen.uses_cull:
            dec.append("OpDecorate %out_cull BuiltIn CullDistance")
        if gen.uses_layer:
            dec.append("OpDecorate %out_layer BuiltIn Layer")
        if gen.uses_viewport:
            dec.append("OpDecorate %out_viewport BuiltIn ViewportIndex")
    for name, tgt in gen.outputs.items():
        if name == "%out_pos":
            dec.append("OpDecorate %out_pos BuiltIn Position")
        else:
            dec.append(f"OpDecorate {name} Location {int(''.join(ch for ch in tgt if ch.isdigit()))}")
    for name, n in gen.inputs.items():
        dec.append(f"OpDecorate {name} Location {n}")
        if name not in gen.interpolated:
            dec.append(f"OpDecorate {name} Flat")
    types = ["%void = OpTypeVoid", "%fn = OpTypeFunction %void", "%bool = OpTypeBool", "%uint = OpTypeInt 32 0",
             "%int = OpTypeInt 32 1", "%float = OpTypeFloat 32", "%v4float = OpTypeVector %float 4",
             "%v2float = OpTypeVector %float 2", "%umul = OpTypeStruct %uint %uint",
             "%smul = OpTypeStruct %int %int",
             "%n256 = OpConstant %uint 256", f"%n_sgprs = OpConstant %uint {gen.SGPRS + len(gen.parked)}",
             "%arr_v = OpTypeArray %uint %n256", "%arr_s = OpTypeArray %uint %n_sgprs",
             "%null_v = OpConstantNull %arr_v", "%null_s = OpConstantNull %arr_s",
             "%_ptr_Function_arr_v = OpTypePointer Function %arr_v", "%_ptr_Function_arr_s = OpTypePointer Function %arr_s",
             "%_ptr_Function_uint = OpTypePointer Function %uint", "%_ptr_Output_v4 = OpTypePointer Output %v4float",
             "%_ptr_Input_int = OpTypePointer Input %int", "%_ptr_Input_v4 = OpTypePointer Input %v4float",
             "%_ptr_Input_float = OpTypePointer Input %float",
             "%f0 = OpConstant %float 0", "%f1 = OpConstant %float 1", "%true = OpConstantTrue %bool",
             "%false = OpConstantFalse %bool",
             "%_ptr_Function_bool = OpTypePointer Function %bool"]
    if psb:
        types += ["%ulong = OpTypeInt 64 0", "%c64_32 = OpConstant %ulong 32", "%long = OpTypeInt 64 1",
                  "%rt_uint = OpTypeRuntimeArray %uint", "%_ptr_PSB_arr = OpTypePointer PhysicalStorageBuffer %rt_uint",
                  "%_ptr_PSB_uint = OpTypePointer PhysicalStorageBuffer %uint"]
        dec.append("OpDecorate %rt_uint ArrayStride 4")
        types += [f"{name} = OpConstant %ulong {v}" for v, name in sorted(gen.lconsts.items())]
    if tex:
        # Bindless set 0: images (binding 0), samplers (binding 1), descriptor hash table (binding 2).
        types += ["%img2d = OpTypeImage %float 2D 0 0 0 1 Unknown", "%smp = OpTypeSampler",
                  "%simg2d = OpTypeSampledImage %img2d", "%v2int = OpTypeVector %int 2",
                  "%rt_img = OpTypeRuntimeArray %img2d", "%rt_smp = OpTypeRuntimeArray %smp",
                  "%_ptr_UC_rt_img = OpTypePointer UniformConstant %rt_img",
                  "%_ptr_UC_rt_smp = OpTypePointer UniformConstant %rt_smp",
                  "%_ptr_UC_img = OpTypePointer UniformConstant %img2d", "%_ptr_UC_smp = OpTypePointer UniformConstant %smp",
                  "%rt_map = OpTypeRuntimeArray %uint", "%DescMap = OpTypeStruct %rt_map",
                  "%_ptr_SB_DescMap = OpTypePointer StorageBuffer %DescMap", "%_ptr_SB_uint = OpTypePointer StorageBuffer %uint"]
        dec += ["OpDecorate %images DescriptorSet 0", "OpDecorate %images Binding 0",
                "OpDecorate %samplers DescriptorSet 0", "OpDecorate %samplers Binding 1",
                "OpDecorate %desc_map DescriptorSet 0", "OpDecorate %desc_map Binding 2",
                "OpDecorate %DescMap Block", "OpMemberDecorate %DescMap 0 Offset 0",
                "OpDecorate %rt_map ArrayStride 4", "OpDecorate %desc_map NonWritable"]
        types += ["%v3float = OpTypeVector %float 3", "%v3int = OpTypeVector %int 3"]
        if gen.uses_tex_array:
            types += ["%img2da = OpTypeImage %float 2D 0 1 0 1 Unknown", "%simg2da = OpTypeSampledImage %img2da",
                      "%rt_imga = OpTypeRuntimeArray %img2da",
                      "%_ptr_UC_rt_imga = OpTypePointer UniformConstant %rt_imga",
                      "%_ptr_UC_imga = OpTypePointer UniformConstant %img2da"]
            dec += ["OpDecorate %images_arr DescriptorSet 0", "OpDecorate %images_arr Binding 3"]
        if gen.uses_storage3d:
            types += ["%simg3d_u = OpTypeImage %uint 3D 0 0 0 2 R32ui", "%rt_simg3 = OpTypeRuntimeArray %simg3d_u",
                      "%_ptr_UC_rt_simg3 = OpTypePointer UniformConstant %rt_simg3",
                      "%_ptr_UC_simg3 = OpTypePointer UniformConstant %simg3d_u"]
            if not gen.uses_storage:
                types += ["%_ptr_Image_uint = OpTypePointer Image %uint"]
            dec += ["OpDecorate %simages3d DescriptorSet 0", "OpDecorate %simages3d Binding 7"]
        if gen.uses_tex_3d:
            types += ["%img3d = OpTypeImage %float 3D 0 0 0 1 Unknown", "%simg3d = OpTypeSampledImage %img3d",
                      "%rt_img3 = OpTypeRuntimeArray %img3d",
                      "%_ptr_UC_rt_img3 = OpTypePointer UniformConstant %rt_img3",
                      "%_ptr_UC_img3 = OpTypePointer UniformConstant %img3d"]
            dec += ["OpDecorate %images_3d DescriptorSet 0", "OpDecorate %images_3d Binding 6"]
        if gen.uses_tex_1d:
            types += ["%img1d = OpTypeImage %float 1D 0 0 0 1 Unknown", "%simg1d = OpTypeSampledImage %img1d",
                      "%rt_img1 = OpTypeRuntimeArray %img1d",
                      "%_ptr_UC_rt_img1 = OpTypePointer UniformConstant %rt_img1",
                      "%_ptr_UC_img1 = OpTypePointer UniformConstant %img1d"]
            dec += ["OpDecorate %images_1d DescriptorSet 0", "OpDecorate %images_1d Binding 5"]
        if gen.uses_storage:
            types += ["%simg2d_u = OpTypeImage %uint 2D 0 0 0 2 R32ui", "%rt_simg = OpTypeRuntimeArray %simg2d_u",
                      "%_ptr_UC_rt_simg = OpTypePointer UniformConstant %rt_simg",
                      "%_ptr_UC_simg = OpTypePointer UniformConstant %simg2d_u",
                      "%_ptr_Image_uint = OpTypePointer Image %uint", "%v4uint = OpTypeVector %uint 4"]
            dec += ["OpDecorate %simages DescriptorSet 0", "OpDecorate %simages Binding 4"]
    if sg:
        types += ["%_ptr_Input_uint = OpTypePointer Input %uint"]
        dec += ["OpDecorate %sg_lane BuiltIn SubgroupLocalInvocationId"]
        if gen.stage == "ps":
            dec += ["OpDecorate %sg_lane Flat"]
        gen.consts.setdefault(3, "%c3")
    if vs:
        gen.consts.setdefault(3, "%c3")              # the instance index goes to v3
    if gen.uses_clip:
        gen.cu(gen.clip_count)
    if gen.uses_cull:
        gen.cu(gen.cull_count)
    if ms_blk:
        gen.cu(0)
        if ms_clip_member is not None:
            gen.cu(ms_clip_member)
        if ms_cull_member is not None:
            gen.cu(ms_cull_member)
    gen.consts.setdefault(0, "%c0")
    for v, name in sorted(gen.consts.items()):
        types.append(f"{name} = OpConstant %uint {v}")
    if gen.desc_fn:
        types.append("%fn_desc_lookup = OpTypeFunction %uint " + " ".join(["%uint"] * 8))
    if gen.uses_ballot:
        types.append("%v4uint_bal = OpTypeVector %uint 4")       # dedupe_types merges it with any other uvec4
    if gen.uses_bvh and not any(t.startswith("%v3float = ") for t in types):
        types.append("%v3float = OpTypeVector %float 3")
    gvars = [f"{n} = OpVariable %_ptr_Output_v4 Output" for n in gen.outputs]
    # Push constants: s0..s31 user data, then the 64-bit host address of the Xbox GPU VA window (heap base).
    types += [f"%n_ud = OpConstant %uint {USER_SGPRS}", "%arr_ud = OpTypeArray %uint %n_ud",
              "%UserData = OpTypeStruct %arr_ud %uint %uint %uint %uint",
              "%_ptr_PC_UserData = OpTypePointer PushConstant %UserData",
              "%_ptr_PC_uint = OpTypePointer PushConstant %uint"]
    dec += ["OpDecorate %arr_ud ArrayStride 4", "OpDecorate %UserData Block", "OpMemberDecorate %UserData 0 Offset 0",
            f"OpMemberDecorate %UserData 1 Offset {4 * USER_SGPRS}",
            f"OpMemberDecorate %UserData 2 Offset {4 * USER_SGPRS + 4}",
            f"OpMemberDecorate %UserData 3 Offset {4 * USER_SGPRS + 8}",
            f"OpMemberDecorate %UserData 4 Offset {4 * USER_SGPRS + 12}"]
    gvars += ["%user_data = OpVariable %_ptr_PC_UserData PushConstant"]
    if gen.uses_fhalf:
        types += ["%fhalf_g = OpConstant %float 0.5"]
    if gen.uses_f64:
        types += ["%double = OpTypeFloat 64", "%v2uint_d = OpTypeVector %uint 2"]
    if gen.uses_ballot:
        types += ["%v4uint_bal = OpTypeVector %uint 4"]
    if gen.uses_store_views:
        u, si = Gen.int_format_masks()
        types += ["%v4uint_st = OpTypeVector %uint 4", "%v4int_st = OpTypeVector %int 4",
                  f"%fmt_mask_arr = OpTypeArray %uint {gen.cu(16)}",
                  "%_ptr_Private_fmt = OpTypePointer Private %fmt_mask_arr",
                  "%_ptr_Private_uint = OpTypePointer Private %uint"]
        for nm, words in (("uint", u), ("sint", si)):
            ids = " ".join(gen.cu(w) for w in words)
            types += [f"%fmt_{nm}_init = OpConstantComposite %fmt_mask_arr {ids}"]
            gvars += [f"%fmt_{nm}_mask = OpVariable %_ptr_Private_fmt Private %fmt_{nm}_init"]
        for di in sorted(gen.uses_store_views):
            _, sdim, arr, _ = next(v for v in Gen.STORE_DIMS.values() if v[0] == di)
            for cls, st in enumerate(("%float", "%uint", "%int")):
                types += [f"%stimg{cls}_{di} = OpTypeImage {st} {sdim} 0 {arr} 0 2 Unknown",
                          f"%rt_stv{cls}_{di} = OpTypeRuntimeArray %stimg{cls}_{di}",
                          f"%_ptr_UC_rt_stv{cls}_{di} = OpTypePointer UniformConstant %rt_stv{cls}_{di}",
                          f"%_ptr_UC_stv{cls}_{di} = OpTypePointer UniformConstant %stimg{cls}_{di}"]
                dec += [f"OpDecorate %stv{cls}_{di} DescriptorSet 0", f"OpDecorate %stv{cls}_{di} Binding {8 + 3 * di + cls}"]
                gvars += [f"%stv{cls}_{di} = OpVariable %_ptr_UC_rt_stv{cls}_{di} UniformConstant"]
    if gsx:
        nst = gen.gs["maxv"] * gen.gs["nreg"]
        types += [f"%gs_arr_in = OpTypeArray %v4float {gen.cu(3)}", "%_ptr_Input_gs_arr = OpTypePointer Input %gs_arr_in",
                  "%_ptr_Output_v4_gs = OpTypePointer Output %v4float",
                  f"%gs_stage_arr = OpTypeArray %v4float {gen.cu(nst)}",
                  "%_ptr_Function_gs_stage = OpTypePointer Function %gs_stage_arr",
                  "%_ptr_Function_v4_gs = OpTypePointer Function %v4float",
                  "%null_gs_stage = OpConstantNull %gs_stage_arr"]
        gvars += [("%gs_in_pos" if sv == 1 else f"%gs_in{r}") + " = OpVariable %_ptr_Input_gs_arr Input"
                  for r, sv in gen.gs["in"]]
        gvars += [("%gs_out_pos" if sv == 1 else f"%gs_out{r}") + " = OpVariable %_ptr_Output_v4_gs Output"
                  for r, sv in gen.gs["out"]]
    if ms:
        maxv, maxp = gen.cu(gen.ms["maxv"]), gen.cu(gen.ms["maxp"])
        types += ["%v3uint = OpTypeVector %uint 3", "%_ptr_Input_v3u = OpTypePointer Input %v3uint",
                  "%_ptr_Input_uint_ms = OpTypePointer Input %uint",
                  "%_ptr_WG_uint_ms = OpTypePointer Workgroup %uint",
                  f"%ms_arr_v4 = OpTypeArray %v4float {maxv}", "%_ptr_Out_ms_arr_v4 = OpTypePointer Output %ms_arr_v4",
                  "%_ptr_Out_v4_ms = OpTypePointer Output %v4float",
                  f"%ms_arr_idx = OpTypeArray %v3uint {maxp}", "%_ptr_Out_ms_arr_idx = OpTypePointer Output %ms_arr_idx",
                  "%_ptr_Out_v3u_ms = OpTypePointer Output %v3uint",
                  "%ms_cull_blk = OpTypeStruct %bool",
                  f"%ms_arr_cull = OpTypeArray %ms_cull_blk {maxp}",
                  "%_ptr_Out_ms_arr_cull = OpTypePointer Output %ms_arr_cull",
                  "%_ptr_Out_bool_ms = OpTypePointer Output %bool",
                  "%_ptr_Function_v4_ms = OpTypePointer Function %v4float",
                  "%_ptr_Function_bool_ms = OpTypePointer Function %bool",
                  "%null_v4_ms = OpConstantNull %v4float"]
        if ms_blk:
            types += ["%_ptr_Function_float = OpTypePointer Function %float"]
            blk_fields = ["%v4float"]
            if gen.uses_clip:
                types += [f"%ms_arr_clip = OpTypeArray %float {gen.cu(gen.clip_count)}",
                          "%_ptr_Fn_ms_clip = OpTypePointer Function %ms_arr_clip",
                          "%_ptr_Out_ms_clip = OpTypePointer Output %ms_arr_clip",
                          "%null_ms_clip = OpConstantNull %ms_arr_clip"]
                blk_fields.append("%ms_arr_clip")
            if gen.uses_cull:
                types += [f"%ms_arr_cull_d = OpTypeArray %float {gen.cu(gen.cull_count)}",
                          "%_ptr_Fn_ms_cull_d = OpTypePointer Function %ms_arr_cull_d",
                          "%_ptr_Out_ms_cull_d = OpTypePointer Output %ms_arr_cull_d",
                          "%null_ms_cull_d = OpConstantNull %ms_arr_cull_d"]
                blk_fields.append("%ms_arr_cull_d")
            types += [f"%ms_vtx_blk = OpTypeStruct {' '.join(blk_fields)}",
                      f"%ms_arr_vtx_blk = OpTypeArray %ms_vtx_blk {maxv}",
                      "%_ptr_Out_ms_arr_vtx_blk = OpTypePointer Output %ms_arr_vtx_blk"]
        gvars += ["%wg_id = OpVariable %_ptr_Input_v3u Input", "%num_wg = OpVariable %_ptr_Input_v3u Input",
                  "%local_index = OpVariable %_ptr_Input_uint_ms Input",
                  "%ms_counts = OpVariable %_ptr_WG_uint_ms Workgroup",
                  ("%ms_out_vtx = OpVariable %_ptr_Out_ms_arr_vtx_blk Output" if ms_blk else
                   "%ms_out_pos = OpVariable %_ptr_Out_ms_arr_v4 Output"),
                  "%ms_prim_idx = OpVariable %_ptr_Out_ms_arr_idx Output",
                  "%ms_cull = OpVariable %_ptr_Out_ms_arr_cull Output"]
        gvars += [f"%ms_out_param{n} = OpVariable %_ptr_Out_ms_arr_v4 Output" for n in sorted(gen.ms_params)]
        if gen.uses_lds:
            types += [f"%lds_arr = OpTypeArray %uint {gen.cu(gen.lds_dwords)}",
                      "%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr", "%_ptr_WG_uint = OpTypePointer Workgroup %uint"]
            gvars += ["%lds = OpVariable %_ptr_WG_lds Workgroup"]
    if gen.emulate:
        types += [f"%x_arr = OpTypeArray %uint {gen.cu(x_sink + 1)}",
                  "%_ptr_WG_x_arr = OpTypePointer Workgroup %x_arr", "%_ptr_WG_uint_x = OpTypePointer Workgroup %uint"]
        gvars += ["%x_buf = OpVariable %_ptr_WG_x_arr Workgroup"]
    if cs:
        types += ["%v3uint = OpTypeVector %uint 3", "%_ptr_Input_v3u = OpTypePointer Input %v3uint"]
        gvars += ["%wg_id = OpVariable %_ptr_Input_v3u Input", "%local_id = OpVariable %_ptr_Input_v3u Input"]
        if gen.uses_lds:
            types += [f"%lds_arr = OpTypeArray %uint {gen.cu(gen.lds_dwords)}",
                      "%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr", "%_ptr_WG_uint = OpTypePointer Workgroup %uint"]
            gvars += ["%lds = OpVariable %_ptr_WG_lds Workgroup"]
    if gen.frag_inputs:
        dec += ["OpDecorate %frag_coord BuiltIn FragCoord"]
        gvars += ["%frag_coord = OpVariable %_ptr_Input_v4 Input"]
        if "front" in gen.frag_inputs.values():
            types += ["%_ptr_Input_bool = OpTypePointer Input %bool"]
            dec += ["OpDecorate %front_facing BuiltIn FrontFacing"]
            gvars += ["%front_facing = OpVariable %_ptr_Input_bool Input"]
        if bary:
            if not tex:
                types += ["%v3float = OpTypeVector %float 3"]
            types += ["%_ptr_Input_v3f = OpTypePointer Input %v3float"]
            dec += ["OpDecorate %bary BuiltIn BaryCoordKHR"]
            gvars += ["%bary = OpVariable %_ptr_Input_v3f Input"]
    if gen.vertex_inputs:
        types += ["%v4uint_in = OpTypeVector %uint 4", "%v4int_in = OpTypeVector %int 4",
                  "%_ptr_Input_v4u = OpTypePointer Input %v4uint_in", "%_ptr_Input_v4i = OpTypePointer Input %v4int_in"]
        for reg, ctype in gen.vertex_inputs:
            dec.append(f"OpDecorate %vin{reg} Location {reg}")
    if gen.writes_depth:
        types += ["%_ptr_Output_float = OpTypePointer Output %float"]
        dec += ["OpDecorate %out_depth BuiltIn FragDepth"]
        gvars += ["%out_depth = OpVariable %_ptr_Output_float Output"]
    if vs:
        if gen.uses_clip or gen.uses_cull:
            types += ["%_ptr_Output_float = OpTypePointer Output %float"]
        if gen.uses_clip:
            types += [f"%arr_clip = OpTypeArray %float {gen.cu(gen.clip_count)}",
                      "%_ptr_Out_arr_clip = OpTypePointer Output %arr_clip"]
            gvars += ["%out_clip = OpVariable %_ptr_Out_arr_clip Output"]
        if gen.uses_cull:
            types += [f"%arr_cull = OpTypeArray %float {gen.cu(gen.cull_count)}",
                      "%_ptr_Out_arr_cull = OpTypePointer Output %arr_cull"]
            gvars += ["%out_cull = OpVariable %_ptr_Out_arr_cull Output"]
        if gen.uses_layer or gen.uses_viewport:
            types += ["%_ptr_Output_int = OpTypePointer Output %int"]
            if gen.uses_layer:
                gvars += ["%out_layer = OpVariable %_ptr_Output_int Output"]
            if gen.uses_viewport:
                gvars += ["%out_viewport = OpVariable %_ptr_Output_int Output"]
    gvars += [f"{n} = OpVariable %_ptr_Input_v4 Input" for n in gen.inputs]
    if sg:
        gvars += ["%sg_lane = OpVariable %_ptr_Input_uint Input"]
    if tex:
        gvars += ["%images = OpVariable %_ptr_UC_rt_img UniformConstant",
                  "%samplers = OpVariable %_ptr_UC_rt_smp UniformConstant",
                  "%desc_map = OpVariable %_ptr_SB_DescMap StorageBuffer"]
        if gen.uses_tex_array:
            gvars += ["%images_arr = OpVariable %_ptr_UC_rt_imga UniformConstant"]
        if gen.uses_storage:
            gvars += ["%simages = OpVariable %_ptr_UC_rt_simg UniformConstant"]
        if gen.uses_tex_1d:
            gvars += ["%images_1d = OpVariable %_ptr_UC_rt_img1 UniformConstant"]
        if gen.uses_tex_3d:
            gvars += ["%images_3d = OpVariable %_ptr_UC_rt_img3 UniformConstant"]
        if gen.uses_storage3d:
            gvars += ["%simages3d = OpVariable %_ptr_UC_rt_simg3 UniformConstant"]
    if vs:
        gvars.append("%vertex_index = OpVariable %_ptr_Input_int Input")
        gvars.append("%instance_index = OpVariable %_ptr_Input_int Input")
    for reg, ctype in gen.vertex_inputs:
        ptr = {1: "%_ptr_Input_v4u", 2: "%_ptr_Input_v4i"}.get(ctype, "%_ptr_Input_v4")
        gvars.append(f"%vin{reg} = OpVariable {ptr} Input")
    fn = ["%main = OpFunction %void None %fn", "%entry = OpLabel",
          # Registers start at zero (deterministic); real inputs are written explicitly at entry.
          "%vgpr = OpVariable %_ptr_Function_arr_v Function %null_v",
          "%sgpr = OpVariable %_ptr_Function_arr_s Function %null_s",
          "%vcc = OpVariable %_ptr_Function_uint Function %c0", "%m0 = OpVariable %_ptr_Function_uint Function %c0",
          "%vcc_hi = OpVariable %_ptr_Function_uint Function %c0",
          "%scc = OpVariable %_ptr_Function_uint Function %c0",
          "%active = OpVariable %_ptr_Function_bool Function %true"]
    if gen.emulate:
        fn += ["%x_it = OpVariable %_ptr_Function_uint Function %c0",
               "%x_bar = OpVariable %_ptr_Function_bool Function %false",
               "%x_chain = OpVariable %_ptr_Function_bool Function %false"]
        fn += [f"{var} = OpVariable {ptr} Function" for var, ptr in gen.x_spills]
        if gen.uses_perm:
            fn.append("%x_perm = OpVariable %_ptr_Function_uint Function %c0")
    # User data: s0..s31 are loaded at entry from a push-constant block (the driver's user-SGPR setup).
    if gen.uses_heap:
        gen.consts.setdefault(1, "%c1")
        gen.consts.setdefault(2, "%c2")
        fn += ["%ud_base_lo_p = OpAccessChain %_ptr_PC_uint %user_data %c1",
               "%ud_base_hi_p = OpAccessChain %_ptr_PC_uint %user_data %c2"]
    if gen.uses_codebase:
        fn += ["%ud_code_lo_p = OpAccessChain %_ptr_PC_uint %user_data %c3",
               "%ud_code_hi_p = OpAccessChain %_ptr_PC_uint %user_data %c4"]
    for i in range(USER_SGPRS):
        fn += [f"%ud{i}p = OpAccessChain %_ptr_PC_uint %user_data %c0 {gen.cu(i)}",
               f"%ud{i} = OpLoad %uint %ud{i}p",
               f"%ud{i}s = OpAccessChain %_ptr_Function_uint %sgpr {gen.cu(i)}",
               f"OpStore %ud{i}s %ud{i}"]
    if vs:
        fn += ["%vi = OpLoad %int %vertex_index", "%viu = OpBitcast %uint %vi",
               "%v0p = OpAccessChain %_ptr_Function_uint %vgpr %c0", "OpStore %v0p %viu"]
        # v3 = instance id. Over the corpus, the 494 VS that read v3 before writing it are exactly ones whose input
        # signature declares SV_InstanceID, and no VS reads v1 or v2 on entry.
        fn += ["%ii = OpLoad %int %instance_index", "%iiu = OpBitcast %uint %ii",
               f"%v3p = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(3)}", "OpStore %v3p %iiu"]
    if gsx:
        fn[2:2] = ["%gs_stage = OpVariable %_ptr_Function_gs_stage Function %null_gs_stage",
                   "%gs_emitted = OpVariable %_ptr_Function_uint Function %c0"]
        # Input vertex i at dword 64*i (v0 = vtx0 | vtx1 << 16, v1 = vtx2); s3[15:8] = 64 lanes of work (the
        # prologue's s_bfm/s_cselect then leave EXEC full), s3[23:16] = 0.
        fn += [f"%gv0p = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(0)}", f"OpStore %gv0p {gen.cu(64 << 16)}",
               f"%gv1p = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(1)}", f"OpStore %gv1p {gen.cu(128)}",
               f"%gs3p = OpAccessChain %_ptr_Function_uint %sgpr {gen.cu(3)}", f"OpStore %gs3p {gen.cu(0x4000)}"]
    if ms:
        fn[2:2] = ["%ms_pos = OpVariable %_ptr_Function_v4_ms Function %null_v4_ms",
                   "%ms_prim = OpVariable %_ptr_Function_uint Function %c0",
                   "%ms_has_vtx = OpVariable %_ptr_Function_bool_ms Function %false",
                   "%ms_has_prim = OpVariable %_ptr_Function_bool_ms Function %false"] + \
            (["%ms_clip = OpVariable %_ptr_Fn_ms_clip Function %null_ms_clip"] if gen.uses_clip else []) + \
            (["%ms_cull_dist = OpVariable %_ptr_Fn_ms_cull_d Function %null_ms_cull_d"] if gen.uses_cull else []) + \
            [f"%ms_param{n} = OpVariable %_ptr_Function_v4_ms Function %null_v4_ms" for n in sorted(gen.ms_params)]
        # v5 = first launch vertex ID of the lane = linear group index * NumThreads + local index (the game's mesh
        # entry does v5 >> 7 on lane 0 to get the group index; NumThreads is 128 in this corpus).
        # s3 = merged-wave info with the wave id in [27:24].
        fn += ["%wgi = OpLoad %v3uint %wg_id", "%nwg = OpLoad %v3uint %num_wg", "%lidx = OpLoad %uint %local_index",
               "%wgx = OpCompositeExtract %uint %wgi 0", "%wgy = OpCompositeExtract %uint %wgi 1",
               "%wgz = OpCompositeExtract %uint %wgi 2", "%nwx = OpCompositeExtract %uint %nwg 0",
               "%nwy = OpCompositeExtract %uint %nwg 1", "%nxy = OpIMul %uint %nwx %nwy",
               "%wz = OpIMul %uint %wgz %nxy", "%wy = OpIMul %uint %wgy %nwx", "%wyz = OpIAdd %uint %wy %wz",
               "%wlin = OpIAdd %uint %wgx %wyz",
               f"%wbase = OpIMul %uint %wlin {gen.cu(gen.ms['threads'])}", "%v5v = OpIAdd %uint %wbase %lidx",
               f"%v5p = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(5)}", "OpStore %v5p %v5v",
               f"%wave = OpShiftRightLogical %uint %lidx {gen.cu(6)}",
               f"%mwi = OpShiftLeftLogical %uint %wave {gen.cu(24)}",
               f"%s3p = OpAccessChain %_ptr_Function_uint %sgpr {gen.cu(3)}", "OpStore %s3p %mwi"]
    if cs:
        # After the user-data copy: workgroup ids overwrite s[user..], thread ids go to v0..v2.
        fn += ["%wgi = OpLoad %v3uint %wg_id", "%lci = OpLoad %v3uint %local_id"]
        for k, c in enumerate(gen.cs["tgid"]):
            r = gen.cs["user"] + k
            fn += [f"%wg{c} = OpCompositeExtract %uint %wgi {c}",
                   f"%wgp{c} = OpAccessChain %_ptr_Function_uint %sgpr {gen.cu(r)}", f"OpStore %wgp{c} %wg{c}"]
        for c in range(gen.cs["tid"] + 1):
            fn += [f"%li{c} = OpCompositeExtract %uint %lci {c}",
                   f"%lip{c} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(c)}", f"OpStore %lip{c} %li{c}"]
    if gen.frag_inputs:
        fn += ["%fc = OpLoad %v4float %frag_coord"]
        if bary:
            fn += ["%bc = OpLoad %v3float %bary"]
        for r, c in gen.frag_inputs.items():
            if str(c).startswith("bary"):
                k = 1 if c == "bary_i" else 2
                fn += [f"%bc{k} = OpCompositeExtract %float %bc {k}", f"%bcu{k} = OpBitcast %uint %bc{k}",
                       f"%bcp{k} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(r)}", f"OpStore %bcp{k} %bcu{k}"]
                continue
            if c == "front":
                fn += ["%ff = OpLoad %bool %front_facing", "%ffm1 = OpFNegate %float %f1",
                       "%ffv = OpSelect %float %ff %f1 %ffm1", "%ffu = OpBitcast %uint %ffv",
                       f"%ffp = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(r)}", "OpStore %ffp %ffu"]
                continue
            fn += [f"%fc{c} = OpCompositeExtract %float %fc {c}", f"%fcu{c} = OpBitcast %uint %fc{c}",
                   f"%fcp{c} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(r)}", f"OpStore %fcp{c} %fcu{c}"]
    for reg, ctype in gen.vertex_inputs:
        vt = {1: "%v4uint_in", 2: "%v4int_in"}.get(ctype, "%v4float")
        st = {1: "%uint", 2: "%int"}.get(ctype, "%float")
        fn.append(f"%vinv{reg} = OpLoad {vt} %vin{reg}")
        for c in range(4):
            vg = 4 + 4 * reg + c
            fn += [f"%vinc{reg}_{c} = OpCompositeExtract {st} %vinv{reg} {c}",
                   f"%vinb{reg}_{c} = OpBitcast %uint %vinc{reg}_{c}" if ctype != 1 else
                   f"%vinb{reg}_{c} = OpCopyObject %uint %vinc{reg}_{c}",
                   f"%vinp{reg}_{c} = OpAccessChain %_ptr_Function_uint %vgpr {gen.cu(vg)}",
                   f"OpStore %vinp{reg}_{c} %vinb{reg}_{c}"]
    # Spill pointers last: a spilled value's type may be declared anywhere above.
    types += [f"{ptr} = OpTypePointer Function {t}" for t, ptr in gen.x_ptr_types.items()]
    out = hdr + dec + types + gvars + (gen.desc_fn or []) + fn
    tree = cfg_structure.structure(gen.succ) if gen.blocks and STRUCTURED and not gen.emulate else None
    gen.control_flow = "structured" if tree is not None else "dispatch loop" if gen.blocks else "straight line"
    if gen.emulate:
        gen.control_flow = "emulated wave64 rounds"
    if tree is not None:
        # Real loops and conditionals (cfg_structure): the blocks nested by dominance, each behind a test of %pc,
        # and every natural loop as a loop that repeats while %pc names its header.
        lines = ["%pc = OpVariable %_ptr_Function_uint Function %c0",
                 "%done = OpVariable %_ptr_Function_bool Function %false"]
        body, count = [], [0]

        def label(prefix):
            count[0] += 1
            return f"%{prefix}{count[0]}"

        def place(items):
            for it in items:
                if it[0] == "block":
                    # Under its own label, as in the dispatch form: OpPhi in a body names it as a predecessor.
                    body.extend([f"OpBranch %blk{it[1]}", f"%blk{it[1]} = OpLabel"] + gen.blocks[it[1]])
                elif it[0] == "guard":
                    v, c, do, end = label("g_pc"), label("g_is"), label("g_do"), label("g_end")
                    body.extend([f"{v} = OpLoad %uint %pc", f"{c} = OpIEqual %bool {v} {gen.consts[it[1]]}",
                                 f"OpSelectionMerge {end} None", f"OpBranchConditional {c} {do} {end}",
                                 f"{do} = OpLabel"])
                    place(it[2])
                    body.extend([f"OpBranch {end}", f"{end} = OpLabel"])
                else:
                    head, top, cont, end = label("l_head"), label("l_body"), label("l_cont"), label("l_end")
                    v, again = label("l_pc"), label("l_again")
                    body.extend([f"OpBranch {head}", f"{head} = OpLabel", f"OpLoopMerge {end} {cont} None",
                                 f"OpBranch {top}", f"{top} = OpLabel"])
                    place(it[2])
                    body.extend([f"OpBranch {cont}", f"{cont} = OpLabel", f"{v} = OpLoad %uint %pc",
                                 f"{again} = OpIEqual %bool {v} {gen.consts[it[1]]}",
                                 f"OpBranchConditional {again} {head} {end}", f"{end} = OpLabel"])

        place(tree)
    elif gen.emulate:
        lines = x_scheduler(gen, x_threads)
        body = []
        for i, blines in sorted(list(enumerate(gen.blocks)) + list(gen.x_segs.items())):
            body.append(f"%blk{i} = OpLabel")
            body += blines
            body.append("OpBranch %sw_merge")
        body += x_scheduler_tail(gen)
    elif gen.blocks:
        # Dispatch loop: any CFG (forward/backward branches) in structured SPIR-V form. Used for the irreducible
        # graphs cfg_structure refuses (a loop with two entries), and with --dispatch-loop.
        n = len(gen.blocks)
        lines = ["%pc = OpVariable %_ptr_Function_uint Function %c0",
                 "%done = OpVariable %_ptr_Function_bool Function %false",
                 "OpBranch %loop_head", "%loop_head = OpLabel", "OpLoopMerge %loop_exit %loop_cont None",
                 "OpBranch %dispatch", "%dispatch = OpLabel", "%pcv = OpLoad %uint %pc",
                 "OpSelectionMerge %sw_merge None",
                 "OpSwitch %pcv %sw_default " + " ".join(f"{i} %blk{i}" for i in range(n))]
        body = []
        for i, blines in enumerate(gen.blocks):
            body.append(f"%blk{i} = OpLabel")
            body += blines
            body.append("OpBranch %sw_merge")
        body += ["%sw_default = OpLabel", "OpStore %done %true", "OpBranch %sw_merge",
                 "%sw_merge = OpLabel", "OpBranch %loop_cont", "%loop_cont = OpLabel",
                 "%dn = OpLoad %bool %done", "OpBranchConditional %dn %loop_exit %loop_head",
                 "%loop_exit = OpLabel"]
    if gen.blocks:
        # %pc/%done must be declared with the other Function variables, at the top of the entry block,
        # before the vertex-shader prologue.
        fn_vars = lines[:2]
        flow = lines[2:]
        last_var = max(i for i, l in enumerate(fn) if " = OpVariable " in l)
        fn = fn[:last_var + 1] + fn_vars + fn[last_var + 1:]
        if pops:
            fn.append("OpBeginInvocationInterlockEXT")   # end of the entry block: uniform, executed once
        out = hdr + dec + types + gvars + (gen.desc_fn or []) + fn
        gen.body =["         " + l if not (l.startswith("%") and " = OpLabel" in l) else l for l in flow + body]
    end_lock = "         OpEndInvocationInterlockEXT\n" if pops else ""
    if ms:
        # All invocations reach the end: wait for the ALLOC_REQ write, declare the counts once (uniform), then each
        # thread writes the vertex and primitive it exported (index = LocalInvocationIndex).
        c = gen.cu
        L = ["OpControlBarrier %c2 %c2 " + c(0x108),
             "%msc = OpLoad %uint %ms_counts",
             f"%msv0 = OpBitwiseAnd %uint %msc {c(0xFFF)}", f"%msp1 = OpShiftRightLogical %uint %msc {c(12)}",
             f"%msp0 = OpBitwiseAnd %uint %msp1 {c(0xFFF)}",
             f"%msv = OpExtInst %uint %glsl UMin %msv0 {c(gen.ms['maxv'])}",
             f"%msp = OpExtInst %uint %glsl UMin %msp0 {c(gen.ms['maxp'])}",
             "OpSetMeshOutputsEXT %msv %msp",
             "%mli = OpLoad %uint %local_index",
             "%mhv = OpLoad %bool %ms_has_vtx", "%mvi = OpULessThan %bool %mli %msv",
             "%mwv = OpLogicalAnd %bool %mhv %mvi",
             "OpSelectionMerge %ms_vend None", "OpBranchConditional %mwv %ms_vdo %ms_vend",
             "%ms_vdo = OpLabel",
             "%mpv = OpLoad %v4float %ms_pos",
             ("%mpp = OpAccessChain %_ptr_Out_v4_ms %ms_out_vtx %mli %c0" if ms_blk else
              "%mpp = OpAccessChain %_ptr_Out_v4_ms %ms_out_pos %mli"),
             "OpStore %mpp %mpv"]
        if gen.uses_clip:
            L += ["%mcv = OpLoad %ms_arr_clip %ms_clip",
                  f"%mcvp = OpAccessChain %_ptr_Out_ms_clip %ms_out_vtx %mli {c(ms_clip_member)}",
                  "OpStore %mcvp %mcv"]
        if gen.uses_cull:
            L += ["%mkv = OpLoad %ms_arr_cull_d %ms_cull_dist",
                  f"%mkvp = OpAccessChain %_ptr_Out_ms_cull_d %ms_out_vtx %mli {c(ms_cull_member)}",
                  "OpStore %mkvp %mkv"]
        for n in sorted(gen.ms_params):
            L += [f"%mq{n} = OpLoad %v4float %ms_param{n}",
                  f"%mqp{n} = OpAccessChain %_ptr_Out_v4_ms %ms_out_param{n} %mli", f"OpStore %mqp{n} %mq{n}"]
        L += ["OpBranch %ms_vend", "%ms_vend = OpLabel",
              "%mhp = OpLoad %bool %ms_has_prim", "%mpi = OpULessThan %bool %mli %msp",
              "%mwp = OpLogicalAnd %bool %mhp %mpi",
              "OpSelectionMerge %ms_pend None", "OpBranchConditional %mwp %ms_pdo %ms_pend",
              "%ms_pdo = OpLabel",
              "%mpr = OpLoad %uint %ms_prim",
              f"%mi0 = OpBitwiseAnd %uint %mpr {c(0x1FF)}",
              f"%mr1 = OpShiftRightLogical %uint %mpr {c(10)}", f"%mi1 = OpBitwiseAnd %uint %mr1 {c(0x1FF)}",
              f"%mr2 = OpShiftRightLogical %uint %mpr {c(20)}", f"%mi2 = OpBitwiseAnd %uint %mr2 {c(0x1FF)}",
              "%mix = OpCompositeConstruct %v3uint %mi0 %mi1 %mi2",
              "%mixp = OpAccessChain %_ptr_Out_v3u_ms %ms_prim_idx %mli", "OpStore %mixp %mix",
              f"%mnb = OpShiftRightLogical %uint %mpr {c(31)}", "%mnull = OpINotEqual %bool %mnb %c0",
              f"%mcp = OpAccessChain %_ptr_Out_bool_ms %ms_cull %mli {c(0)}", "OpStore %mcp %mnull",
              "OpBranch %ms_pend", "%ms_pend = OpLabel"]
        end_lock = "".join(("         " + l if " = OpLabel" not in l else l) + "\n" for l in L) + end_lock
    if gen.uses_active and gen.stage == "ps":
        # A pixel lane switched off by v_cmpx at the final export writes nothing: equivalent to discard.
        tail = end_lock + ("         %act_end = OpLoad %bool %active\n"
                "         OpSelectionMerge %live None\n"
                "         OpBranchConditional %act_end %live %dead\n"
                "%dead = OpLabel\n         OpKill\n"
                "%live = OpLabel\n         OpReturn\n         OpFunctionEnd\n")
    else:
        tail = end_lock + "         OpReturn\n         OpFunctionEnd\n"
    return split_big_switch(no_contraction(dedupe_types(
        "\n".join("         " + l if not l.startswith("%") or " = " not in l else l for l in out) +
        "\n" + "\n".join(gen.body) + "\n" + tail)))


# SPIR-V caps an instruction at 65,535 words. The dispatch loop's OpSwitch has one (literal, label) pair per block,
# and with --wave64-emulate the two ScreenProbes compute shaders reach ~37k blocks (74k words). Past the limit the
# switch becomes two levels: an outer one on pc / SWITCH_CHUNK picks a chunk; each chunk is its own selection
# construct (inner OpSwitch on pc, merge %swm<k>), and that chunk's blocks branch to %swm<k>, which goes on to
# %sw_merge. Every block keeps its label and code, so only the routing changes.
SWITCH_CHUNK = 16000      # spirv-val also caps a switch at 16,383 (literal, label) pairs
_BIG_SWITCH = re.compile(r"^(\s*)OpSwitch %pcv %sw_default ((?:\d+ %\w+ ?)+)$", re.M)


def split_big_switch(text):
    m = _BIG_SWITCH.search(text)
    if not m:
        return text
    pairs = m.group(2).split()
    cases = [(int(pairs[i]), pairs[i + 1]) for i in range(0, len(pairs), 2)]
    if len(cases) <= 16383 and 3 + 2 * len(cases) <= 65535:
        return text
    ind = m.group(1)
    chunk_of = {lbl: v // SWITCH_CHUNK for v, lbl in cases}
    chunks = sorted(set(chunk_of.values()))
    outer = f"{ind}OpSwitch %pcc %sw_default " + " ".join(f"{k} %swc{k}" for k in chunks)
    heads = []
    for k in chunks:
        inner = " ".join(f"{v} {lbl}" for v, lbl in cases if chunk_of[lbl] == k)
        heads += [f"%swc{k} = OpLabel", f"{ind}OpSelectionMerge %swm{k} None", f"{ind}OpSwitch %pcv %swm{k} {inner}"]
    # OpSelectionMerge must stay right before its switch, so the chunk index is computed above it
    pre = text[:m.start()]
    sel = pre.rfind("OpSelectionMerge %sw_merge None")
    sel = pre.rfind("\n", 0, sel) + 1
    pre = pre[:sel] + f"{ind}%pcc = OpUDiv %uint %pcv %c_swchunk\n" + pre[sel:]
    lines = pre.split("\n") + [outer] + heads + text[m.end():].split("\n")[1:]
    # route each block's exits to its chunk's merge: a block runs from its label to the next case label
    out, cur = [], None
    for line in lines:
        st = line.strip()
        if st.endswith("= OpLabel"):
            lbl = st.split(" = ")[0]
            if lbl in chunk_of:
                cur = chunk_of[lbl]
            elif lbl in ("%sw_default", "%sw_merge") or lbl.startswith("%swc"):
                cur = None
        if lbl_is_sw_default(st):
            out += [f"%swm{k} = OpLabel\n{ind}OpBranch %sw_merge" for k in chunks]   # after every chunk's blocks
        if cur is not None and st == "OpBranch %sw_merge":
            line = line.replace("OpBranch %sw_merge", f"OpBranch %swm{cur}")
        out.append(line)
    text = "\n".join(out)
    # the chunk-size constant, next to the other uint constants
    first = re.search(r"^\s*%\w+ = OpConstant %uint \d+$", text, re.M)
    return text[:first.start()] + f"%c_swchunk = OpConstant %uint {SWITCH_CHUNK}\n" + text[first.start():]


def lbl_is_sw_default(st):
    return st == "%sw_default = OpLabel"


_FLOAT_ARITH = re.compile(r"^\s*(%\w+) = Op(?:FMul|FAdd|FSub|FDiv|FNegate|FRem|FMod) ", re.M)


def no_contraction(text):
    """Decorate every float arithmetic instruction NoContraction (multiply, add, subtract, divide, negate,
    remainder). The hardware rounds a v_mul and the v_add after it separately (its fused instructions are
    emitted as Fma), but a driver may fuse or otherwise rewrite an undecorated expression, and whether it does
    depends on how much of the code its optimiser can see at once: with real control flow instead of the
    dispatch loop, one shader's output moved by a 16-bit-float step in 49 of 4,096 pixels on an AMD GPU (a
    multiply and an add), and two more by one step in one pixel on Mesa's software driver (divide or negate)."""
    ids = _FLOAT_ARITH.findall(text)
    at = text.find("\n%void = OpTypeVoid")
    if not ids or at < 0:
        return text
    return text[:at] + "".join(f"\n         OpDecorate {i} NoContraction" for i in ids) + text[at:]


_STAGES = None


def stage_of(name, path):
    """Shader stage from the carve index (the DXIL program header in each container), not the name."""
    global _STAGES
    if _STAGES is None:
        import json
        idx = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "index.json")
        _STAGES = {}
        if os.path.isfile(idx):
            for e in json.load(open(idx, encoding="utf-8")):
                _STAGES[e["file"][:-5]] = e["stage"]
    return _STAGES.get(name) or (name[:2].lower() if name[2:3] == "_" else "?")


# Wave64 markers in the instruction text. Besides EXEC/VCC forms, 64-bit lane-mask SOP1/SOP2 ops
# (bcnt/ff/wqm/bfm) imply a 64-lane wave even when the mask lives in ordinary SGPR pairs (#210 part 2).
WAVE64_RE = re.compile(
    r"exec_hi|v_mbcnt_hi_u32_b32|_saveexec_b64|_wrexec_b64|"
    r"s_bcnt[01]_i32_b64|s_ff[01]_i32_b64|s_wqm_b64|s_bfm_b64"
)


def fold_far_branches(insns):
    """s_getpc_b64 s[a:a+1]; s_add_u32 sa, sa, K; s_addc_u32 sa+1, sa+1, 0|-1; s_setpc_b64 s[a:a+1] is a far
    branch (s_branch only reaches +-32K dwords): the target is the address after s_getpc plus K. Fold the four
    instructions into one s_branch with an explicit byte target so the CFG sees an ordinary branch."""
    out, i = [], 0
    while i < len(insns):
        w = insns[i:i + 4]
        ops4 = [base_op(x["op"]) for x in w]
        if len(w) == 4 and ops4 in (["s_getpc_b64", "s_add_u32", "s_addc_u32", "s_setpc_b64"],
                                    ["s_getpc_b64", "s_sub_u32", "s_subb_u32", "s_setpc_b64"]):
            g, ad, ac, sp = w
            r = g["args"][0].get("r") if g["args"] and g["args"][0]["k"] == "sgpr" else None
            ok = r is not None and                 [x.get("r") for x in ad["args"][:2]] == [r, r] and ad["args"][2]["k"] == "imm" and                 [x.get("r") for x in ac["args"][:2]] == [r + 1, r + 1] and ac["args"][2]["k"] == "imm" and                 sp["args"] and sp["args"][0].get("r") == r
            if ok:
                k = ad["args"][2]["v"] & 0xFFFFFFFF
                hi = ac["args"][2]["v"] & 0xFFFFFFFF
                if hi == 0xFFFFFFFF:
                    k -= 1 << 32
                elif hi != 0:
                    ok = False
                if ops4[1] == "s_sub_u32":          # backward far branch (loop back-edge)
                    k = -k
            if ok:
                out.append({"op": "s_branch", "args": [], "mods": [], "pc": g["pc"],
                            "enc": "".join(x["enc"] for x in w), "target": g["pc"] + 4 + k})
                i += 4
                continue
        out.append(insns[i])
        i += 1
    return out


def reachable_code(insns):
    """Instructions reachable from the first one, following fall-through and branch targets. Fall-through stops
    after s_endpgm, s_branch and s_setpc_b64. Raises Unsupported if a branch target isn't an instruction start."""
    by_pc = {i["pc"]: k for k, i in enumerate(insns)}
    seen, work = set(), [0]
    while work:
        k = work.pop()
        while k is not None and k < len(insns) and k not in seen:
            seen.add(k)
            ins = insns[k]
            op = base_op(ins["op"])
            nxt = ins["pc"] + len(ins["enc"]) // 2
            if op in BRANCHES:
                imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
                v = (imm["v"] & 0xFFFF) if imm else 0
                tgt = ins["target"] if "target" in ins else nxt + 4 * (v - 0x10000 if v & 0x8000 else v)
                if tgt not in by_pc:
                    raise Unsupported("branch into data")
                work.append(by_pc[tgt])
                if op == "s_branch":
                    break
            if op in ("s_endpgm", "s_setpc_b64"):
                break
            k = by_pc.get(nxt)
            if k is None and op not in ("s_endpgm",):
                break
    return [insns[k] for k in sorted(seen)]


def read_before_write_vgprs(insns):
    """VGPRs a shader reads before writing them: for a fetch-calling VS these are the fetched attributes."""
    written, first = set(), []
    for ins in insns:
        a, op = ins["args"], ins["op"]
        if not a:
            continue
        reads_all = op.startswith(("exp", "buffer_store", "image_store", "v_cmp", "v_cmpx", "ds_write", "global_store"))
        regs = []
        for x in (a if reads_all else a[1:]):
            if x["k"] == "vgpr":
                regs += range(x["r"], x["r"] + x.get("n", 1))
            elif x["k"] == "vlist":
                regs += x["r"]
        if base_op(op) in ("v_fmac_f32", "v_mac_f32", "v_fmac_f16") and a[0]["k"] == "vgpr":
            regs.append(a[0]["r"])
        for r in regs:
            if r not in written and r not in first:
                first.append(r)
        if not reads_all and a[0]["k"] == "vgpr":
            written.update(range(a[0]["r"], a[0]["r"] + a[0].get("n", 1)))
    return set(first)


def input_signature_full(path, name):
    """ISG1 elements as (system value, declared mask, read/write mask)."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"ISG1":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            cnt, eo = struct.unpack_from("<II", p, 0)
            return [(struct.unpack_from("<I", p, eo + 32 * i + 12)[0], p[eo + 32 * i + 24], p[eo + 32 * i + 25])
                    for i in range(cnt)]
    return []


def input_signature(path, name):
    """ISG1 elements of the shader's DXBC container: (register, mask, system value, component type)."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"ISG1":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            cnt, eo = struct.unpack_from("<II", p, 0)
            out = []
            for i in range(cnt):
                e = p[eo + 32 * i:eo + 32 * i + 32]
                _, _, _, sysv, ctype, reg = struct.unpack_from("<6I", e, 0)
                out.append((reg, e[24], sysv, ctype))
            return out
    return []


def same_encoding(raw, printed, op):
    """True if the bytes in the blob are the instruction LLVM printed. Exact match, except for fields the
    hardware ignores that LLVM re-encodes with its default: VOP3P op_sel_hi[2] (bit 14) on two-source packed
    ops (seen: v_pk_add_f16 stored with 0, printed with 1), and unused VOP3/VOP3P SRC1 (bits 49:41) and SRC2
    (bits 58:50) fields, which the game's compiler stores as 0 and LLVM 23 prints as 0x80 (inline constant 0;
    seen on two-source v_mul_legacy_f32/v_sub_f32/v_cmp_*_e64/v_mul_lo_u32/v_pk_mul_f16 and one-source
    v_rcp/v_log/v_sqrt_f32_e64). LLVM decoded these bytes, so a source it actually reads would have printed as
    s0, not 0x80."""
    if raw == printed:
        return True
    if len(raw) != len(printed) or len(raw) < 8:
        return False
    r = int.from_bytes(raw, "little")
    diff = r ^ int.from_bytes(printed, "little")
    vop3 = (printed[3] >> 2) == 0x35               # VOP3 (encoding 110101)
    vop3p = printed[3] == 0xCC
    if vop3 or vop3p:
        for shift in (41, 50):                     # SRC1, SRC2: unused field stored 0, printed 0x80
            if (diff >> shift) & 0x1FF == 0x80 and (r >> shift) & 0x1FF == 0:
                diff &= ~(0x1FF << shift)
    two_src = op.startswith("v_pk_") and not any(k in op for k in ("fma", "mad"))
    if vop3p and two_src:
        diff &= ~(1 << 14)
    return diff == 0


def output_signature(path, name):
    """OSG1 elements as (register, system value), one per register (first element wins)."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"OSG1":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            cnt, eo = struct.unpack_from("<II", p, 0)
            seen = {}
            for i in range(cnt):
                e = p[eo + 32 * i:eo + 32 * i + 32]
                _, _, _, sysv, _, reg = struct.unpack_from("<6I", e, 0)
                seen.setdefault(reg, sysv)
            return sorted(seen.items())
    return []


def output_signature_full(path, name):
    """OSG1 elements as (register, semantic_index, system_value, mask, rw_mask)."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    if not os.path.isfile(dxbc):
        return []
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"OSG1":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            cnt, eo = struct.unpack_from("<II", p, 0)
            out = []
            for i in range(cnt):
                e = p[eo + 32 * i:eo + 32 * i + 32]
                _, _, sem_idx, sysv, _, reg = struct.unpack_from("<6I", e, 0)
                out.append((reg, sem_idx, sysv, e[24], e[25]))
            return out
    return []


def pos_export_layout(osg1, insns):
    """Map RDNA/GFX10 pos1..pos3 export components to BuiltIn ClipDistance / CullDistance (and misc).

    Hardware ABI (Mesa src/amd/common/nir/ac_nir_prerast_utils.c: ac_nir_export_position;
    AMD LLPC lgc/lowering/LowerInOut.cpp: LowerInOut::visitReturnInst):
      - pos0 = SV_Position
      - if any misc output (PointSize / EdgeFlag / ShadingRate / Layer / ViewportIndex) is active,
        pos1 holds the misc vector (z[10:0] = Layer, z[19:16] = ViewportIndex on GFX9+) and
        clip/cull distances start at pos2; otherwise clip/cull distances start at pos1.
      - clip distances precede cull distances, packed contiguously into 4-component pos exports (<= 8 total).
    Contrasted with DXBC OSG1 (D3D_NAME: 2=SV_ClipDistance, 3=SV_CullDistance,
    4=SV_RenderTargetArrayIndex, 5=SV_ViewportArrayIndex, 23=SV_ShadingRate).
    When OSG1 is absent (or declares no clip/cull/misc outputs), pos1..pos3 exports are ignored.
    """
    misc_svs = {sv for _, _, sv, _, _ in osg1 if sv in (4, 5, 23)}
    has_misc = bool(misc_svs)
    clip_elems = sorted((sem, reg, mask) for reg, sem, sv, mask, _ in osg1 if sv == 2)
    cull_elems = sorted((sem, reg, mask) for reg, sem, sv, mask, _ in osg1 if sv == 3)
    slots = []
    clip_n = 0
    for _, _, mask in clip_elems:
        for c in range(4):
            if (mask >> c) & 1:
                slots.append(("clip", clip_n))
                clip_n += 1
    cull_n = 0
    for _, _, mask in cull_elems:
        for c in range(4):
            if (mask >> c) & 1:
                slots.append(("cull", cull_n))
                cull_n += 1
    base_pos = 2 if has_misc else 1
    pos_map = {}
    for k, target in enumerate(slots[:8]):
        p = base_pos + k // 4
        pos_map.setdefault(p, {})[k % 4] = target
    clip_n = min(clip_n, 8)
    cull_n = min(cull_n, max(0, 8 - clip_n))
    return pos_map, misc_svs, clip_n, cull_n


def mesh_psv(path, name):
    """PSV0 runtime info of a mesh shader: MSInfo (GroupSharedBytesUsed, GroupSharedViewIDDependentBytes,
    PayloadSizeInBytes, u16 MaxOutputVertices, u16 MaxOutputPrimitives), wave lane counts, then PSVRuntimeInfo1
    (stage, UsesViewID, u8 SigPrimVectors, u8 MeshOutputTopology, ...) and PSVRuntimeInfo2 NumThreadsX/Y/Z."""
    dxbc = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), name + ".dxbc")
    d = open(dxbc, "rb").read()
    for k in range(struct.unpack_from("<I", d, 28)[0]):
        off = struct.unpack_from("<I", d, 32 + 4 * k)[0]
        if d[off:off + 4] == b"PSV0":
            p = d[off + 8:off + 8 + struct.unpack_from("<I", d, off + 4)[0]]
            size = struct.unpack_from("<I", p, 0)[0]
            r = p[4:4 + size]
            if size < 48 or r[24] != 13:
                return None
            maxv, maxp = struct.unpack_from("<HH", r, 12)
            tx, ty, tz = struct.unpack_from("<III", r, 36)
            shared, = struct.unpack_from("<I", r, 0)
            return {"maxv": maxv, "maxp": maxp, "topology": r[27], "threads": tx * ty * tz, "size": (tx, ty, tz),
                    "shared": shared}
    return None


def translate(path):
    info = parse_file(path)
    stage = stage_of(info["name"], path)
    if stage not in ("vs", "ps", "cs", "ms", "gs"):
        raise Unsupported(f"stage {stage}")
    insns = info["insns"]
    if info["invalid"]:
        # Undecodable words are constant tables embedded in the code range (147/155 such VS/PS use s_getpc_b64 to
        # read them). LLVM skips them, so re-anchor every instruction to its true offset in the FSSM blob, then
        # translate only code reachable from entry; a branch into data is rejected below.
        fssm = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "fssm", info["name"] + ".fssm")
        blob = open(fssm, "rb").read()
        pos = info["code_start"]
        for ins in insns:
            enc = bytes.fromhex(ins["enc"])
            skipped, at = 0, pos
            while not same_encoding(blob[at:at + len(enc)], enc, base_op(ins["op"])):
                at += 4
                skipped += 4
                if skipped > 4096 or at >= len(blob):
                    at = None
                    break
            if at is None:
                # LLVM prints the encoding of the instruction it decoded, not the bytes it read: data words that
                # decode with ignored fields set (e.g. 0xbe8d2242 -> s_rfe_b64, printed 0xbe802242) never match.
                # The decoder consumed the same number of bytes, so the instruction sits at the current position.
                # Such words are data in every case seen; reachable_code drops them unless a branch lands there.
                if pos + len(enc) > len(blob):
                    raise Unsupported("undecodable words (cannot re-anchor)")
                at = pos
                ins["normalized"] = True
            ins["pc"] = at
            pos = at + len(enc)
        insns = fold_far_branches(insns)
        insns = reachable_code(insns)
        if any(i.get("normalized") for i in insns):
            raise Unsupported("reachable instruction with normalized encoding")
    g = Gen(stage)
    g.wave64 = any(WAVE64_RE.search(" ".join([i["op"]] + [x.get("t", "") for x in i["args"]])) or
                   (base_op(i["op"]).endswith("_b64") and any(x.get("t") in ("exec", "vcc") for x in i["args"]))
                   for i in insns)
    if stage == "ms":
        g.wave64 = True            # NGG mesh shaders here are wave64 (thread = wave id * 64 + lane)
    if stage == "gs":
        ins_sig = [(reg, sv) for reg, mask, sv, ctype in input_signature(path, info["name"])]
        outs = output_signature(path, info["name"])
        if not ins_sig or not outs:
            raise Unsupported("gs signature")
        n_emit = sum(1 for i in insns if base_op(i["op"]) == "s_sendmsg" and
                     any("GS_OP_EMIT" in x.get("t", "") for x in i["args"]))
        if not (1 <= n_emit <= 64):
            raise Unsupported("gs emit count")
        g.gs = {"in": ins_sig, "out": outs, "nreg": max(r for r, _ in outs) + 1, "maxv": n_emit}
        for i in (0, 1, 3, 8, 64, 128, 0x4000, g.gs["nreg"], 64 << 16):          # used by assemble
            g.cu(i)
        for i in range(n_emit * g.gs["nreg"] + 1):
            g.cu(i)
    if stage == "ms":
        # Mesh shaders run as NGG wave64 programs. PSV0 (DXBC) gives NumThreads, MaxOutputVertices/Primitives and
        # the output topology. System inputs (Mesa radv_shader_args.c, mesh as NGG on GFX10.3): v5 = first launch
        # vertex ID of the lane, groupIndex * NumThreads + lane ("vertex_id" slot); the merged-wave-info SGPR (s3 here) holds the wave id in [27:24]; user data
        # follows in s16.. (from push constants, like every stage).
        psv = mesh_psv(path, info["name"])
        if psv is None:
            raise Unsupported("mesh PSV0")
        if psv["topology"] != 2:
            raise Unsupported(f"mesh topology {psv['topology']}")
        if not (1 <= psv["threads"] <= 1024) or not (1 <= psv["maxv"] <= 256) or not (1 <= psv["maxp"] <= 512):
            raise Unsupported("mesh limits")
        g.ms = psv
        g.cu(psv["threads"])     # the v5 seed (assemble) multiplies by it; constants must exist before assembly
        # LDS: the declared group-shared size, grown to cover every constant offset the code uses.
        need = (psv.get("shared", 0) + 3) // 4
        for ins in insns:
            if not ins["op"].startswith("ds_") or ins["op"].startswith(("ds_swizzle", "ds_bpermute", "ds_permute")):
                continue
            for m in ins["mods"]:
                k, _, v = m.partition(":")
                if not v[:1].isdigit():
                    continue
                if k == "offset":
                    need = max(need, int(v, 0) // 4 + 4)
                elif k in ("offset0", "offset1"):
                    need = max(need, int(v, 0) * (64 if "st64" in ins["op"] else 1) * 2 + 4)
        n = 16
        while n < need:
            n *= 2
        g.lds_dwords = min(n, 16384)
        g.cu(g.lds_dwords)
        g.cu(g.lds_dwords - 1)
        for i in (2, 3, 5, 6, 10, 12, 20, 24, 31, 63, 0x108, 0x1FF, 0xFFF, psv["maxv"], psv["maxp"]):   # used by assemble
            g.cu(i)
    if stage == "cs":
        # Compute system registers come from the FSSM header. Word 0x138 is COMPUTE_PGM_RSRC2 (checked against
        # what 2,689 CS read before writing: 2,281 fit exactly, the rest are loop artifacts of the straight-line
        # check): USER_SGPR [5:1], TGID_X/Y/Z_EN [9:7], TG_SIZE_EN [10], TIDIG_COMP_CNT [12:11]. Words
        # 0x13C/0x140/0x144 are the thread-group size. Layout (Mesa radv_shader_args.c): s0..user-1 = user data,
        # then the enabled workgroup ids; v0, v1, v2 = thread id x, y, z (separate VGPRs on GFX10).
        fssm = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "fssm", info["name"] + ".fssm")
        hdr = open(fssm, "rb").read(0x148)
        rsrc2, lx, ly, lz = struct.unpack_from("<4I", hdr, 0x138)
        if (rsrc2 >> 10) & 1:
            raise Unsupported("CS thread-group-size SGPR")
        if not (1 <= lx * ly * lz <= 1024):
            raise Unsupported("CS group size")
        user = (rsrc2 >> 1) & 31
        g.cs = {"user": user, "tgid": [c for c in range(3) if (rsrc2 >> (7 + c)) & 1],
                "tid": (rsrc2 >> 11) & 3, "size": (lx, ly, lz)}
        # LDS_SIZE [23:15] in 512-byte units (470 of 486 LDS-using CS stay below it; the rest index past it with
        # constant offsets, so the array also covers the largest offset in the code).
        need = ((rsrc2 >> 15) & 0x1FF) * 128
        for ins in insns:
            if not ins["op"].startswith("ds_") or ins["op"].startswith(("ds_swizzle", "ds_bpermute", "ds_permute")):
                continue
            for m in ins["mods"]:
                k, _, v = m.partition(":")
                if not v[:1].isdigit():
                    continue
                if k == "offset":
                    need = max(need, int(v, 0) // 4 + 4)
                elif k in ("offset0", "offset1"):
                    need = max(need, int(v, 0) * (64 if "st64" in ins["op"] else 1) * 2 + 4)
        n = 16
        while n < need:
            n *= 2
        g.lds_dwords = min(n, 16384)
        g.cu(g.lds_dwords)
        g.cu(g.lds_dwords - 1)
        for i in range(user, user + 3):
            g.cu(i)
    if stage == "ps":
        # Pixel-shader system VGPRs (verified on 2,531 of 3,000 PS): v0-v1 = barycentrics (modelled through
        # v_interp -> attributes), then v2.. = the SV_Position components the shader reads, packed in order.
        # Anything else a PS reads before writing is an input layout we haven't confirmed: reject it.
        sig = input_signature_full(path, info["name"])
        pos = [rw for sv, m, rw in sig if sv == 1]
        used = pos[0] if pos else 0
        # Position w: AMD's POS_W_FLOAT input holds 1/w. Mesa (src/amd/common/nir/ac_nir_lower_intrinsics_to_args.c)
        # lowers nir_intrinsic_load_frag_coord_w_rcp straight to frag_pos[3]. That's the same value as Vulkan's
        # FragCoord.w, so all four components map directly.
        # Hardware order (Mesa src/amd/vulkan/radv_shader_args.c): barycentric pairs, ..., POS_X/Y/Z/W, FRONT_FACE,
        # ANCILLARY, SAMPLE_COVERAGE, POS_FIXED_PT. Only enabled inputs are packed. Front face is a float whose
        # sign gives facing (Mesa: load_front_face = front_face > 0; load_front_face_fsign = the raw value).
        layout = {}
        nxt = 2
        for c in range(4):
            if used >> c & 1:
                layout[nxt] = c
                nxt += 1
        if any(sv == 9 for sv, _, _ in sig):           # D3D_NAME_IS_FRONT_FACE
            layout[nxt] = "front"
            nxt += 1
        no_bary = [dict(i, args=[x for k, x in enumerate(i["args"])
                                 if not (i["op"].startswith("v_interp") and x["k"] == "vgpr" and k != 0)])
                   for i in insns]
        reads = read_before_write_vgprs(no_bary)
        # Reads inside the system-input range that aren't in the layout (raw barycentrics, unknown inputs): reject.
        # Reads past the last hardware input can't be hardware inputs: they're values undefined on some path, and
        # the zero-initialised registers stand in for them.
        # Raw reads of the barycentric pair (v0 = i, v1 = j) outside v_interp, e.g. v_readfirstlane_b32 s41, v1 in
        # 315 PS: AMD interpolates P = P0 + i(P1 - P0) + j(P2 - P0), so i and j are the weights of vertices 1 and 2,
        # which VK_KHR_fragment_shader_barycentric gives as BaryCoordKHR.y / .z. Assumes the pair is the
        # perspective-center one, the only pair this layout models.
        for r, comp in ((0, "bary_i"), (1, "bary_j")):
            if r in reads:
                layout[r] = comp
        unexplained = {r for r in reads if r < nxt and r not in layout}
        if unexplained:
            raise Unsupported("PS system inputs")
        g.frag_inputs = layout
        for r in layout:
            g.cu(r)
        # FSSM pixel-input table (tools/fssm_header.py): the authoritative interpolant list. Position i of the
        # table is v_interp attr i; interpolants() gives the count and the 16-bit flag mask by position. Every
        # table position is declared as a fragment input (the table lists inputs the code never reads: 775 of
        # 12,101 PS overcount by one or more), and the mask marks 16-bit interpolants for the v_interp path
        # above. No sidecar, or a sidecar with no table, leaves the code-driven declarations untouched.
        try:
            with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(path))), "fssm",
                                   info["name"] + ".fssm"), "rb") as f:
                count, fp16 = fssm_header.interpolants(f.read())
        except OSError:
            count, fp16 = 0, 0
        if count:
            g.pixel_interpolants = (count, fp16)
            for n in range(count):
                g.inputs.setdefault(f"%in_attr{n}", n)
    if stage in ("vs", "ms"):
        g.pos_map, g.pos_misc, g.clip_count, g.cull_count = pos_export_layout(
            output_signature_full(path, info["name"]), insns)
    if stage == "vs" and any(base_op(i["op"]) == "s_swappc_b64" for i in insns):
        # The call goes to a driver-generated fetch program (not in the blob). Verified rule over 788 shaders:
        # input-signature register r, component c is fetched into v(4 + 4r + c). Replace the call with Vulkan
        # vertex inputs (location = register) written into those VGPRs at entry.
        sig = [s for s in input_signature(path, info["name"]) if s[2] == 0]
        # Any component of a declared register may be read: components the vertex format lacks come back as the
        # format defaults (0, 0, 1 for y, z, w) from the hardware fetch, and Vulkan fills a vec4 input from a
        # narrower format the same way. Seen: .w of an xyz position.
        predicted = {4 + 4 * reg + c for reg, mask, _, _ in sig for c in range(4)}
        reads = {r for r in read_before_write_vgprs(insns) if r >= 4}
        if not reads <= predicted:
            raise Unsupported("fetch mapping")
        g.vertex_inputs = [(reg, ctype) for reg, _, _, ctype in sig]
        g.vertex_signature = [{"register": reg, "mask": mask, "system_value": sv, "component_type": ctype}
                              for reg, mask, sv, ctype in sig]
        for reg, _ in g.vertex_inputs:          # register the VGPR index constants before assembly
            for c in range(4):
                g.cu(4 + 4 * reg + c)
        insns = [i for i in insns if base_op(i["op"]) != "s_swappc_b64"]
    user_limit = g.cs["user"] if g.cs else USER_SGPRS
    sys_sgprs = (3,) if stage in ("ms", "gs") else ()
    if WAVE64_EMULATE and g.wave64 and stage in ("cs", "ms"):
        g.emulate = True
        size = (g.cs or g.ms)["size"]
        g.x_lanes = -(-size[0] * size[1] * size[2] // 64) * 64
    insns = fold_far_branches(insns)
    branchy = any(base_op(i["op"]) in BRANCHES for i in insns)
    if not branchy and not g.emulate:
        g.park_masks(([insns], [[]]))
        g.build_meta = _shader_meta.collect(path, info["name"], [insns], [[]],
                                            user_sgpr_limit=user_limit, system_sgprs=sys_sgprs)
        for ins in insns:
            if base_op(ins["op"]) == "s_endpgm":
                break
            g.insn(ins)
        translate.last_gen = g
        return assemble(g)

    # ---- basic blocks ----
    def next_pc(ins):
        return ins["pc"] + len(ins["enc"]) // 2

    def target(ins):
        if "target" in ins:
            return ins["target"]
        imm = next((x for x in ins["args"] if x["k"] == "imm"), None)
        if imm is None:
            raise Unsupported("branch without offset")
        v = imm["v"] & 0xFFFF
        return next_pc(ins) + 4 * (v - 0x10000 if v & 0x8000 else v)

    leaders = {insns[0]["pc"]}
    for ins in insns:
        if base_op(ins["op"]) in BRANCHES:
            leaders.add(target(ins))
            leaders.add(next_pc(ins))
    pcs = {i["pc"] for i in insns}
    starts = sorted(p for p in leaders if p in pcs)
    if any(p not in starts for p in (target(i) for i in insns if base_op(i["op"]) in BRANCHES)):
        raise Unsupported("branch target outside code")
    index = {p: n for n, p in enumerate(starts)}
    blocks = [[] for _ in starts]
    for n in range(len(blocks) + 1):
        g.cu(n)                                      # block numbers: %pc is compared with any of them
    cur = -1
    for ins in insns:
        if ins["pc"] in index:
            cur = index[ins["pc"]]
        blocks[cur].append(ins)

    # Successors ahead of emission, for node splitting and lane-mask analysis (the loop below works them out
    # again as it goes, and the two are compared at the end).
    raw_edges = []
    for n, blk in enumerate(blocks):
        last = next((i for i in blk if base_op(i["op"]) == "s_endpgm" or base_op(i["op"]) in BRANCHES), None)
        fall = [n + 1] if n + 1 < len(blocks) else []
        if last is None:
            raw_edges.append(fall)
        elif base_op(last["op"]) == "s_endpgm":
            raw_edges.append([])
        elif base_op(last["op"]) == "s_branch":
            raw_edges.append([index[target(last)]])
        else:
            raw_edges.append([index[target(last)]] + fall)
    split_edges = None
    if STRUCTURED:
        res = cfg_structure.split_irreducible(raw_edges)
        if res is not None and len(res[0]) > len(blocks):
            split_edges, orig_of = res
            # Shallow-copy each instruction dict so lane_masks.py's id(ins) keys stay unique across clones.
            blocks = [[dict(ins) for ins in blocks[o]] for o in orig_of]
            for n in range(len(blocks) + 1):
                g.cu(n)
            raw_edges = split_edges
    succ = [list(dict.fromkeys(e)) for e in raw_edges]
    g.park_masks((blocks, succ))
    g.build_meta = _shader_meta.collect(path, info["name"], blocks, succ,
                                        user_sgpr_limit=user_limit, system_sgprs=sys_sgprs)

    g.x_seg = len(blocks) + 1           # emulate: continuation segments are numbered after "done" (len(blocks))
    for n, blk in enumerate(blocks):
        g.body = []
        g.block_label = f"%blk{n}"
        g.x_marks, g.x_dirty = [], set()
        g.x_tok, g.x_mcache = {}, {}     # a register's value is only known from where this block wrote it
        g.x_gen += 1
        g.desc_cache.clear()            # a cached lookup is only good inside the block that made it
        g.sgpr_imm.clear()
        term = None
        for ins in blk:
            op = base_op(ins["op"])
            if op == "s_endpgm":
                term = ("end",)
                break
            if op in BRANCHES:
                term = ("br", op, index[target(ins)])
                break
            g.insn(ins)
        edges = raw_edges[n]
        fall = (edges[0] if edges else None) if term is None else (edges[1] if len(edges) > 1 else None)
        # Every block leaves the number of the block to run next in %pc (len(blocks) = none: the program is over),
        # and its possible successors in g.succ for cfg_structure.
        if term is None or term[0] == "end":
            nxt = fall if term is None else None
            g.emit(f"OpStore %pc {g.cu(nxt if nxt is not None else len(blocks))}")
            if nxt is None:
                g.emit("OpStore %done %true")
            g.succ.append([nxt] if nxt is not None else [])
        else:
            _, op, _ = term
            tgt = edges[0]
            if op == "s_branch":
                g.emit(f"OpStore %pc {g.cu(tgt)}")
                g.succ.append([tgt])
            else:
                marks0, body0 = len(g.x_marks), len(g.body)
                cond = branch_condition(g, op)
                alt = g.cu(fall) if fall is not None else g.cu(len(blocks))   # past the end = default -> done
                r = g.tmp()
                g.emit(f"{r} = OpSelect %uint {cond} {g.cu(tgt)} {alt}")
                g.emit(f"OpStore %pc {r}")
                if g.emulate and op in ("s_cbranch_vccz", "s_cbranch_vccnz", "s_cbranch_execz", "s_cbranch_execnz"):
                    # This segment only reads the vote and sets %pc, and the code before the vote ran in an earlier
                    # round: the successor can start in this one (x_scheduler). That only holds when the vote's own
                    # x_split started this segment: then it holds the vote read and the %pc select and nothing else.
                    # A vote that reused an earlier exchange (no split) would put the end of the block in the
                    # successor's round, with no barrier between their memory accesses.
                    if len(g.x_marks) != marks0 + 1 or g.x_marks[-1][0] < body0:
                        raise Unsupported("wave64 emulation: a voted branch must start its own segment")
                    g.emit("OpStore %x_chain %true")
                g.succ.append(list(dict.fromkeys([tgt] + ([fall] if fall is not None else []))))
        lines = [l.strip() for l in g.body]
        if g.x_marks:
            cuts = [0] + [i for i, _ in g.x_marks] + [len(lines)]
            parts = x_spill(g, [lines[a:b] for a, b in zip(cuts, cuts[1:])])
            g.blocks.append(parts[0])
            for (_, seg), part in zip(g.x_marks, parts[1:]):
                g.x_segs[seg] = part
        else:
            g.blocks.append(lines)
    assert succ == g.succ, "the lane-mask analysis ran on a different graph"
    g.body = []
    translate.last_gen = g
    return assemble(g)


_DEF = re.compile(r"(%\w+) = (Op\w+)(?: (%\w+))?")


def x_spill(g, parts):
    """Emulated wave64: the segments of one block run in different rounds, i.e. in different cases of the
    scheduler's switch and different loop iterations, so an SSA value defined in one and used in a later one
    can't be referenced directly. It goes through a Function variable: stored at the end of the segment that
    defines it and loaded, under a new id, at the top of each later segment that uses it."""
    defs, out = {}, []
    for k, part in enumerate(parts):
        used = []
        for line in part:
            for ref in _ID_REF.findall(line):
                if ref in defs and ref not in used:
                    used.append(ref)
        loads, rename = [], {}
        for ref in used:
            seg, typ, var = defs[ref]
            if var is None:
                if typ is None or typ.startswith("%_ptr"):
                    raise Unsupported("wave64 emulation: a pointer lives across a round")
                ptr = {"%uint": "%_ptr_Function_uint", "%bool": "%_ptr_Function_bool"}.get(typ)
                if ptr is None:
                    ptr = g.x_ptr_types.setdefault(typ, f"%_ptr_Function_x_{typ[1:]}")
                var = f"%xs_{ref[1:]}"
                g.x_spills.append((var, ptr))
                out[seg].append(f"OpStore {var} {ref}")
                defs[ref] = (seg, typ, var)
            new = f"{ref}_r{k}"
            loads.append(f"{new} = OpLoad {typ} {var}")
            rename[ref] = new
        body = [_ID_REF.sub(lambda t: rename.get(t.group(0), t.group(0)), l) for l in part] if rename else list(part)
        out.append(loads + body)
        for line in part:
            m = _DEF.match(line)
            if m and m.group(2) != "OpLabel":
                defs[m.group(1)] = (k, m.group(3), None)
    return out


def x_scheduler(gen, threads):
    """Emulated wave64: the head of the round loop (see Gen.x_round), up to the switch over %pc. The first two
    lines are the Function variables %pc and %done, as for the dispatch loop.

    Every invocation runs every round, so the loop and the barrier at its end are in uniform control flow. In a
    round an invocation runs one segment unless it is done or waits at s_barrier. After the segment it adds
    itself to the round's counts (not done; neither done nor waiting); after the barrier everyone reads the same
    counts: no lane left ends the loop, and no lane left that isn't waiting releases the s_barrier."""
    c = lambda v: gen.consts[v & 0xFFFFFFFF]                                          # noqa: E731
    x_waves, x_mask, x_sched, x_sink = gen.x_sizes()
    sx, sy, _ = (gen.cs or gen.ms)["size"]
    L = ["%pc = OpVariable %_ptr_Function_uint Function %c0",
         "%done = OpVariable %_ptr_Function_bool Function %false"]
    # This invocation's place: index in the workgroup, lane and wave (the same split as Gen.wave_lane).
    if gen.stage == "ms":
        L += ["%x_li = OpLoad %uint %local_index"]
    else:
        L += ["%x_lid = OpLoad %v3uint %local_id", "%x_lx = OpCompositeExtract %uint %x_lid 0",
              "%x_ly = OpCompositeExtract %uint %x_lid 1", "%x_lz = OpCompositeExtract %uint %x_lid 2",
              f"%x_ly2 = OpIMul %uint %x_ly {c(sx)}", f"%x_lz2 = OpIMul %uint %x_lz {c(sx * sy)}",
              "%x_lxy = OpIAdd %uint %x_lx %x_ly2", "%x_li = OpIAdd %uint %x_lxy %x_lz2"]
    L += [f"%x_lane = OpBitwiseAnd %uint %x_li {c(63)}", f"%x_wave = OpShiftRightLogical %uint %x_li {c(6)}",
          f"%x_wbase = OpBitwiseAnd %uint %x_li {c(0xFFFFFFC0)}"]
    # Workgroup memory starts undefined: zero the mask and schedule words, every invocation a share of them.
    zero = x_sink - x_mask
    for k in range(-(-zero // threads)):
        L += [f"%x_z{k} = OpIAdd %uint %x_li {c(k * threads)}",
              f"%x_zm{k} = OpExtInst %uint %glsl UMin %x_z{k} {c(zero - 1)}",
              f"%x_zi{k} = OpIAdd %uint %x_zm{k} {c(x_mask)}",
              f"%x_zp{k} = OpAccessChain %_ptr_WG_uint_x %x_buf %x_zi{k}", f"OpStore %x_zp{k} {c(0)}"]
    # A partial last wave: its missing lanes never write their data slots. They read as 0, as a lane that is
    # off does for ds_bpermute (and v_readlane of a lane that doesn't exist has no defined value anyway).
    missing = gen.x_lanes - threads
    for k in range(-(-missing // threads) if missing else 0):
        for par, base in ((0, threads), (1, gen.x_lanes + threads)):
            L += [f"%x_y{k}_{par} = OpIAdd %uint %x_li {c(k * threads)}",
                  f"%x_ym{k}_{par} = OpExtInst %uint %glsl UMin %x_y{k}_{par} {c(missing - 1)}",
                  f"%x_yi{k}_{par} = OpIAdd %uint %x_ym{k}_{par} {c(base)}",
                  f"%x_yp{k}_{par} = OpAccessChain %_ptr_WG_uint_x %x_buf %x_yi{k}_{par}",
                  f"OpStore %x_yp{k}_{par} {c(0)}"]
    L += [f"OpControlBarrier {c(2)} {c(2)} {c(0x948)}",
          "OpBranch %loop_head", "%loop_head = OpLabel", "OpLoopMerge %loop_exit %loop_cont None",
          "OpBranch %x_round", "%x_round = OpLabel",
          "%x_itv = OpLoad %uint %x_it", f"%x_s = OpUMod %uint %x_itv {c(3)}",
          # Free the slot of round + 1 for its writers. Its readers ran before the last barrier (round - 2 wrote
          # it, round - 1 read it); its writers come after the next one. Lane 0 of each wave clears the wave's
          # mask words, invocation 0 the counts; the others store to the sink word.
          f"%x_n1 = OpIAdd %uint %x_itv {c(1)}", f"%x_sn = OpUMod %uint %x_n1 {c(3)}",
          f"%x_m0 = OpIMul %uint %x_sn {c(x_waves)}", "%x_m1 = OpIAdd %uint %x_m0 %x_wave",
          f"%x_m2 = OpIMul %uint %x_m1 {c(2)}", f"%x_m3 = OpIAdd %uint %x_m2 {c(x_mask)}",
          f"%x_m4 = OpIAdd %uint %x_m3 {c(1)}", f"%x_ml = OpIEqual %bool %x_lane {c(0)}",
          f"%x_m5 = OpSelect %uint %x_ml %x_m3 {c(x_sink)}", f"%x_m6 = OpSelect %uint %x_ml %x_m4 {c(x_sink)}",
          "%x_mp5 = OpAccessChain %_ptr_WG_uint_x %x_buf %x_m5", f"OpStore %x_mp5 {c(0)}",
          "%x_mp6 = OpAccessChain %_ptr_WG_uint_x %x_buf %x_m6", f"OpStore %x_mp6 {c(0)}",
          f"%x_q0 = OpIMul %uint %x_sn {c(2)}", f"%x_q1 = OpIAdd %uint %x_q0 {c(x_sched)}",
          f"%x_q2 = OpIAdd %uint %x_q1 {c(1)}", f"%x_ql = OpIEqual %bool %x_li {c(0)}",
          f"%x_q3 = OpSelect %uint %x_ql %x_q1 {c(x_sink)}", f"%x_q4 = OpSelect %uint %x_ql %x_q2 {c(x_sink)}",
          "%x_qp3 = OpAccessChain %_ptr_WG_uint_x %x_buf %x_q3", f"OpStore %x_qp3 {c(0)}",
          "%x_qp4 = OpAccessChain %_ptr_WG_uint_x %x_buf %x_q4", f"OpStore %x_qp4 {c(0)}",
          "%x_d = OpLoad %bool %done", "%x_b = OpLoad %bool %x_bar", "%x_nd = OpLogicalNot %bool %x_d",
          "%x_nb = OpLogicalNot %bool %x_b", "%x_run = OpLogicalAnd %bool %x_nd %x_nb",
          "OpSelectionMerge %x_ran None", "OpBranchConditional %x_run %dispatch %x_ran",
          # A segment that only reads a vote and picks the next block sets %x_chain: that block runs in the same
          # round (its code before the vote was in an earlier one, and it starts a fresh segment).
          "%dispatch = OpLabel", "OpBranch %x_ch_head", "%x_ch_head = OpLabel",
          "OpLoopMerge %x_ch_merge %x_ch_cont None", "OpBranch %x_ch_body", "%x_ch_body = OpLabel",
          "%pcv = OpLoad %uint %pc", "OpSelectionMerge %sw_merge None",
          "OpSwitch %pcv %sw_default " + " ".join(f"{i} %blk{i}" for i in
                                                  sorted(list(range(len(gen.blocks))) + list(gen.x_segs)))]
    return L


def x_scheduler_tail(gen):
    """Emulated wave64: the end of a round, after the switch (see x_scheduler)."""
    c = lambda v: gen.consts[v & 0xFFFFFFFF]                                          # noqa: E731
    _, _, x_sched, _ = gen.x_sizes()
    return ["%sw_default = OpLabel", "OpStore %done %true", "OpBranch %sw_merge",
            "%sw_merge = OpLabel", "OpBranch %x_ch_cont", "%x_ch_cont = OpLabel",
            "%x_chv = OpLoad %bool %x_chain", "OpStore %x_chain %false",
            "OpBranchConditional %x_chv %x_ch_head %x_ch_merge", "%x_ch_merge = OpLabel",
            "OpBranch %x_ran", "%x_ran = OpLabel",
            "%x_d2 = OpLoad %bool %done", "%x_b2 = OpLoad %bool %x_bar",
            "%x_nd2 = OpLogicalNot %bool %x_d2", "%x_nb2 = OpLogicalNot %bool %x_b2",
            "%x_w = OpLogicalAnd %bool %x_nd2 %x_nb2",
            f"%x_c0 = OpSelect %uint %x_nd2 {c(1)} {c(0)}", f"%x_c1 = OpSelect %uint %x_w {c(1)} {c(0)}",
            f"%x_s0 = OpIMul %uint %x_s {c(2)}", f"%x_s1 = OpIAdd %uint %x_s0 {c(x_sched)}",
            f"%x_s2 = OpIAdd %uint %x_s1 {c(1)}",
            "%x_p0 = OpAccessChain %_ptr_WG_uint_x %x_buf %x_s1",
            "%x_p1 = OpAccessChain %_ptr_WG_uint_x %x_buf %x_s2",
            f"%x_a0 = OpAtomicIAdd %uint %x_p0 {c(2)} {c(0)} %x_c0",
            f"%x_a1 = OpAtomicIAdd %uint %x_p1 {c(2)} {c(0)} %x_c1",
            "OpBranch %loop_cont", "%loop_cont = OpLabel",
            # Workgroup, acquire-release, workgroup + uniform + image memory: also orders the stores a round made.
            f"OpControlBarrier {c(2)} {c(2)} {c(0x948)}",
            "%x_r0 = OpLoad %uint %x_p0", "%x_r1 = OpLoad %uint %x_p1",
            f"%x_rel = OpIEqual %bool %x_r1 {c(0)}", "%x_b3 = OpLoad %bool %x_bar",
            "%x_nrel = OpLogicalNot %bool %x_rel", "%x_b4 = OpLogicalAnd %bool %x_b3 %x_nrel",
            "OpStore %x_bar %x_b4", f"%x_it1 = OpIAdd %uint %x_itv {c(1)}", "OpStore %x_it %x_it1",
            f"%x_fin = OpIEqual %bool %x_r0 {c(0)}", "OpBranchConditional %x_fin %loop_exit %loop_head",
            "%loop_exit = OpLabel"]


BRANCHES = {"s_branch", "s_cbranch_scc0", "s_cbranch_scc1", "s_cbranch_vccz", "s_cbranch_vccnz",
            "s_cbranch_execz", "s_cbranch_execnz"}


def branch_condition(g, op):
    """Boolean SPIR-V id: whether the branch is taken for this lane."""
    if op in ("s_cbranch_scc0", "s_cbranch_scc1"):
        s = g.tmp()
        g.emit(f"{s} = OpLoad %uint %scc")
        b = g.tmp()
        g.emit(f"{b} = {'OpIEqual' if op.endswith('0') else 'OpINotEqual'} %bool {s} {g.cu(0)}")
        return b
    # VCC and EXEC are wave-wide registers: the hardware branch is taken (or not) by the whole wave. With one
    # invocation per lane, each lane holds only its own bit, so the condition is reduced across the subgroup
    # (= the wave). Deciding per lane would let lanes skip scalar instructions the wave executes, leaving their
    # SGPRs stale, and would break cross-lane ops and barriers inside the block.
    if op in ("s_cbranch_vccz", "s_cbranch_vccnz"):
        s = g.tmp()
        g.emit(f"{s} = OpLoad %uint %vcc")
        lane = g.tmp()
        g.emit(f"{lane} = OpINotEqual %bool {s} {g.cu(0)}")
    else:
        lane = g.active_b()
    any_ = g.tmp()
    g.g_vote(any_, "Any", lane)
    g.uses_vote = True
    if op in ("s_cbranch_vccnz", "s_cbranch_execnz"):
        return any_
    b = g.tmp()
    g.emit(f"{b} = OpLogicalNot %bool {any_}")
    return b


def _sha256_file(path):
    """Hex SHA-256 of a file, or None when the path is missing (fssm sidecars are optional)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def shader_meta(g, asm, src_path=None, fssm_path=None):
    """What a runtime needs to create the pipeline for this module without inspecting it: stage, wave size,
    Vulkan features/extensions it requires, descriptor bindings (set 0), push-constant size, workgroup size,
    LDS bytes, mesh/geometry limits, the input/output locations, plus identity metadata (src/fssm hashes and
    the full ISG1 vertex-input signature) for PSO lookup and corpus auditing."""
    caps = sorted(set(re.findall(r"OpCapability (\w+)", asm)))
    exts = sorted(set(re.findall(r'OpExtension "([^"]+)"', asm)))
    binds = sorted({int(b) for b in re.findall(r"OpDecorate %\w+ Binding (\d+)", asm)})
    feats = {"FragmentBarycentricKHR": "fragmentShaderBarycentric",
             "FragmentShaderPixelInterlockEXT": "fragmentShaderPixelInterlock",
             "MeshShadingEXT": "meshShader", "Geometry": "geometryShader", "Float64": "shaderFloat64",
             "Int64": "shaderInt64", "StorageImageWriteWithoutFormat": "shaderStorageImageWriteWithoutFormat",
             "StorageImageReadWithoutFormat": "shaderStorageImageReadWithoutFormat",
             "PhysicalStorageBufferAddresses": "bufferDeviceAddress",
             "RuntimeDescriptorArray": "runtimeDescriptorArray", "ImageGatherExtended": "shaderImageGatherExtended",
             "MinLod": "shaderResourceMinLod", "Image1D": "shaderImage1D? (core)", "Sampled1D": "core",
             "ClipDistance": "shaderClipDistance", "CullDistance": "shaderCullDistance",
             "ShaderLayer": "shaderOutputLayer", "ShaderViewportIndex": "shaderOutputViewportIndex"}
    m = {"stage": g.stage, "wave64": bool(g.wave64), "control_flow": getattr(g, "control_flow", "straight line"),
         "capabilities": caps, "extensions": exts,
         "features": sorted({feats[c] for c in caps if c in feats and not feats[c].startswith(("core", "shaderImage1D"))}),
         "bindings": binds, "push_constant_bytes": 4 * USER_SGPRS + 16,
         "subgroup_size": 64 if g.wave64 else 32,
         # 64-wide subgroups can only be requested for compute/mesh on RADV and NVIDIA; VS/PS/GS wave64
         # shaders must be treated as emulated (per-32-lane halves) until cross-half ops go through LDS.
         "wave64_emulated": bool(g.wave64 and (g.stage in ("vs", "ps", "gs") or g.emulate))}
    if g.emulate:
        # Exact on any subgroup size: no subgroup operation, cross-lane data in %x_buf (workgroup memory).
        m["wave64_exchange"] = "workgroup"
        m["subgroup_size"] = None
    if "Offset " in asm and re.search(r"OpImageSample\w+ [^\n]*Offset", asm):
        m["needs_maintenance8"] = True
    if g.uses_codebase:
        m["uses_code_base"] = True
    if g.cs:
        m["local_size"] = list(g.cs["size"])
        m["user_sgprs"] = g.cs["user"]
    if g.ms:
        m["local_size"] = list(g.ms["size"])
        m["max_vertices"], m["max_primitives"] = g.ms["maxv"], g.ms["maxp"]
        m["params"] = sorted(g.ms_params)
    if g.gs:
        m["max_vertices"] = g.gs["maxv"]
        m["gs_inputs"] = [r for r, _ in g.gs["in"]]
        m["gs_outputs"] = [r for r, _ in g.gs["out"]]
        m["gs_input_locations"] = sorted(r for r, sv in g.gs["in"] if sv != 1)
        m["gs_output_locations"] = sorted(r for r, sv in g.gs["out"] if sv != 1)
    if g.uses_lds:
        m["lds_bytes"] = 4 * g.lds_dwords
    if g.emulate:
        m["lds_bytes"] = m.get("lds_bytes", 0) + 4 * (g.x_sizes()[3] + 1)     # + %x_buf
    if g.vertex_inputs:
        m["vertex_inputs"] = [{"location": r, "type": {1: "uint", 2: "sint"}.get(t, "float")} for r, t in g.vertex_inputs]
    if getattr(g, "vertex_signature", None):
        m["vertex_input_signature"] = list(g.vertex_signature)
    if getattr(g, "dynamic_format", False):
        # A runtime V# FORMAT means codes outside FORMATS decode to 0; flag for atlas coverage.
        m["dynamic_format"] = True
    if getattr(g, "fixed_format_codes", None):
        m["fixed_format_codes"] = sorted(g.fixed_format_codes)
    if src_path:
        src_hash = _sha256_file(src_path)
        if src_hash:
            m["src_sha256"] = src_hash
    if fssm_path:
        fssm_hash = _sha256_file(fssm_path)
        if fssm_hash:
            m["fssm_sha256"] = fssm_hash
    if g.frag_inputs:
        m["ps_system_inputs"] = sorted({str(c) for c in g.frag_inputs.values()})
    if getattr(g, "pixel_interpolants", None):
        # FSSM pixel-input table the PS input path used: interpolant count and 16-bit positions by table order.
        count, fp16 = g.pixel_interpolants
        m["ps_interpolants"] = count
        m["ps_fp16_interpolants"] = [n for n in range(count) if (fp16 >> n) & 1]
    outs = sorted(int("".join(ch for ch in t if ch.isdigit()) or 0) for t in g.outputs.values() if not t.startswith("pos"))
    if outs:
        m["outputs"] = outs
    if g.writes_depth:
        m["writes_depth"] = True
    if g.inputs:
        m["inputs"] = sorted(g.inputs.values())
    if g.build_meta:
        m.update(g.build_meta)
    return m


def write_index(path, index):
    """Write an index under another name and rename it into place, so a run stopped part-way never leaves half of
    one: make_game.py takes these files as proof that a shard, or the whole translation, finished."""
    tmp = path + ".tmp"     # must not end in .json: --merge globs index.part*.json
    with open(tmp, "w", encoding="utf-8") as f:
        _json.dump(index, f, indent=0, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())    # on disk before the rename, or a power cut could leave the name with no data
    os.replace(tmp, path)   # ponytail: no retry; add one if a scanner on Windows ever holds the new file open


def assemble_spirv(sa, sv):
    """Keep the assembled module out of the final path until validation succeeds."""
    with tempfile.NamedTemporaryFile(prefix=".spirv-", suffix=".tmp", dir=os.path.dirname(sv), delete=False) as f:
        tmp = Path(f.name)
    try:
        result = subprocess.run(["spirv-as", sa, "-o", str(tmp), "--target-env", "vulkan1.2"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            # Offset on non-gather: VK_KHR_maintenance8
            result = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand",
                                     str(tmp)], capture_output=True, text=True)
        if result.returncode == 0:
            tmp.replace(sv)
        else:
            Path(sv).unlink(missing_ok=True)
        return result
    finally:
        tmp.unlink(missing_ok=True)


def _limit_arg(text):
    """--limit N: at least 1. 0 used to mean "no limit" (the whole corpus) and a negative N dropped the last N."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _shard_arg(text):
    """--shard i/n with n >= 1 and 0 <= i < n. n = 0 was a ValueError (slice step 0) and i >= n translated nothing."""
    try:
        si, sn = (int(x) for x in text.split("/"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected i/n, for example 0/8, not {text!r}") from None
    if sn < 1 or not 0 <= si < sn:
        raise argparse.ArgumentTypeError(f"{text!r}: need n >= 1 and 0 <= i < n")
    return si, sn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=_limit_arg)
    ap.add_argument("--shard", type=_shard_arg,
                    help="i/n: translate every n-th shader starting at i (run n processes in parallel, "
                         "then merge the index.part*.json files with --merge)")
    ap.add_argument("--no-asm", action="store_true",
                    help="delete each .spvasm once its .spv is written (the text is ~2.7x the binary: 35 GB for the corpus)")
    ap.add_argument("--merge", action="store_true", help="merge index.part*.json in --out into index.json and exit")
    ap.add_argument("--fssm-dir", default=None,
                    help="optional dir with <name>.fssm sidecars: records fssm_sha256 per shader in the index")
    ap.add_argument("--dispatch-loop", action="store_true",
                    help="emit control flow the old way, as one loop around a switch (to compare with the default)")
    ap.add_argument("--wave64-emulate", action="store_true",
                    help="wave64 compute and mesh shaders exact on any subgroup size (32-wide NVIDIA, Intel): "
                         "cross-lane operations go through workgroup memory instead of subgroup operations")
    ap.add_argument("--graphics-replay-glsl", action="store_true",
                    help="emit a single experimental VS/PS register-replay test shader as GLSL (--out FILE)")
    a = ap.parse_args()
    if a.graphics_replay_glsl:
        import graphics_register_replay
        if not a.src.endswith(".s") or any((a.merge, a.shard, a.limit, a.no_asm, a.dispatch_loop, a.wave64_emulate, a.fssm_dir)):
            ap.error("graphics replay expects one disassembly and --out FILE")
        info = parse_file(a.src)
        try:
            source = graphics_register_replay.generate(info, stage_of(info["name"], a.src), Unsupported)
        except Unsupported as error:
            ap.error(str(error))
        Path(a.out).write_text(source, encoding="ascii")
        return 0
    global STRUCTURED, WAVE64_EMULATE
    STRUCTURED = not a.dispatch_loop
    WAVE64_EMULATE = a.wave64_emulate
    if a.merge:
        import glob as _gl, json as _js
        merged = {}
        for part in sorted(_gl.glob(os.path.join(a.out, "index.part*.json"))):
            merged.update(_js.load(open(part, encoding="utf-8")))
        # The stubs go first: index.json is what tells make_game.py the translation finished.
        import spirv_stubs
        spirv_stubs.generate_for_index(a.out, merged)
        write_index(os.path.join(a.out, "index.json"), merged)
        print(f"merged {len(merged)} entries")
        return 0
    files = [a.src] if a.src.endswith(".s") else sorted(os.path.join(a.src, f) for f in os.listdir(a.src) if f.endswith(".s"))
    files = files[: a.limit] if a.limit is not None else files
    shard = None
    if a.shard:
        si, sn = a.shard
        files = files[si::sn]
        shard = si
    os.makedirs(a.out, exist_ok=True)
    why, ok, valid = Counter(), 0, 0
    meta = {}
    invalid = set()
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            asm = translate(p)
        except Unsupported as e:
            why[str(e)] += 1
            continue
        except (IndexError, KeyError, ValueError) as e:
            why[f"translator error {type(e).__name__}"] += 1
            continue
        ok += 1
        sa = os.path.join(a.out, name + ".spvasm")
        sv = os.path.join(a.out, name + ".spv")
        with open(sa, "w") as f:
            f.write(asm)
        r = assemble_spirv(sa, sv)
        if r.returncode == 0:
            valid += 1
            try:
                fssm_sidecar = os.path.join(a.fssm_dir, name + ".fssm") if a.fssm_dir else None
                meta[name] = shader_meta(translate.last_gen, asm, src_path=p, fssm_path=fssm_sidecar)
            except Exception as e:  # noqa: BLE001 - metadata must never cost a translation
                meta[name] = {"error": f"{type(e).__name__}: {e}"}
        else:
            invalid.add(name)
            why["INVALID: " + (r.stderr.strip().splitlines() or ["?"])[0][:90]] += 1
        if a.no_asm and os.path.exists(sa):
            os.remove(sa)
    import json as _js
    idx_path = os.path.join(a.out, f"index.part{shard}.json" if shard is not None else "index.json")
    old = {}
    if shard is None and os.path.isfile(idx_path) and a.limit is None and len(files) < 1000:
        with open(idx_path, encoding="utf-8") as f:
            old = _js.load(f)  # partial runs merge into an existing index
        for name in invalid:
            old.pop(name, None)
    old.update(meta)
    write_index(idx_path, old)
    if shard is None:
        import spirv_stubs
        spirv_stubs.generate_for_index(a.out, old)
    print(f"{len(files)} shaders: {ok} translated, {valid} pass spirv-val")
    for k, v in why.most_common(25):
        print(f"  {v:6}  {k}")


if __name__ == "__main__":
    sys.exit(main())
