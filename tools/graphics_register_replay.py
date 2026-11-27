"""Experimental straight-line wave64 graphics backend for the three-draw replay test ABI."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Op:
    dest: str
    args: tuple
    symbol: str


def generate(info, stage, unsupported):
    def fail(reason):
        raise unsupported("graphics replay: " + reason)

    def operand(a):
        if any(a.get(flag) for flag in ("neg", "abs", "sext")):
            fail("modified operand")
        if a["k"] in ("sgpr", "vgpr") and a["n"] == 1:
            return f"{a['k'][0]}{a['r']}"
        if a["k"] == "imm":
            return f"{a['v'] & 0xffffffff}u"
        fail("expected a register or integer")

    if info["invalid"] or stage not in ("vs", "ps"):
        fail("expected valid VS/PS disassembly")
    insns = info["insns"]
    if len(insns) < 7 or insns[0]["op"] != "s_mov_b64" or [a.get("t", a.get("v")) for a in insns[0]["args"]] != ["exec", -1]:
        fail("expected full EXEC at entry")
    if insns[-1]["op"] != "s_endpgm" or insns[-2]["op"] != "exp":
        fail("expected a terminal export")
    export = insns[-2]["args"]
    if len(export) != 5 or export[0].get("t") != ("param1" if stage == "vs" else "mrt0"):
        fail("unsupported export")
    outputs = [operand(a) for a in export[1:]]
    if any(not r.startswith("v") for r in outputs) or set(insns[-2]["mods"]) - {"done", "vm"}:
        fail("unsupported export operands or modifiers")
    ops = {"v_add_nc_u32": "+", "v_xor_b32": "^", "v_and_b32": "&",
           "v_mov_b32": "", "s_mov_b32": ""}
    segments = [[]]
    ballots = []
    defined = {"v0", "s0"}  # Synthetic input ABI: stable invocation key and scalar seed.
    regs = set(defined)
    i = 1
    while i < len(insns)-2:
        ins = insns[i]
        op = ins["op"].removesuffix("_e32").removesuffix("_e64")
        a = ins["args"]
        if ins["mods"]:
            fail("instruction modifiers")
        if op == "v_cmp_ne_u32":
            count = insns[i+1]
            if (len(a) != 3 or a[0]["k"] != "sgpr" or a[0]["n"] != 2 or
                count["op"] != "s_bcnt1_i32_b64" or count["mods"] or len(count["args"]) != 2 or
                count["args"][1] != a[0] or count["args"][0]["k"] != "sgpr"):
                fail("expected immediate popcount of the comparison mask")
            rhs = tuple(operand(x) for x in a[1:])
            for r in range(a[0]["r"], a[0]["r"]+2):
                defined.discard(f"s{r}")  # Subsequent direct mask-word reads are outside this backend.
            dest = operand(count["args"][0])
            ballots.append(Op(dest, rhs, "!="))
            segments.append([])
            i += 2
        elif op in ops:
            if len(a) != (3 if ops[op] else 2) or a[0]["k"] != ("vgpr" if op.startswith("v") else "sgpr"):
                fail("invalid ALU operands")
            dest = operand(a[0])
            rhs = tuple(operand(x) for x in a[1:])
            if op.startswith("s") and any(r.startswith("v") for r in rhs):
                fail("vector source for scalar instruction")
            segments[-1].append(Op(dest, rhs, ops[op]))
            i += 1
        else:
            fail(f"unsupported instruction {op}")
        if any(r[0] in "sv" and r not in defined for r in rhs):
            fail("unsupported live-in or direct mask use")
        defined.add(dest)
        regs.add(dest)
    if len(ballots) != 2 or not set(outputs) <= defined:
        fail("expected two ballot boundaries and defined outputs")

    # A vote destination becomes available on resume. Only earlier definitions
    # still needed beyond that point require scratch storage.
    live = set(outputs)
    cuts = [None, None]
    for phase in (2, 1, 0):
        for op in reversed(segments[phase]):
            live.discard(op.dest)
            live.update(r for r in op.args if r[0] in "sv")
        if phase:
            vote = ballots[phase-1]
            live.discard(vote.dest)
            cuts[phase-1] = sorted(live)
            live.update(r for r in vote.args if r[0] in "sv")

    body = ["uint " + ", ".join(sorted(regs)) + ";"]
    for phase, segment in enumerate(segments):
        body.append(f"if (pc.phase == {phase}u) {{")
        if phase:
            for n, reg in enumerate(cuts[phase-1], 6):
                body.append(f"    {reg} = pc.s.data[base + {n}u];")
            table = f"pc.s.votes[{phase-1}]"
            body.append(f"    {ballots[phase-1].dest} = uint(bitCount({table}[pair]) + bitCount({table}[pair+1u]));")
        else:
            body += ["    v0 = key;", "    s0 = pc.seed;"]
        for op in segment:
            body.append(f"    {op.dest} = {(' ' + op.symbol + ' ').join(op.args)};")
        if phase < 2:
            for n, reg in enumerate(cuts[phase], 6):
                body.append(f"    pc.s.data[base + {n}u] = {reg};")
            a, b = ballots[phase].args
            body.append(f"    if ({a} != {b}) atomicOr(pc.s.votes[{phase}][halfID], 1u << (slot & 31u));")
        else:
            for n, reg in enumerate(outputs, 2):
                body.append(f"    pc.s.data[base + {n}u] = {reg};")
            body.append(f"    result = uintBitsToFloat(uvec4({', '.join(outputs)}));")
        body.append("}")
    template = Path(__file__).with_suffix(".glsl").read_text(encoding="ascii")
    if stage == "vs":
        template = template.replace("#version 460", "#version 460\n#define VERTEX_STAGE")
    return template.replace("STRIDE", f"{6 + max(map(len,cuts))}u").replace("// GENERATED_BODY", "\n    ".join(body))
