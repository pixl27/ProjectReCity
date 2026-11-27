"""Experimental wave64 graphics backend for the three-draw replay test ABI."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Op:
    dest: str
    args: tuple
    symbol: str


@dataclass
class Conditional:
    lhs: str
    symbol: str
    rhs: str
    fallthrough: list
    taken: list


def live_before(block, live_out):
    live = set(live_out)
    for op in reversed(block):
        if isinstance(op, Conditional):
            live = live_before(op.fallthrough, live) | live_before(op.taken, live)
            args = (op.lhs, op.rhs)
        else:
            live.discard(op.dest)
            args = op.args
        live.update(r for r in args if r[0] in "sv")
    return live


def emit_block(body, block, indent):
    for op in block:
        if isinstance(op, Conditional):
            body.append(f"{indent}if ({op.lhs} {op.symbol} {op.rhs}) {{")
            emit_block(body, op.fallthrough, indent + "    ")
            if op.taken:
                body.append(indent + "} else {")
                emit_block(body, op.taken, indent + "    ")
            body.append(indent + "}")
        else:
            expr = (" " + op.symbol + " ").join(op.args)
            body.append(f"{indent}{op.dest} = {expr};")


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
    indices = {ins["pc"]: idx for idx, ins in enumerate(insns)}

    def forward_target(idx):
        jump = insns[idx]
        args = jump["args"]
        if len(args) != 1 or args[0]["k"] != "imm" or jump["mods"] or len(jump["enc"] or "") != 8:
            fail("expected an encoded SOPP branch displacement")
        offset = args[0]["v"] & 0xffff
        if offset >= 0x8000:
            fail("backward branch")
        target = indices.get(jump["pc"] + 4 + 4 * offset)
        if target is None or target >= len(insns)-1:
            fail("invalid forward branch target")
        return target

    def check_reads(rhs, incoming):
        if any(r[0] in "sv" and r not in incoming for r in rhs):
            fail("unsupported live-in or register undefined on a branch")

    def alu(ins, incoming):
        op = ins["op"].removesuffix("_e32").removesuffix("_e64")
        a = ins["args"]
        if op not in ops or ins["mods"]:
            fail(f"unsupported ALU instruction {op}")
        if len(a) != (3 if ops[op] else 2) or a[0]["k"] != ("vgpr" if op.startswith("v") else "sgpr"):
            fail("invalid ALU operands")
        dest = operand(a[0])
        rhs = tuple(operand(x) for x in a[1:])
        if op.startswith("s") and any(r.startswith("v") for r in rhs):
            fail("vector source for scalar instruction")
        check_reads(rhs, incoming)
        incoming.add(dest)
        regs.add(dest)
        return Op(dest, rhs, ops[op])

    comparisons = {"s_cmp_eq_u32": ("==", "!="), "s_cmp_lg_u32": ("!=", "=="),
                   "s_cmp_ge_u32": (">=", "<"), "s_cmp_gt_u32": (">", "<="),
                   "s_cmp_le_u32": ("<=", ">"), "s_cmp_lt_u32": ("<", ">=")}

    def arm(start, stop, incoming):
        block = []
        defined = set(incoming)
        i = start
        while i < stop:
            if insns[i]["op"] in comparisons:
                node, i, defined = conditional(i, stop, defined)
                block.append(node)
            else:
                block.append(alu(insns[i], defined))
                i += 1
        return block, defined

    def conditional(i, stop, incoming):
        cmp = insns[i]
        if (i+1 >= stop or cmp["mods"] or len(cmp["args"]) != 2 or
                insns[i+1]["op"] not in ("s_cbranch_scc0", "s_cbranch_scc1")):
            fail("expected scalar comparison and conditional jump in the same arm")
        lhs, rhs = (operand(x) for x in cmp["args"])
        if lhs.startswith("v") or rhs.startswith("v"):
            fail("expected scalar condition")
        check_reads((lhs, rhs), incoming)
        target = forward_target(i+1)
        if target > stop:
            fail("conditional jump escapes enclosing arm")
        join = fall_end = target
        if target > i+2 and insns[target-1]["op"] == "s_branch":
            join = forward_target(target-1)
            if join > stop:
                fail("branch join escapes enclosing arm")
            fall_end -= 1
        fall, left = arm(i+2, fall_end, incoming)
        taken, right = arm(target, join, incoming)
        symbol = comparisons[cmp["op"]][int(insns[i+1]["op"] == "s_cbranch_scc1")]
        return Conditional(lhs, symbol, rhs, fall, taken), join, left & right

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
        elif op in comparisons:
            if len(ballots) != 1:
                fail("expected scalar conditional between ballots")
            node, i, defined = conditional(i, len(insns)-2, defined)
            segments[-1].append(node)
            continue
        elif op in ops:
            segments[-1].append(alu(ins, defined))
            i += 1
            continue
        else:
            fail(f"unsupported instruction {op}")
        check_reads(rhs, defined)
        defined.add(dest)
        regs.add(dest)
    if len(ballots) != 2 or not set(outputs) <= defined:
        fail("expected two ballot boundaries and defined outputs")

    # A vote destination becomes available on resume. Only earlier definitions
    # still needed beyond that point require scratch storage.
    live = set(outputs)
    cuts = [None, None]
    for phase in (2, 1, 0):
        live = live_before(segments[phase], live)
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
        emit_block(body, segment, "    ")
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
