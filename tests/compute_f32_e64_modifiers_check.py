"""Check the VOP3 binary32 add, sub, subrev and mul with neg and abs source modifiers, translated and dispatched."""
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compute_store_check as base  # noqa: E402  (also puts tools/ on the path)

aot = base.aot
NAME = "read_f32_e64_modifiers"
CAP = base.F32_CAP  # the float 20.0, constant word 1 (s0)
HALF = 0x3fe00000  # the float 1.75, a literal (not an inline constant)
# The read-compute-write fixture with the operand construction of the binary32 fixture in compute_store_check.py
# (x in [1.5, 2) and y in [3, 4) from the source word w) and a chain of VOP3 operations, hand-edited in LLVM 15
# spelling (llvm-mc -mcpu=gfx1013 encodes every line):
#   a = x + -y;  b = cap - |a|;  t = x - 1.75;  d = -|t| * b;  e = d - -cap (subrev);  out = e + |t|
# a is always negative, so |a| is not a. t is negative in some lanes and positive in others, so -|t| differs from t
# in the positive lanes and from -t in the negative ones, and |t| differs from t in the negative ones. The modifiers
# cover a VGPR and an SGPR, both slots, and all three forms (-r, |r|, -|r|); every operation is in the _e64 spelling.
CHAIN = ("v_add_f32_e64 v4, v2, -v3", "v_sub_f32_e64 v5, s0, |v4|", f"v_sub_f32_e64 v7, v2, 0x{HALF:08x}",
         "v_mul_f32_e64 v8, -|v7|, v5", "v_subrev_f32_e64 v6, -s0, v8", "v_add_f32_e64 v1, v6, |v7|")
LINES = base.variant(
    base.READ_COMPUTE,
    ("v_and_b32_e32 v2, 0xffff, v1", "v_and_b32_e32 v2, 0x3fffff, v1\nv_or_b32_e32 v2, 0x3fc00000, v2\n"
                                     "v_lshrrev_b32_e32 v3, 10, v1\nv_or_b32_e32 v3, 0x40400000, v3"),
    ("v_lshrrev_b32_e32 v1, 16, v1", "\n".join(CHAIN)),
    ("v_mad_u32_u24 v1, v2, v1, v0", ""),
    ("v_min_u32_e32 v1, s0, v1", ""),
)
BINDING = base.READ_BINDING
LOAD = base.READ_LOADS["read_copy"]
X = f"(0x3fc00000u | (0x003fffffu & {LOAD}))"
Y = f"(0x40400000u | ({LOAD} >> (0x0000000au & 31u)))"
A = f"f32_add({X}, ({Y} ^ 0x80000000u))"
B = f"f32_sub(params[0].y, ({A} & 0x7fffffffu))"
T = f"f32_sub({X}, 0x{HALF:08x}u)"
D = f"f32_mul(({T} | 0x80000000u), {B})"
E = f"f32_sub({D}, (params[0].y ^ 0x80000000u))"
STORE = f"        outputData.Store(tid.x * 4u + 0u, f32_add({E}, ({T} & 0x7fffffffu)));"

MODIFIERS = {"": lambda v: v, "neg": lambda v: -v, "abs": abs, "neg_abs": lambda v: -abs(v)}


def steps(gid, *, sites=(), swap=()):
    """Independent CPU model of one lane, every value. sites replaces the modifier at a named site; swap exchanges
    the operands of the sub that gives b or the subrev that gives e. Python's -v and abs(v) change only the sign of a binary32 value."""
    def mod(site, default):
        return MODIFIERS[dict(sites).get(site, default)]

    r, value = base.f32_round, base.f32_value
    word = base.source_word(gid)
    x, y = value(0x3fc00000 | word & 0x3fffff), value(0x40400000 | word >> 10)
    cap = value(CAP)
    a = r(x + mod("y", "neg")(y))
    b = r(mod("a", "abs")(a) - cap if "b" in swap else cap - mod("a", "abs")(a))
    t = r(x - value(HALF))
    d = r(mod("t", "neg_abs")(t) * b)
    e = r(mod("cap", "neg")(cap) - d if "e" in swap else d - mod("cap", "neg")(cap))
    return x, y, a, b, t, d, e, r(e + mod("t2", "abs")(t))


def words(groups, bound, **reference):
    result = [base.SENTINEL] * 320
    for gid in range(min(32 * groups, bound)):
        result[gid] = base.f32_bits(steps(gid, **reference)[-1])
    return result


# Wrong references; each must fail the run at the first word where it differs. Every modifier is dropped, the
# -|t| site also loses only its neg or only its abs, and the sub and subrev have their operands exchanged. (The
# operand order of t = x - 1.75 is not observable: t is only read through abs.)
WRONG = (("neg dropped on y", dict(sites=(("y", ""),))), ("abs dropped on a", dict(sites=(("a", ""),))),
         ("-|t| as t", dict(sites=(("t", ""),))), ("-|t| as -t", dict(sites=(("t", "neg"),))),
         ("-|t| as |t|", dict(sites=(("t", "abs"),))), ("neg dropped on cap", dict(sites=(("cap", ""),))),
         ("abs dropped on the last t", dict(sites=(("t2", ""),))), ("sub exchanged", dict(swap=("b",))),
         ("subrev exchanged", dict(swap=("e",))))
# Translations a wrong modifier or template would produce; each must fail on the GPU.
MUTATIONS = (("neg on y as abs", f"{Y} ^ 0x80000000u", f"{Y} & 0x7fffffffu"),
             ("abs on a dropped", f"({A} & 0x7fffffffu)", A),
             ("-|t| as -t", f"({T} | 0x80000000u)", f"({T} ^ 0x80000000u)"),
             ("-|t| as |t|", f"({T} | 0x80000000u)", f"({T} & 0x7fffffffu)"),
             ("neg on the SGPR dropped", "(params[0].y ^ 0x80000000u)", "params[0].y"),
             ("abs on the last t as neg", f"{E}, ({T} & 0x7fffffffu)", f"{E}, ({T} ^ 0x80000000u)"),
             ("subrev as sub", E, f"f32_sub((params[0].y ^ 0x80000000u), {D})"))


def lower(*changes, **binding):
    return aot.lower(base.variant(LINES, *changes), 32, **(BINDING | binding)).splitlines()


def controls():
    """Model checks, exact output, then one rejection per modifier, encoding or operand outside the form."""
    lanes = [steps(gid) for gid in range(base.READ_RUNS[-1][1])]
    for lane in lanes:
        for v in lane:
            # Every value is finite and normal, or a zero (t and d are zero where x is exactly 1.75).
            assert v == 0 or 0x00800000 <= base.f32_bits(v) & 0x7fffffff < 0x7f800000, lane
    assert all(lane[2] < 0 for lane in lanes), "a must be negative in every lane"
    assert any(lane[4] < 0 for lane in lanes) and any(lane[4] > 0 for lane in lanes), "t must take both signs"
    assert any(base.f32_round(lane[4] * lane[3]) != lane[4] * lane[3] for lane in lanes), "every multiply exact"
    expected = words(*base.READ_RUNS[-1])
    for label, reference in WRONG:
        assert words(*base.READ_RUNS[-1], **reference) != expected, label

    lines = lower()
    helpers = [line.split("(")[0] for line in lines if line.startswith("uint f32_")]
    assert helpers == ["uint f32_add", "uint f32_sub", "uint f32_mul"], helpers
    assert lines[-4:] == [base.GUARD_IF, STORE, "    }", "}"], lines
    # Without modifiers the _e64 spelling lowers exactly like the _e32 one, and a copy of a result is the same program.
    assert (aot.lower(base.variant(base.READ_F32, ("v_add_f32_e32 v4, v2, v3", "v_add_f32_e64 v4, v2, v3"),
                                   ("v_mul_f32_e32 v6, v2, v6", "v_mul_f32_e64 v6, v2, v6")), 32, **BINDING)
            == aot.lower(base.READ_F32, 32, **BINDING))
    assert lower((CHAIN[-1], "v_mov_b32_e32 v9, v7\nv_add_f32_e64 v1, v6, |v9|")) == lines
    # The _e64 slots take any source: an SGPR as src1 and an inline constant, with or without a modifier.
    assert lower((CHAIN[4], "v_subrev_f32_e64 v6, v8, -|s0|"))[-3] == STORE.replace(
        f"f32_sub({D}, (params[0].y ^ 0x80000000u))", f"f32_sub((params[0].y | 0x80000000u), {D})")
    assert f"f32_sub({X}, 0xbf800000u)" in lower((f"v2, 0x{HALF:08x}", "v2, -1.0"))[-3]

    unsupported, operand = "Unsupported compute instruction: ", "Unverified float operand"
    first = CHAIN[0]
    for message, changes in (
        # A modifier on a constant, a modifier spelling LLVM does not print for these opcodes, and modifiers on the
        # _e32 spelling, which has no modifier bits.
        (unsupported + "v_add_f32_e64", ((first, "v_add_f32_e64 v4, v2, -|0.5|"),)),
        (unsupported + "v_add_f32_e64", ((first, "v_add_f32_e64 v4, v2, |0x3fe00000|"),)),
        (unsupported + "v_add_f32_e64", ((first, "v_add_f32_e64 v4, v2, neg(v3)"),)),
        (unsupported + "v_add_f32_e64", ((first, "v_add_f32_e64 v4, v2, abs(v3)"),)),
        (unsupported + "v_add_f32_e64", ((first, "v_add_f32_e64 v4, v2, |-v3|"),)),
        (unsupported + "v_add_f32_e32", ((first, "v_add_f32_e32 v4, v2, -v3"),)),
        (unsupported + "v_add_f32_e32", ((first, "v_add_f32_e32 v4, -v3, v2"),)),
        # Output modifiers and clamp, a register range, a third source, and _e64 forms outside the subset.
        (unsupported + "v_add_f32_e64", ((first, first + " clamp"),)),
        (unsupported + "v_add_f32_e64", ((first, first + " mul:2"),)),
        (unsupported + "v_add_f32_e64", ((first, first + " div:2"),)),
        (unsupported + "v_add_f32_e64", ((first, "v_add_f32_e64 v4, v2, v[3:4]"),)),
        (unsupported + "v_add_f32_e64", ((first, first + ", v2"),)),
        (unsupported + "v_max_f32_e64", ((first, "v_max_f32_e64 v4, v2, -v3"),)),
        (unsupported + "v_fma_f32", ((first, "v_fma_f32 v4, v2, -v3, v2"),)),
        (unsupported + "v_mul_legacy_f32_e64", ((first, "v_mul_legacy_f32_e64 v4, v2, -v3"),)),
        # Sources that are not known words, with and without a modifier, NaN, infinite and denormal constants,
        # and a destination past v255.
        (operand, ((first, "v_add_f32_e64 v4, v2, -v9"),)),
        (operand, ((first, "v_add_f32_e64 v4, |s20|, v3"),)),
        (operand, ((first, "v_add_f32_e64 v4, -|s8|, v3"),)),
        (operand, ((first, "v_add_f32_e64 v4, v2, 0x7fc00000"),)),
        (operand, ((first, "v_add_f32_e64 v4, 0xff800000, -v3"),)),
        (operand, ((first, "v_add_f32_e64 v4, v2, 0x00000001"),)),
        (operand, ((first, "v_add_f32_e64 v4, v2, 1.5"),)),
        (operand, ((first, "v_add_f32_e64 v256, v2, -v3"),)),
        # A result is data only: not a store address and not a lane-guard compare operand.
        ("Unsupported store address", (("v1, v0, s[8:11]", "v0, v1, s[8:11]"),)),
        ("Unverified lane guard compare", ((CHAIN[-1], CHAIN[-1] + "\nv_cmp_gt_u32_e32 vcc_lo, s3, v1"),)),
        ("ALU expression too large", ((CHAIN[-1], CHAIN[-1] + "\nv_add_f32_e64 v1, -|v1|, v1" * 3),)),
    ):
        base.rejects(message, lower, *changes)


def draw(directory, compiler, renderer, backend):
    """Translate, compile and dispatch the fixture, then require wrong references and mutated sources to fail."""
    src, cso, case = [directory / x for x in ("store.hlsl", "store.cso", "store.case")]
    source = aot.lower(LINES, 32, **BINDING)
    src.write_text(source, encoding="ascii")
    subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", cso, src], check=True)
    cmd = [renderer, backend, "--compute-store-case", cso, case]
    constants = [0x5eed0000 + i for i in range(16)]
    constants[1] = CAP

    def write(groups, bound, result):
        case.write_bytes(struct.pack("<337I", groups, *result, bound, *constants[1:]))

    for groups, bound in base.READ_RUNS:
        write(groups, bound, words(groups, bound))
        result = subprocess.run(cmd)
        if result.returncode:
            return result.returncode
    groups, bound = base.READ_RUNS[-1]
    expected = words(groups, bound)
    for label, reference in ((label, words(groups, bound, **wrong)) for label, wrong in WRONG):
        word = next(i for i, (a, b) in enumerate(zip(expected, reference)) if a != b)
        write(groups, bound, reference)
        bad = subprocess.run(cmd, capture_output=True, text=True)
        if bad.returncode != 1 or f"word={word} expected={reference[word]:08x}" not in bad.stderr:
            raise AssertionError(f"{NAME}: reference '{label}' escaped detection\n{bad.stderr}")
        print(f"{label}: word={word} expected={reference[word]:08x} actual={expected[word]:08x}")
    write(groups, bound, expected)
    for label, old, new in MUTATIONS:
        mutated = source.replace(old, new, 1)
        if mutated == source:
            raise AssertionError(f"{NAME}: mutation '{label}' did not change the source")
        bad_src, bad_cso = directory / "mutated.hlsl", directory / "mutated.cso"
        bad_src.write_text(mutated, encoding="ascii")
        subprocess.run([compiler, "-T", "cs_6_0", "-E", "main", "-Fo", bad_cso, bad_src], check=True)
        bad = subprocess.run([*cmd[:3], bad_cso, case], capture_output=True, text=True)
        if bad.returncode != 1 or "Compute readback differs" not in bad.stderr:
            raise AssertionError(f"{NAME}: mutation '{label}' escaped detection\n{bad.stderr}")
    if record := os.environ.get("FORM_COVERAGE_RECORD"):
        path = Path(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        accepted = sorted({line.split()[0] for line in LINES if not line.endswith(":")})
        path.write_text("".join(f"{op}\n" for op in accepted), encoding="utf-8")
    print(f"{NAME}: {len(base.READ_RUNS)} dispatches, {len(WRONG)} wrong references and "
          f"{len(MUTATIONS)} mutated sources checked")
    return 0


if __name__ == "__main__":
    mode, *args = sys.argv[1:]
    controls()
    if mode == "controls":
        assert not args, args
        sys.exit(0)
    if mode != "draw" or len(args) != 3:
        sys.exit("usage: compute_f32_e64_modifiers_check.py controls | draw <dxc> <renderer> <adapter>")
    with tempfile.TemporaryDirectory() as tmp:
        sys.exit(draw(Path(tmp), *args))
