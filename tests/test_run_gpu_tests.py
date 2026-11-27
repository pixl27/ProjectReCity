"""scripts/run_gpu_tests.py: how it judges the output of a compute "dump" test (no device needed)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_gpu_tests as rgt

WANT = [55, 48]
DEVICE_32 = "device: X (subgroup 32..32)\n"
DEVICE_64 = "device: X (subgroup 32..64)\n"
CONFIRMED = "compute subgroup size: 64\n"


def dump(words, pad=8):
    return "out[0..7] = " + " ".join(str(w) for w in (list(words) + [0] * pad)[:8]) + "\n"


def judge(rc, msg, **kw):
    kw = {"wave64": False, "partial": False, "emulated": False, **kw}
    return rgt.judge_cs_dump(rc, msg, WANT, **kw)


class DumpWords(unittest.TestCase):
    def test_eight_decimal_words(self):
        self.assertEqual(rgt.dump_words("noise\n" + dump(WANT) + "tail\n"), WANT + [0] * 6)

    def test_anything_else_is_not_a_dump(self):
        for line in ("out[0..7] = 55 48\n", "out[0..7] = 55 48 BAD 0 0 0 0 0\n", "out[0..7] = 1 BAD 3\n",
                     "out[0..7] = 55 48 0 0 0 0 0\n", "out[0..7] = 55 48 0 0 0 0 0 0 0\n",
                     "out[0..7] = 55 48 0 0 0 0 0 0 x\n", "out[0..7] = \n", "no dump here\n"):
            with self.subTest(line=line):
                self.assertIsNone(rgt.dump_words(line))


class PackedFormatFixture(unittest.TestCase):
    def test_filtered_suite_counts_a_packed_store_mismatch(self):
        name = "cs_fmt_1010102"
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused")

            def translate(_cmd, env=None, gpu=True):
                (Path(tmp) / "spirv" / f"{name}.spv").touch()
                return 0, ""

            output = io.StringIO()
            problem = "word 15: expected 0xffe00003, got 0xe00003ff"
            with (patch.object(rgt, "assemble_to_s", side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(rgt, "spirv_as"),
                  patch.object(rgt, "run_cmd", side_effect=translate),
                  patch.object(rgt, "fmt_1010102_test", return_value=problem) as check_fmt,
                  contextlib.redirect_stdout(output)):
                self.assertEqual(rgt.run_suite(args, tmp), 1)
            check_fmt.assert_called_once()
            self.assertIn(problem, output.getvalue())
            self.assertIn("0/1 passed", output.getvalue())


class WholeWaveDump(unittest.TestCase):
    """wave64: run with --require-wave64 (the whole-wave mask tests)."""
    def judge(self, rc, msg):
        return judge(rc, msg, wave64=True)

    def test_passes_with_the_words_and_the_confirmation(self):
        self.assertEqual(self.judge(0, DEVICE_64 + CONFIRMED + dump(WANT)), (True, "55 48"))

    def test_a_device_without_64_wide_subgroups_skips(self):
        ok, detail = self.judge(77, DEVICE_32 + "full 64-wide compute subgroups unavailable\n")
        self.assertIs(ok, None)
        self.assertTrue(detail.startswith("SKIP"))

    def test_right_words_without_the_confirmation_fail(self):    # a vk_compute_test that ignores the flag
        ok, detail = self.judge(0, DEVICE_64 + dump(WANT))
        self.assertIs(ok, False)
        self.assertIn("not confirmed", detail)

    def test_wrong_word_and_harness_errors_fail(self):
        for i in range(len(WANT)):
            wrong = list(WANT)
            wrong[i] ^= 1
            with self.subTest(word=i):
                self.assertIs(self.judge(0, CONFIRMED + dump(wrong))[0], False)
        self.assertIs(self.judge(1, "Vulkan pipeline failed\n")[0], False)       # an error is not a skip
        self.assertIs(self.judge(1, CONFIRMED + dump(WANT))[0], False)

    def test_malformed_dump_lines_fail_without_raising(self):
        for line in ("out[0..7] = 55 48\n", "out[0..7] = 55 48 BAD 0 0 0 0 0\n", "out[0..7] = 1 BAD 3\n",
                     "out[0..7] = 55 48 0 0 0 0 0 0 0\n"):
            with self.subTest(line=line):
                ok, detail = self.judge(0, CONFIRMED + line)
                self.assertIs(ok, False)
                self.assertIn(line.strip(), detail)


class NotWholeWave(unittest.TestCase):
    """Dumps run without --require-wave64: cs_store, emulated wave64 (#152) and partial-group tests."""
    def test_plain_dump_needs_no_confirmation_and_a_77_is_a_failure(self):
        self.assertEqual(judge(0, dump(WANT)), (True, "55 48"))
        self.assertIs(judge(77, "unavailable\n")[0], False)

    def test_hex_words(self):
        ok, detail = rgt.judge_cs_dump(0, dump([0x400080FF, 7]), [0x400080FF, 7], wave64=False, partial=False,
                                       emulated=False, hexwords=True)
        self.assertEqual((ok, detail), (True, "0x400080ff 0x7"))

    def test_emulated_wave64_runs_on_any_subgroup_size_without_the_confirmation(self):
        self.assertEqual(judge(0, DEVICE_32 + dump(WANT), emulated=True), (True, "55 48"))
        self.assertIs(judge(77, DEVICE_32, emulated=True)[0], False)             # no skip: it must run here
        self.assertIs(judge(0, DEVICE_32 + dump([1, 2]), emulated=True)[0], False)

    def test_partial_group_skips_unless_the_device_reports_a_64_subgroup(self):
        ok, detail = judge(0, DEVICE_32 + dump(WANT), partial=True)
        self.assertIs(ok, None)
        self.assertTrue(detail.startswith("SKIP"))
        self.assertEqual(judge(0, DEVICE_64 + dump(WANT), partial=True), (True, "55 48"))   # no confirmation line
        self.assertIs(judge(0, DEVICE_64 + dump([1, 2]), partial=True)[0], False)

    def test_partial_group_emulated_runs_everywhere(self):
        self.assertEqual(judge(0, DEVICE_32 + dump(WANT), partial=True, emulated=True), (True, "55 48"))

    def test_malformed_dump_lines_fail_in_every_mode(self):
        for kw in ({}, {"emulated": True}, {"partial": True, "emulated": True}):
            with self.subTest(**kw):
                self.assertIs(judge(0, DEVICE_64 + "out[0..7] = 55 48 BAD 0 0 0 0 0\n", **kw)[0], False)



class UniformBranchCheck(unittest.TestCase):
    """check_uniform_branch (ps_uniform_branch, ps_scc_mask32) must reject a readback in which no bit=0 pixel got
    s5 = 8: that is what a translator deciding the branch per lane draws (#477, measured on RTX 5080 and RX 9070 XT:
    1856 (8,1) / 192 (8,0) / 2048 (7,0) with the vote, 1856 / 0 / 2240 without)."""

    @staticmethod
    def judge(rows):
        import os
        import struct
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".raw")
        try:
            with os.fdopen(fd, "wb") as f:
                for s5, bit, bary in rows:
                    f.write(struct.pack("<2I2f", s5, bit, bary, 0.0))
            return rgt.check_uniform_branch(path)
        finally:
            os.remove(path)

    def test_wave_vote_passes(self):
        ok, detail = self.judge([(8, 1, 0.5), (8, 0, 0.1), (7, 0, 0.1)])
        self.assertTrue(ok, detail)

    def test_per_lane_decision_fails(self):
        ok, detail = self.judge([(8, 1, 0.5), (7, 0, 0.1), (7, 0, 0.1)])
        self.assertFalse(ok)
        self.assertIn("no bit=0 pixel got s5=8", detail)

    def test_set_bit_without_8_fails(self):
        self.assertFalse(self.judge([(7, 1, 0.5), (8, 0, 0.1), (7, 0, 0.1)])[0])


class Wave32LaneDumps(unittest.TestCase):
    """run_gpu_tests.py skips cs_wqm_w32 and cs_w32_lane_wrap below 32 lanes with check.py's rule."""
    def test_uses_check_py_rule(self):
        import check
        self.assertIs(rgt.narrow_subgroup_skip, check.narrow_subgroup_skip)
        self.assertIs(rgt.WAVE32_LANE_DUMPS, check.WAVE32_LANE_DUMPS)

    def test_rule(self):
        self.assertEqual(rgt.narrow_subgroup_skip("device: llvmpipe (subgroup 8..8)\n"),
                         "SKIP (subgroup 8 is narrower than 32 lanes)")
        for out in ("device: X (subgroup 32..32)\n", "device: X (subgroup 8..32)\n", "device: X (subgroup 32..64)\n",
                    "no device line\n"):
            with self.subTest(out=out):
                self.assertIsNone(rgt.narrow_subgroup_skip(out))


class WaveMaskFixture(unittest.TestCase):
    def test_filtered_suite_dispatches_and_counts_once(self):
        name = "cs_wavemask"
        for mode, require_native in (("0", True), ("yes", False)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                args = SimpleNamespace(filter=name, build=tmp, llvm_mc="unused")
                dispatches = []

                def run(cmd, env=None, gpu=True):
                    if gpu:
                        dispatches.append(cmd)
                        return 0, "compute subgroup size: 64\n0 mismatches"
                    self.assertEqual("--wave64-emulate" in cmd, not require_native)
                    (Path(tmp) / "spirv" / f"{name}.spv").touch()
                    return 0, ""

                output = io.StringIO()
                with (patch.dict(rgt.os.environ, RECITY_W64_EMULATE=mode),
                      patch.object(rgt, "assemble_to_s", side_effect=lambda mc, src, dst, n: Path(dst).touch()),
                      patch.object(rgt, "spirv_as"), patch.object(rgt, "run_cmd", side_effect=run),
                      contextlib.redirect_stdout(output)):
                    self.assertEqual(rgt.run_suite(args, tmp), 0)
                self.assertEqual(len(dispatches), 1)
                self.assertEqual("--require-wave64" in dispatches[0], require_native)
                self.assertIn("1/1 passed", output.getvalue())


if __name__ == "__main__":
    unittest.main()
