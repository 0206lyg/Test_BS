"""Preparation-side artifact contracts; synthetic tables are not physics tests."""
import importlib.util
import math
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("prepare", Path(__file__).parents[1] / "tools/prepare_case.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class HardfloorArtifactTest(unittest.TestCase):
    def test_v4_basis_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "basis.ssb"
            path.write_text("SLURRY_HYBRID_V4\n20 18\nANCHORS 2 0.75 1.5\n")
            result = prepare.read_basis_header(path, 20)
            self.assertEqual(result["anchors_over_c"], [.75, 1.5])
            self.assertEqual(result["source_policy"]["gap_rule"], "max(h,h_min)")
            self.assertFalse(result["source_policy"]["far_slot_used"])

    def test_table_model_and_exact_upper_coverage(self):
        args = SimpleNamespace(ar=20, lubrication_epsilon_nm=1., finite_gap_range=.5, finite_gap_fast_nphi=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.sft"
            lines = ["SLURRY_SPD_LOGFOURIER_V4", "MODEL JANOSCHEK_MATCHED_HARDFLOOR_FULL17_V4",
                     "20 {} 10 4 4 4".format(1/150), "1 0.5",
                     " ".join(map(str, [0., math.pi/6, math.pi/3, math.pi/2])),
                     " ".join(map(str, [-math.pi, -math.pi/2, 0., math.pi/2])),
                     "{} 0.02 0.5 10".format(1/150), "DATA"]
            payload = struct.pack("d", 0.) * (4**4 * 153)
            path.write_bytes(("\n".join(lines)+"\n").encode()+payload)
            header = prepare.read_fast_table_header(path, args, 3e-6)
            self.assertEqual(header["gap_rule"], "max(h,h_min)")
            self.assertEqual(header["h_max_over_c"], 10.)
            for old in ("SLURRY_SPD_LOGFOURIER_V1", "SLURRY_SPD_LOGFOURIER_V2", "SLURRY_SPD_LOGFOURIER_V3"):
                path.write_bytes((old+"\n").encode())
                with self.assertRaisesRegex(ValueError, "old or unsupported"):
                    prepare.read_fast_table_header(path, args, 3e-6)
            lines[1] = "MODEL OLD_PHYSICAL_MODEL"
            path.write_bytes(("\n".join(lines)+"\n").encode()+payload)
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                prepare.read_fast_table_header(path, args, 3e-6)


if __name__ == "__main__":
    unittest.main()
