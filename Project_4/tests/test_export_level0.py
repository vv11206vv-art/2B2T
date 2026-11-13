"""Inspection exports must preserve values and existing files."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fleetkit import load_instance  # noqa: E402
from solution.export_level0 import build_exports, export_level0  # noqa: E402
from solution.fleet import oneway_flows  # noqa: E402


class Level0ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instance = load_instance(data_dir=Path(__file__).resolve().parents[1] / "data")

    def test_export_and_identical_rerun_preserve_files(self):
        with tempfile.TemporaryDirectory(prefix="sixt-level0-export-") as directory:
            target = Path(directory)
            counts = export_level0(instance=self.instance, output_dir=target)
            self.assertEqual(counts, {"initial_fleet.csv": 40, "daily_picture.csv": 560, "base_revenue.csv": 40})
            before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in target.iterdir()}
            export_level0(instance=self.instance, output_dir=target)
            self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in target.iterdir()})
            daily = pd.read_csv(target / "daily_picture.csv")
            self.assertEqual(daily.station_fleet.sum(), 2357 * 14)
            self.assertEqual(daily.committed_car_days.sum(), 8210)

    def test_conflict_reports_difference_and_blocks_all_writes(self):
        with tempfile.TemporaryDirectory(prefix="sixt-level0-conflict-") as directory:
            target = Path(directory)
            existing = target / "base_revenue.csv"
            existing.write_bytes(b"preserve this existing file\n")
            with self.assertRaisesRegex(FileExistsError, "existing contents differ.*not overwritten"):
                export_level0(instance=self.instance, output_dir=target)
            self.assertEqual(existing.read_bytes(), b"preserve this existing file\n")
            self.assertEqual({path.name for path in target.iterdir()}, {"base_revenue.csv"})

    def test_incomplete_component_is_rejected_before_join(self):
        incomplete = oneway_flows(instance=self.instance).iloc[1:]
        with patch("solution.export_level0.oneway_flows", return_value=incomplete):
            with self.assertRaisesRegex(ValueError, "key mismatch"):
                build_exports(instance=self.instance)


if __name__ == "__main__":
    unittest.main()
