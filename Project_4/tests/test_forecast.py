"""Small-data checks for the simple Level 0 estimates."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution.fleet import committed_car_days  # noqa: E402
from solution.forecast import base_revenue, demand_to_come  # noqa: E402


class ForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SimpleNamespace(
            horizon=pd.DataFrame({"first_date": pd.to_datetime(["2027-03-01"]), "last_date": pd.to_datetime(["2027-03-14"])}),
            stations=pd.DataFrame({"station_id": ["S1", "S2"]}),
            categories=pd.DataFrame({"category": ["economy", "premium", "midsize"]}),
            reservations=pd.DataFrame(
                [
                    ("R1", "economy", "S1", "2027-01-04", "2027-01-05", 300.0, "completed"),
                    ("R2", "economy", "S1", "2027-01-11", "2027-01-11", 90.0, "completed"),
                    ("R3", "premium", "S1", "2027-01-04", "2027-01-05", 200.0, "completed"),
                    ("R4", "economy", "S1", "2027-03-01", "2027-03-02", 0.0, "booked"),
                ],
                columns=["reservation_id", "category", "pickup_station_id", "pickup_date", "return_date", "revenue", "status"],
            ),
        )
        for column in ("pickup_date", "return_date"):
            self.instance.reservations[column] = pd.to_datetime(self.instance.reservations[column])

    def test_inclusive_length_and_weighted_base_revenue(self) -> None:
        rates = base_revenue(instance=self.instance).set_index(["station_id", "category"])
        # 300 / 2 days plus 90 / 1 day, weighted by the three car-days.
        self.assertAlmostEqual(rates.loc[("S1", "economy"), "base_revenue_per_car_day"], 390 / 3)
        self.assertEqual(rates.loc[("S1", "economy"), "estimate_source"], "station_category")

    def test_zero_rental_weekdays_count_in_mean(self) -> None:
        forecast = demand_to_come(instance=self.instance, committed=committed_car_days(instance=self.instance))
        cell = forecast.set_index(["station_id", "category", "date"])["demand_to_come"]
        # Two occupied historical Mondays across eight Mondays; no commitment March 8.
        self.assertAlmostEqual(cell.loc[("S1", "economy", pd.Timestamp("2027-03-08"))], 2 / 8)
        self.assertEqual(cell.loc[("S2", "economy", pd.Timestamp("2027-03-08"))], 0)
        self.assertEqual(len(forecast), 2 * 3 * 14)

    def test_subtracts_commitments_and_clips_at_zero(self) -> None:
        forecast = demand_to_come(instance=self.instance, committed=committed_car_days(instance=self.instance))
        cell = forecast.set_index(["station_id", "category", "date"])["demand_to_come"]
        self.assertEqual(cell.loc[("S1", "economy", pd.Timestamp("2027-03-01"))], 0)
        self.assertEqual(cell.loc[("S1", "economy", pd.Timestamp("2027-03-02"))], 0)
        self.assertTrue(forecast["demand_to_come"].ge(0).all())

    def test_sparse_history_falls_back_to_category_then_overall(self) -> None:
        rates = base_revenue(instance=self.instance).set_index(["station_id", "category"])
        self.assertEqual(len(rates), 6)
        self.assertEqual(rates.loc[("S2", "economy"), "estimate_source"], "category")
        self.assertAlmostEqual(rates.loc[("S2", "economy"), "base_revenue_per_car_day"], 390 / 3)
        self.assertEqual(rates.loc[("S2", "midsize"), "estimate_source"], "overall")
        self.assertAlmostEqual(rates.loc[("S2", "midsize"), "base_revenue_per_car_day"], 590 / 5)

    def test_no_history_does_not_invent_a_rate(self) -> None:
        self.instance.reservations = self.instance.reservations.loc[self.instance.reservations["status"].eq("booked")]
        with self.assertRaisesRegex(ValueError, "no completed rentals"):
            base_revenue(instance=self.instance)

    def test_commitment_grid_rejects_missing_extra_and_duplicate_keys(self) -> None:
        valid = committed_car_days(instance=self.instance)
        extra = valid.iloc[:1].copy()
        extra["station_id"] = "UNKNOWN"
        outside = valid.iloc[:1].copy()
        outside["date"] = pd.Timestamp("2027-03-15")
        for label, table, message in [
            ("missing", valid.iloc[1:], "missing 1 keys"),
            ("extra", pd.concat([valid, extra]), "extra 1 keys"),
            ("outside horizon", pd.concat([valid, outside]), "extra 1 keys"),
            ("duplicate", pd.concat([valid, valid.iloc[:1]]), "duplicate keys"),
            ("missing column", valid.drop(columns="committed_car_days"), "missing required columns"),
        ]:
            with self.subTest(label=label), self.assertRaisesRegex(ValueError, message):
                demand_to_come(instance=self.instance, committed=table)

    def test_commitments_reject_null_keys_and_values(self) -> None:
        for column in ["station_id", "category", "date", "committed_car_days"]:
            table = committed_car_days(instance=self.instance)
            if column == "committed_car_days":
                table[column] = table[column].astype(float)
            table.loc[0, column] = None
            with self.subTest(column=column), self.assertRaisesRegex(ValueError, "null"):
                demand_to_come(instance=self.instance, committed=table)

    def test_commitments_reject_negative_nonfinite_and_nonnumeric_values(self) -> None:
        for value in [-1., float("inf"), -float("inf"), "invalid"]:
            table = committed_car_days(instance=self.instance)
            table["committed_car_days"] = table["committed_car_days"].astype(object if isinstance(value, str) else float)
            table.loc[0, "committed_car_days"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "numeric|finite and nonnegative"):
                demand_to_come(instance=self.instance, committed=table)

    def test_valid_commitments_keep_complete_forecast_and_are_not_mutated(self) -> None:
        valid = committed_car_days(instance=self.instance)
        before = valid.copy(deep=True)
        forecast = demand_to_come(instance=self.instance, committed=valid)
        self.assertEqual(len(forecast), 84)
        self.assertFalse(forecast.isna().any().any())
        self.assertAlmostEqual(forecast.set_index(["station_id", "category", "date"]).loc[("S1", "economy", pd.Timestamp("2027-03-08")), "demand_to_come"], .25)
        pd.testing.assert_frame_equal(valid, before, check_exact=True)


if __name__ == "__main__":
    unittest.main()
