"""Focused Level 0 checks with a deliberately small fleet and horizon."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution.fleet import (  # noqa: E402
    check_plan_validity,
    committed_car_days,
    compare_check_values,
    initial_fleet,
    oneway_flows,
    simulate_fleet,
    transit_cars,
    validate_plan,
)
from fleetkit import baseline_plan, load_instance  # noqa: E402


class FleetLevel0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SimpleNamespace(
            horizon=pd.DataFrame({"first_date": pd.to_datetime(["2027-03-01"]), "last_date": pd.to_datetime(["2027-03-04"])}),
            stations=pd.DataFrame({"station_id": ["S1", "S2"]}),
            categories=pd.DataFrame({"category": ["economy", "premium"], "addition_cap": [1, 1], "removal_cap": [1, 1]}),
            distances=pd.DataFrame({"from_station_id": ["S1", "S2"], "to_station_id": ["S2", "S1"], "lead_time_days": [1, 2]}),
            transfer_costs=pd.DataFrame({"mode": ["driver", "truck"]}),
            fleet_snapshot=pd.DataFrame(
                [
                    ("V1", "economy", "on_rent", None, "R1"),
                    ("V2", "economy", "idle", "S2", None),
                ],
                columns=["vehicle_id", "category", "status", "station_id", "reservation_id"],
            ),
            reservations=pd.DataFrame(
                [
                    ("R1", "economy", "S1", "2027-02-28", "S2", "2027-03-02", "open"),
                    ("R2", "economy", "S2", "2027-03-02", "S2", "2027-03-03", "booked"),
                    ("R3", "premium", "S1", "2027-02-20", "S1", "2027-02-21", "completed"),
                    ("R4", "economy", "S1", "2027-03-03", "S2", "2027-03-04", "booked"),
                ],
                columns=["reservation_id", "category", "pickup_station_id", "pickup_date", "return_station_id", "return_date", "status"],
            ),
        )
        for column in ("pickup_date", "return_date"):
            self.instance.reservations[column] = pd.to_datetime(self.instance.reservations[column])

    @staticmethod
    def value(df: pd.DataFrame, station: str, category: str, date: str, column: str) -> int:
        row = df.loc[
            df["station_id"].eq(station)
            & df["category"].eq(category)
            & df["date"].eq(pd.Timestamp(date))
        ]
        return int(row[column].item())

    def test_on_rent_snapshot_car_belongs_to_pickup_station(self) -> None:
        fleet = initial_fleet(instance=self.instance).set_index(["station_id", "category"])
        self.assertEqual(fleet.loc[("S1", "economy"), "cars"], 1)
        self.assertEqual(fleet.loc[("S2", "economy"), "cars"], 1)
        self.assertEqual(fleet["cars"].sum(), len(self.instance.fleet_snapshot))

    def test_missing_or_unknown_snapshot_reservation_links_are_rejected(self) -> None:
        for reservation_id in [None, "", "NOT_A_RESERVATION"]:
            with self.subTest(reservation_id=reservation_id):
                self.instance.fleet_snapshot.loc[0, "reservation_id"] = reservation_id
                with self.assertRaisesRegex(ValueError, "missing or unknown reservation links.*V1"):
                    initial_fleet(instance=self.instance)

    def test_initial_fleet_checks_snapshot_car_count_conservation(self) -> None:
        self.instance.fleet_snapshot.loc[1, "station_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "count mismatch: counted 1 cars from 2 snapshot cars"):
            initial_fleet(instance=self.instance)

    def test_pickup_and_return_dates_are_inclusive(self) -> None:
        committed = committed_car_days(instance=self.instance)
        self.assertEqual(self.value(committed, "S1", "economy", "2027-03-02", "committed_car_days"), 1)
        self.assertEqual(self.value(committed, "S1", "economy", "2027-03-03", "committed_car_days"), 1)
        self.assertEqual(self.value(committed, "S1", "economy", "2027-03-04", "committed_car_days"), 1)
        self.assertEqual(self.value(committed, "S2", "economy", "2027-03-03", "committed_car_days"), 1)

    def test_oneway_moves_day_after_return_only(self) -> None:
        flows = oneway_flows(instance=self.instance)
        fleet = simulate_fleet(instance=self.instance, plan=pd.DataFrame())
        self.assertEqual(self.value(flows, "S1", "economy", "2027-03-02", "oneway_departures"), 0)
        self.assertEqual(self.value(flows, "S1", "economy", "2027-03-03", "oneway_departures"), 1)
        self.assertEqual(self.value(flows, "S2", "economy", "2027-03-03", "oneway_arrivals"), 1)
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-02", "station_fleet"), 1)
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-03", "station_fleet"), 0)
        self.assertEqual(self.value(fleet, "S2", "economy", "2027-03-03", "station_fleet"), 2)
        self.assertEqual(self.value(flows, "S2", "economy", "2027-03-04", "oneway_arrivals"), 0)

    def test_complete_grid_includes_zero_cells(self) -> None:
        for table, column in (
            (initial_fleet(instance=self.instance), "cars"),
            (committed_car_days(instance=self.instance), "committed_car_days"),
            (oneway_flows(instance=self.instance), "oneway_arrivals"),
            (simulate_fleet(instance=self.instance, plan=pd.DataFrame()), "station_fleet"),
        ):
            self.assertEqual(len(table), 4 if column == "cars" else 16)
            self.assertEqual(table.loc[table["category"].eq("premium"), column].sum(), 0)

    def test_comparison_reports_location_and_values(self) -> None:
        self.instance.check_values = pd.DataFrame(
            [("S1", "economy", pd.Timestamp("2027-03-01"), 99, 0, 0, 1)],
            columns=["station_id", "category", "date", "committed_car_days", "oneway_arrivals", "oneway_departures", "station_fleet"],
        )
        mismatch = compare_check_values(instance=self.instance)
        self.assertEqual(len(mismatch), 1)
        self.assertEqual(mismatch.iloc[0][["station_id", "category", "metric", "expected", "calculated"]].tolist(), ["S1", "economy", "committed_car_days", 99, 1])

    @staticmethod
    def plan(*rows: tuple) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["action", "date", "category", "station_id", "to_station_id", "mode", "cars"])

    def test_addition_and_removal_apply_same_day(self) -> None:
        plan = self.plan(("add", "2027-03-01", "economy", "S1", None, None, 2),
                         ("remove", "2027-03-02", "economy", "S1", None, None, 1))
        fleet = simulate_fleet(instance=self.instance, plan=plan)
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-01", "station_fleet"), 3)
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-02", "station_fleet"), 2)
        self.assertEqual(len(fleet), 16)

    def test_one_and_two_day_transfers_and_transit(self) -> None:
        plan = self.plan(("transfer", "2027-03-01", "economy", "S1", "S2", "driver", 1),
                         ("transfer", "2027-03-01", "economy", "S2", "S1", "truck", 1))
        fleet = simulate_fleet(instance=self.instance, plan=plan)
        transit = transit_cars(instance=self.instance, plan=plan)
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-01", "station_fleet"), 0)
        self.assertEqual(self.value(fleet, "S2", "economy", "2027-03-02", "station_fleet"), 1)
        # The one-way rental also leaves S1 on March 3, offsetting this arrival.
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-03", "station_fleet"), 0)
        two_day_only = simulate_fleet(instance=self.instance, plan=self.plan(
            ("transfer", "2027-03-01", "economy", "S2", "S1", "truck", 1)))
        self.assertEqual(self.value(two_day_only, "S1", "economy", "2027-03-03", "station_fleet"), 1)
        self.assertEqual(transit.set_index(["category", "date"]).loc[("economy", pd.Timestamp("2027-03-01")), "in_transit"], 2)
        self.assertEqual(transit.set_index(["category", "date"]).loc[("economy", pd.Timestamp("2027-03-02")), "in_transit"], 1)
        self.assertEqual(transit.set_index(["category", "date"]).loc[("economy", pd.Timestamp("2027-03-03")), "in_transit"], 0)

    def test_arrival_after_horizon_stays_in_transit(self) -> None:
        plan = self.plan(("transfer", "2027-03-04", "economy", "S2", "S1", "driver", 1))
        fleet = simulate_fleet(instance=self.instance, plan=plan)
        transit = transit_cars(instance=self.instance, plan=plan)
        self.assertEqual(self.value(fleet, "S1", "economy", "2027-03-04", "station_fleet"), 0)
        self.assertEqual(transit.set_index(["category", "date"]).loc[("economy", pd.Timestamp("2027-03-04")), "in_transit"], 1)

    def test_shortage_can_appear_days_after_action(self) -> None:
        plan = self.plan(("add", "2027-03-01", "economy", "S1", None, None, 1),
                         ("remove", "2027-03-02", "economy", "S1", None, None, 1))
        shortages = [v for v in check_plan_validity(instance=self.instance, plan=plan) if v["type"] == "fleet_shortage"]
        self.assertTrue(any(v["station_id"] == "S1" and v["date"] == pd.Timestamp("2027-03-03")
                            and v["station_fleet"] == 0 and v["committed_car_days"] == 1 and v["shortage"] == 1
                            for v in shortages))

    def test_caps_apply_across_stations_and_dates(self) -> None:
        plan = self.plan(("add", "2027-03-01", "economy", "S1", None, None, 1),
                         ("add", "2027-03-02", "economy", "S2", None, None, 1),
                         ("remove", "2027-03-03", "premium", "S1", None, None, 1),
                         ("remove", "2027-03-04", "premium", "S2", None, None, 1))
        caps = [v for v in check_plan_validity(instance=self.instance, plan=plan) if v["type"] == "category_cap"]
        self.assertEqual({(v["category"], v["action"], v["total"], v["cap"]) for v in caps},
                         {("economy", "add", 2, 1), ("premium", "remove", 2, 1)})

    def test_invalid_values_and_same_station_transfer(self) -> None:
        plan = self.plan(("transfer", "2027-03-01", "economy", "S1", "S1", "ferry", 1.5),
                         ("add", "2027-03-09", "unknown", "S9", "S2", None, 0))
        errors = check_plan_validity(instance=self.instance, plan=plan)
        self.assertEqual([v["row"] for v in errors], [1, 2])
        self.assertIn("destination must differ", errors[0]["message"])
        self.assertIn("cars must be a positive integer", errors[0]["message"])
        self.assertIn("date must be a horizon date", errors[1]["message"])
        with self.assertRaises(ValueError):
            simulate_fleet(instance=self.instance, plan=plan)

    def test_supplied_do_nothing_and_generated_baseline_are_valid(self) -> None:
        project = Path(__file__).resolve().parents[1]
        instance = load_instance(data_dir=project / "data")
        empty = pd.read_csv(project / "plans" / "do_nothing.csv")
        self.assertEqual(check_plan_validity(instance=instance, plan=empty), [])
        self.assertIsNone(validate_plan(instance=instance, plan=baseline_plan(instance=instance)))


if __name__ == "__main__":
    unittest.main()
