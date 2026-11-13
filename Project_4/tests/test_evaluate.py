"""Manually worked revenue and cost cases for README R6-R11."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution.evaluate import estimate_profit, revenue_for_served  # noqa: E402
from solution.fleet import simulate_fleet, PLAN_COLUMNS  # noqa: E402


class EvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.names = ["economy", "compact", "midsize", "premium"]
        self.instance = SimpleNamespace(
            horizon=pd.DataFrame({"first_date": pd.to_datetime(["2027-03-01"]), "last_date": pd.to_datetime(["2027-03-03"])}),
            stations=pd.DataFrame({"station_id": ["S1", "S2"], "station_type": ["downtown", "downtown"], "max_utilization": [1.0, 1.0]}),
            categories=pd.DataFrame({"category": self.names, "rank": [1, 2, 3, 4], "holding_cost_per_day": [10, 20, 30, 40], "addition_cost_per_car": [100]*4, "addition_cap": [100]*4, "removal_cap": [100]*4}),
            distances=pd.DataFrame({"from_station_id": ["S1", "S2"], "to_station_id": ["S2", "S1"], "distance_km": [10, 10], "lead_time_days": [2, 2]}),
            transfer_costs=pd.DataFrame({"mode": ["driver", "truck"], "fixed_cost": [5, 100], "cost_per_km": [2, 3], "capacity": [1, 8]}),
            reservations=pd.DataFrame(columns=["reservation_id", "status", "category", "pickup_station_id", "return_station_id", "pickup_date", "return_date", "revenue"]),
            price_factors=pd.DataFrame([( "downtown", cat, share, 1.0) for cat in self.names for share in [.6, .7, .8, .9, 1]], columns=["station_type", "category", "served_share", "price_factor"]),
        )
        for column in ["pickup_date", "return_date"]:
            self.instance.reservations[column] = pd.to_datetime(self.instance.reservations[column])
        self.set_cars([0, 0, 0, 0])

    def set_cars(self, counts: list[int]) -> None:
        rows = [(f"V{cat}{i}", cat, "idle", "S1", None) for cat, n in zip(self.names, counts) for i in range(n)]
        self.instance.fleet_snapshot = pd.DataFrame(rows, columns=["vehicle_id", "category", "status", "station_id", "reservation_id"])

    def evaluate(self, quantities=None, plan=None, **kwargs):
        if plan is None:
            plan = pd.DataFrame(columns=PLAN_COLUMNS)
        fleet = simulate_fleet(instance=self.instance, plan=plan)
        demand = fleet[["station_id", "category", "date"]].copy()
        demand["demand_to_come"] = 0.0
        for cat, value in zip(self.names, quantities or [0]*4):
            demand.loc[demand["station_id"].eq("S1") & demand["category"].eq(cat) & demand["date"].eq(pd.Timestamp("2027-03-01")), "demand_to_come"] = value
        base = pd.DataFrame([(station, cat, rate) for station in ["S1", "S2"] for cat, rate in zip(self.names, [100, 200, 300, 400])], columns=["station_id", "category", "base_revenue_per_car_day"])
        return estimate_profit(instance=self.instance, plan=plan, demand=demand, base_revenue=base, **kwargs)

    def test_appendix_c_total_revenue_interpolation(self) -> None:
        factors = pd.DataFrame({"served_share": [.6, .7, .8, .9, 1], "price_factor": [1.30, 1.21, 1.14, 1.07, 1]})
        for served, expected in [(75, 21987.5), (40, 13000), (60, 19500), (100, 25000)]:
            self.assertAlmostEqual(revenue_for_served(demand=100, served=served, base=250, factors=factors), expected)
        self.assertEqual(revenue_for_served(demand=0, served=0, base=250, factors=factors), 0)

    def test_irregular_downtown_midsize_factors_are_preserved(self) -> None:
        factors = pd.read_csv(Path(__file__).resolve().parents[1] / "data/price_factors.csv")
        factors = factors.loc[factors["station_type"].eq("downtown") & factors["category"].eq("midsize")]
        # Revenue at 70/80/90 served: 7392, 7920, 9144. Marginal revenue rises.
        self.assertAlmostEqual(revenue_for_served(demand=100, served=75, base=100, factors=factors), 7656)
        self.assertAlmostEqual(revenue_for_served(demand=100, served=85, base=100, factors=factors), 8532)

    def test_own_category_before_upgrades(self) -> None:
        self.set_cars([0, 1, 0, 0])
        self.assertEqual(self.evaluate([1, 1, 0, 0])["new_demand_revenue"], 200)

    def test_one_category_upgrade_earns_requested_rate(self) -> None:
        self.set_cars([0, 2, 0, 0])
        self.assertEqual(self.evaluate([1, 1, 0, 0])["new_demand_revenue"], 300)

    def test_no_two_category_upgrade(self) -> None:
        self.set_cars([0, 0, 1, 0])
        self.assertEqual(self.evaluate([1, 0, 0, 0])["new_demand_revenue"], 0)

    def test_committed_revenue_inclusive_proration_and_capacity(self) -> None:
        self.set_cars([2, 0, 0, 0])
        self.instance.reservations = pd.DataFrame([
            ("R1", "open", "economy", "S1", "S1", pd.Timestamp("2027-02-28"), pd.Timestamp("2027-03-02"), 300),
            ("R2", "booked", "economy", "S1", "S1", pd.Timestamp("2027-03-03"), pd.Timestamp("2027-03-04"), 200),
        ], columns=self.instance.reservations.columns)
        self.instance.fleet_snapshot.loc[0, ["status", "station_id", "reservation_id"]] = ["on_rent", None, "R1"]
        self.instance.stations["max_utilization"] = .5
        result = self.evaluate([1, 0, 0, 0])
        self.assertEqual(result["committed_revenue"], 200 + 100)
        self.assertEqual(result["new_demand_revenue"], 0)  # .5 * 2 - 1 = 0 capacity

    def test_transit_holding_and_driver_cost(self) -> None:
        self.set_cars([2, 0, 0, 0])
        for day in ["2027-03-01", "2027-03-03"]:
            result = self.evaluate(plan=pd.DataFrame([("transfer", day, "economy", "S1", "S2", "driver", 2)], columns=PLAN_COLUMNS))
            self.assertEqual(result["holding_cost"], 2 * 3 * 10)
            self.assertEqual(result["transfer_cost"], 2 * (5 + 2 * 10))
            self.assertEqual(result["profit"], -110)

    def test_trucks_share_across_categories_and_round_up(self) -> None:
        self.set_cars([5, 4, 0, 0])
        for compact, trucks in [(3, 1), (4, 2)]:
            plan = pd.DataFrame([
                ("transfer", "2027-03-01", "economy", "S1", "S2", "truck", 5),
                ("transfer", "2027-03-01", "compact", "S1", "S2", "truck", compact),
            ], columns=PLAN_COLUMNS)
            self.assertEqual(self.evaluate(plan=plan)["transfer_cost"], trucks * (100 + 3 * 10))

    def test_trucks_on_different_days_do_not_share(self) -> None:
        self.set_cars([2, 0, 0, 0])
        plan = pd.DataFrame([
            ("transfer", day, "economy", "S1", "S2", "truck", 1)
            for day in ["2027-03-01", "2027-03-02"]
        ], columns=PLAN_COLUMNS)
        self.assertEqual(self.evaluate(plan=plan)["transfer_cost"], 260)

    def test_additions_charged_once_and_removals_reduce_holding(self) -> None:
        self.set_cars([1, 0, 0, 0])
        plan = pd.DataFrame([
            ("add", "2027-03-02", "economy", "S1", None, None, 1),
            ("remove", "2027-03-03", "economy", "S1", None, None, 1),
        ], columns=PLAN_COLUMNS)
        result = self.evaluate(plan=plan)
        self.assertEqual(result["addition_cost"], 100)
        self.assertEqual(result["holding_cost"], (1 + 2 + 1) * 10)

    def test_invalid_plan_and_stale_fleet_rejected(self) -> None:
        plan = pd.DataFrame([("remove", "2027-03-01", "economy", "S1", None, None, 1)], columns=PLAN_COLUMNS)
        with self.assertRaisesRegex(ValueError, "Invalid plan"):
            self.evaluate(plan=plan)
        stale = simulate_fleet(instance=self.instance, plan=pd.DataFrame())
        stale["station_fleet"] = 1
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.evaluate(fleet=stale)


if __name__ == "__main__":
    unittest.main()
