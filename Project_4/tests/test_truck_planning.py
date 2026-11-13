"""Manual Level 3 group-cost, timing, validity and upgrade cases."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_evaluate as fixtures  # noqa: E402
from solution.evaluate import estimate_profit  # noqa: E402
from solution.fleet import PLAN_COLUMNS, simulate_fleet, transit_cars, check_plan_validity  # noqa: E402
from solution.planning import search_level3  # noqa: E402


class TruckPlanningTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvaluatorTests()
        self.fixture.setUp()
        self.instance = self.fixture.instance
        self.instance.distances["lead_time_days"] = 1
        self.plan = pd.DataFrame(columns=PLAN_COLUMNS)

    def inputs(self):
        grid = self.instance.stations[["station_id"]].merge(self.instance.categories[["category"]], how="cross")
        base = grid.copy()
        base["base_revenue_per_car_day"] = base.category.map({"economy": 100., "compact": 200., "midsize": 300., "premium": 400.})
        demand = grid.merge(pd.DataFrame({"date": pd.date_range("2027-03-01", "2027-03-03")}), how="cross")
        demand["demand_to_come"] = 0.
        return demand, base

    def search(self, demand=None, **settings):
        zero, base = self.inputs()
        return search_level3(instance=self.instance, plan=self.plan, demand=zero if demand is None else demand,
                             base_revenue=base, time_limit_seconds=settings.pop("time_limit_seconds", 20),
                             max_rounds=1, max_cars_per_move=9, **settings)

    def evaluate(self, plan):
        demand, base = self.inputs()
        return estimate_profit(instance=self.instance, plan=plan, demand=demand, base_revenue=base)

    def test_group_conversion_shares_across_categories(self):
        self.fixture.set_cars([2, 2, 0, 0])
        self.instance.distances["lead_time_days"] = 2
        self.instance.transfer_costs.loc[self.instance.transfer_costs["mode"].eq("driver"), ["fixed_cost", "cost_per_km"]] = [50., 0.]
        self.plan = pd.DataFrame([
            ("transfer", pd.Timestamp("2027-03-01"), category, "S1", "S2", "driver", 2)
            for category in ["economy", "compact"]
        ], columns=PLAN_COLUMNS)
        # Four bookings require every transferred car at S2 on day 3. Deleting
        # or delaying either transfer is invalid. Each individual conversion
        # costs 130 instead of 100, but the shared group costs 130 instead of 200.
        self.instance.reservations = pd.DataFrame([
            (f"B{category}{i}", "booked", category, "S2", "S2", pd.Timestamp("2027-03-03"), pd.Timestamp("2027-03-03"), 0.)
            for category in ["economy", "compact"] for i in range(2)
        ], columns=self.instance.reservations.columns)
        result = self.search()
        self.assertTrue(result.plan["mode"].eq("truck").all())
        self.assertEqual(result.totals["transfer_cost"], 130)
        self.assertEqual(result.statistics["move_trace"][0]["kind"], "group_mode")
        self.assertEqual(result.statistics["move_trace"][0]["local_profit_gain"], 70)
        self.assertEqual(result.totals["holding_cost"], (2*10 + 2*20)*3)
        self.assertEqual(check_plan_validity(instance=self.instance, plan=result.plan), [])

    def test_eight_vs_nine_boundary_across_categories(self):
        self.fixture.set_cars([8, 1, 0, 0])
        for compact, expected in [(0, 130), (1, 260)]:
            rows = [("transfer", pd.Timestamp("2027-03-01"), "economy", "S1", "S2", "truck", 8)]
            if compact:
                rows.append(("transfer", pd.Timestamp("2027-03-01"), "compact", "S1", "S2", "truck", compact))
            self.assertEqual(self.evaluate(pd.DataFrame(rows, columns=PLAN_COLUMNS))["transfer_cost"], expected)

    def test_different_routes_and_dates_cost_separately(self):
        self.fixture.set_cars([3, 0, 0, 0])
        self.instance.stations = pd.concat([self.instance.stations, pd.DataFrame([{"station_id": "S3", "station_type": "downtown", "max_utilization": 1.}])], ignore_index=True)
        self.instance.distances = pd.concat([self.instance.distances, pd.DataFrame([{"from_station_id": "S1", "to_station_id": "S3", "distance_km": 10., "lead_time_days": 1}])], ignore_index=True)
        plan = pd.DataFrame([
            ("transfer", pd.Timestamp(date), "economy", "S1", destination, "truck", 1)
            for destination, date in [("S2", "2027-03-01"), ("S3", "2027-03-01"), ("S2", "2027-03-02")]
        ], columns=PLAN_COLUMNS)
        self.assertEqual(self.evaluate(plan)["transfer_cost"], 390)

    def test_arrival_and_transit_holding(self):
        self.fixture.set_cars([1, 0, 0, 0])
        self.instance.distances["lead_time_days"] = 2
        plan = pd.DataFrame([("transfer", pd.Timestamp("2027-03-01"), "economy", "S1", "S2", "truck", 1)], columns=PLAN_COLUMNS)
        fleet = simulate_fleet(instance=self.instance, plan=plan).set_index(["station_id", "category", "date"])
        self.assertEqual(fleet.loc[("S2", "economy", pd.Timestamp("2027-03-02")), "station_fleet"], 0)
        self.assertEqual(fleet.loc[("S2", "economy", pd.Timestamp("2027-03-03")), "station_fleet"], 1)
        self.assertEqual(transit_cars(instance=self.instance, plan=plan).in_transit.sum(), 2)
        self.assertEqual(self.evaluate(plan)["holding_cost"], 30)

    def test_future_origin_shortage_rejects_transfer(self):
        self.fixture.set_cars([1, 0, 0, 0])
        self.instance.reservations = pd.DataFrame([
            ("B1", "booked", "economy", "S1", "S1", pd.Timestamp("2027-03-03"), pd.Timestamp("2027-03-03"), 0.)
        ], columns=self.instance.reservations.columns)
        demand, _ = self.inputs()
        demand.loc[demand.station_id.eq("S2") & demand.category.eq("economy"), "demand_to_come"] = 10.
        result = self.search(demand)
        pd.testing.assert_frame_equal(result.plan, self.plan)
        self.assertGreater(result.statistics["invalid_by_type"]["add"], 0)

    def test_upgrade_changes_transfer_profitability(self):
        self.fixture.set_cars([0, 1, 0, 0])
        unchanged = self.search()
        pd.testing.assert_frame_equal(unchanged.plan, self.plan)
        demand, _ = self.inputs()
        demand.loc[demand.station_id.eq("S2") & demand.category.eq("economy"), "demand_to_come"] = 1.
        result = self.search(demand)
        self.assertEqual(result.plan.category.tolist(), ["compact"])
        self.assertEqual(result.plan["mode"].tolist(), ["driver"])
        self.assertEqual(result.totals["new_demand_revenue"], 200)  # two days, requested economy rate
        self.assertEqual(result.totals["profit"], 200 - 60 - 25)

    def test_no_gain_and_zero_budget_retain_exact_fallback(self):
        self.fixture.set_cars([1, 0, 0, 0])
        before = self.plan.copy(deep=True)
        result = self.search()
        pd.testing.assert_frame_equal(result.plan, before)
        self.assertEqual(result.totals, result.references["saved_level2"]["totals"])
        pd.testing.assert_frame_equal(self.search(time_limit_seconds=0).plan, before)
        pd.testing.assert_frame_equal(self.plan, before)

    def test_nonempty_level2_fallback_is_unchanged_when_transfer_is_required(self):
        self.fixture.set_cars([1, 0, 0, 0])
        self.plan = pd.DataFrame([
            ("transfer", pd.Timestamp("2027-03-01"), "economy", "S1", "S2", "driver", 1)
        ], columns=PLAN_COLUMNS, index=[7])
        self.instance.reservations = pd.DataFrame([
            ("B2", "booked", "economy", "S2", "S2", pd.Timestamp("2027-03-02"), pd.Timestamp("2027-03-03"), 0.)
        ], columns=self.instance.reservations.columns)
        before = self.plan.copy(deep=True)
        result = self.search()
        pd.testing.assert_frame_equal(result.plan, before)
        self.assertEqual(result.statistics["accepted_moves"], 0)
        self.assertEqual(result.totals["transfer_cost"], 25)


if __name__ == "__main__":
    unittest.main()
