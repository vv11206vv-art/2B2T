"""Small, manually understandable Level 1 search and cache checks."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from solution.evaluate import estimate_profit  # noqa: E402
from solution.fleet import PLAN_COLUMNS, committed_car_days, simulate_fleet, validate_plan  # noqa: E402
from solution.planning import _Level1Cache, _moves, search_level1  # noqa: E402


class Level1Tests(unittest.TestCase):
    def setUp(self):
        categories = ["economy", "compact", "midsize", "premium"]
        self.instance = SimpleNamespace(
            horizon=pd.DataFrame({"first_date": pd.to_datetime(["2027-03-01"]), "last_date": pd.to_datetime(["2027-03-03"])}),
            stations=pd.DataFrame({"station_id": ["S1", "S2"], "station_type": ["downtown"]*2, "max_utilization": [1., 1.]}),
            categories=pd.DataFrame({"category": categories, "rank": [1, 2, 3, 4], "holding_cost_per_day": [10]*4, "addition_cost_per_car": [100]*4, "addition_cap": [1, 0, 0, 0], "removal_cap": [1, 0, 0, 0]}),
            fleet_snapshot=pd.DataFrame(columns=["vehicle_id", "category", "status", "station_id", "reservation_id"]),
            reservations=pd.DataFrame(columns=["reservation_id", "category", "pickup_station_id", "return_station_id", "pickup_date", "return_date", "revenue", "status"]),
            distances=pd.DataFrame({"from_station_id": ["S1", "S2"], "to_station_id": ["S2", "S1"], "lead_time_days": [1, 1], "distance_km": [1., 1.]}),
            transfer_costs=pd.DataFrame({"mode": ["driver", "truck"], "fixed_cost": [1., 1.], "cost_per_km": [1., 1.]}),
            price_factors=pd.DataFrame([("downtown", k, q, 1.) for k in categories for q in [.6, .7, .8, .9, 1.]], columns=["station_type", "category", "served_share", "price_factor"]),
        )
        for column in ["pickup_date", "return_date"]:
            self.instance.reservations[column] = pd.to_datetime(self.instance.reservations[column])
        self.committed = committed_car_days(instance=self.instance)
        self.demand = self.committed[["station_id", "category", "date"]].copy()
        self.demand["demand_to_come"] = 0.
        self.demand.loc[self.demand.station_id.eq("S1") & self.demand.category.eq("economy"), "demand_to_come"] = 1.
        self.base = self.demand[["station_id", "category"]].drop_duplicates().copy()
        self.base["base_revenue_per_car_day"] = 100.

    def search(self, **settings):
        return search_level1(instance=self.instance, committed=self.committed, demand=self.demand,
                             base_revenue=self.base, time_limit_seconds=settings.pop("time_limit_seconds", 30), max_rounds=5, **settings)

    def test_profitable_addition(self):
        result = self.search()
        self.assertAlmostEqual(result.totals["profit"], 3*100 - 3*10 - 100)
        self.assertEqual(result.plan.action.tolist(), ["add"])
        self.assertEqual(result.plan.cars.sum(), 1)
        validate_plan(instance=self.instance, plan=result.plan)

    def test_unprofitable_addition(self):
        self.base["base_revenue_per_car_day"] = 10.
        result = self.search()
        self.assertTrue(result.plan.empty)
        self.assertEqual(result.totals["profit"], 0)

    def test_future_commitment_blocks_early_removal(self):
        self.instance.fleet_snapshot = pd.DataFrame([("V1", "economy", "idle", "S1", None)], columns=self.instance.fleet_snapshot.columns)
        self.instance.reservations = pd.DataFrame([("R1", "economy", "S1", "S1", pd.Timestamp("2027-03-03"), pd.Timestamp("2027-03-03"), 100., "booked")], columns=self.instance.reservations.columns)
        self.instance.categories["addition_cap"] = 0
        self.committed = committed_car_days(instance=self.instance)
        self.demand["demand_to_come"] = 0.
        result = self.search()
        self.assertTrue(result.plan.empty)
        self.assertEqual(result.totals["profit"], 100 - 30)

    def test_category_cap_is_enforced_across_network(self):
        self.demand.loc[self.demand.category.eq("economy"), "demand_to_come"] = 10.
        result = self.search()
        self.assertEqual(result.plan.loc[result.plan.action.eq("add"), "cars"].sum(), 1)
        self.assertEqual(result.plan.columns.tolist(), PLAN_COLUMNS)

    def test_zero_budget_retains_valid_reference(self):
        result = self.search(time_limit_seconds=0)
        self.assertEqual(result.totals["profit"], max(r["totals"]["profit"] for r in result.references.values() if r["valid"]))
        self.assertEqual(result.statistics["accepted_moves"], 0)
        validate_plan(instance=self.instance, plan=result.plan)

    def test_full_cap_neighborhood_contains_station_and_date_reallocation(self):
        counts = np.zeros((2, 2, 4, 3), dtype=int)
        counts[0, 0, 0, 0] = 1
        moves = list(_moves(counts))
        self.assertIn(((0, 0, 0, 0, -1), (0, 1, 0, 2, 1)), moves)
        self.assertIn(((0, 0, 0, 0, -1), (0, 0, 0, 2, 1)), moves)

    def test_paired_change_escapes_full_addition_cap(self):
        self.demand.loc[self.demand.category.eq("economy"), "demand_to_come"] = 1.
        self.base.loc[self.base.station_id.eq("S2"), "base_revenue_per_car_day"] = 200.
        reference = pd.DataFrame([("add", pd.Timestamp("2027-03-01"), "economy", "S1", None, None, 1)], columns=PLAN_COLUMNS)
        # Reference earns 170. Cancelling it loses profit; adding at S2 alone
        # exceeds the cap. Reallocating its one car to S2 earns 470.
        with patch("solution.planning.baseline_plan", return_value=reference):
            result = self.search()
        self.assertEqual(result.statistics["best_reference"], "baseline")
        self.assertAlmostEqual(result.totals["profit"], 470)
        self.assertEqual(result.plan.station_id.tolist(), ["S2"])

    def test_cached_score_matches_exact_with_upgrades_and_removal(self):
        self.instance.categories["addition_cap"] = 10
        self.instance.fleet_snapshot = pd.DataFrame([("V1", "economy", "idle", "S2", None)], columns=self.instance.fleet_snapshot.columns)
        cache = _Level1Cache(self.instance, self.committed, self.demand, self.base)
        plan = pd.DataFrame([
            ("add", pd.Timestamp("2027-03-01"), "compact", "S1", None, None, 1),
            ("remove", pd.Timestamp("2027-03-02"), "economy", "S2", None, None, 1),
        ], columns=PLAN_COLUMNS)
        counts = cache.counts(plan)
        fleets = cache.fleet(counts)
        cached = sum(cache.station_score(s, fleets[s]) for s in range(2)) - 100
        exact = estimate_profit(instance=self.instance, plan=plan, demand=self.demand, base_revenue=self.base)
        self.assertAlmostEqual(cached, exact["profit"])
        self.assertAlmostEqual(cached, 300 - 30 - 10 - 100)


if __name__ == "__main__":
    unittest.main()
