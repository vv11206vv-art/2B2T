"""Manually worked driver transfers using the evaluator's small fixture."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import test_evaluate as fixtures  # noqa: E402
from solution.evaluate import estimate_profit  # noqa: E402
from solution.fleet import PLAN_COLUMNS, simulate_fleet, transit_cars, check_plan_validity  # noqa: E402
from solution.planning import search_driver_transfers  # noqa: E402


class DriverPlanningTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EvaluatorTests()
        self.fixture.setUp()
        self.fixture.set_cars([1, 0, 0, 0])
        self.instance = self.fixture.instance
        self.instance.distances["lead_time_days"] = 1
        self.plan = pd.DataFrame(columns=PLAN_COLUMNS)
        self.demand = simulate_fleet(instance=self.instance, plan=self.plan)[["station_id", "category", "date"]]
        self.demand["demand_to_come"] = 0.
        self.demand.loc[self.demand.station_id.eq("S2") & self.demand.category.eq("economy"), "demand_to_come"] = 1.
        self.base = self.demand[["station_id", "category"]].drop_duplicates().copy()
        self.base["base_revenue_per_car_day"] = 100.

    def search(self, **settings):
        return search_driver_transfers(instance=self.instance, plan=self.plan, demand=self.demand,
                                       base_revenue=self.base, time_limit_seconds=settings.pop("time_limit_seconds", 20),
                                       max_rounds=2, max_cars_per_transfer=2, **settings)

    def test_one_and_two_day_arrivals_include_transit_holding_and_driver_cost(self):
        for lead, revenue in [(1, 200), (2, 100)]:
            with self.subTest(lead=lead):
                self.instance.distances["lead_time_days"] = lead
                result = self.search()
                self.assertEqual(result.totals["new_demand_revenue"], revenue)
                self.assertEqual(result.totals["holding_cost"], 30)  # one car held all three days
                self.assertEqual(result.totals["transfer_cost"], 25)  # 5 + 2 * 10 km
                self.assertEqual(result.totals["profit"], revenue - 30 - 25)
                move = result.statistics["move_trace"][0]
                self.assertEqual(move["arrival"], f"2027-03-0{1 + lead}")
                self.assertEqual(transit_cars(instance=self.instance, plan=result.plan).in_transit.sum(), lead)
                self.assertEqual(check_plan_validity(instance=self.instance, plan=result.plan), [])

    def test_future_origin_booking_blocks_transfer(self):
        self.instance.reservations = pd.DataFrame([
            ("B1", "booked", "economy", "S1", "S1", pd.Timestamp("2027-03-03"), pd.Timestamp("2027-03-03"), 100.)
        ], columns=self.instance.reservations.columns)
        result = self.search()
        pd.testing.assert_frame_equal(result.plan, self.plan)
        self.assertEqual(result.totals["profit"], 100 - 30)
        self.assertEqual(result.statistics["accepted_moves"], 0)

    def test_destination_commitment_consumes_own_capacity(self):
        extra = self.instance.fleet_snapshot.iloc[:1].copy()
        extra["vehicle_id"] = "destination_car"
        extra["station_id"] = "S2"
        self.instance.fleet_snapshot = pd.concat([self.instance.fleet_snapshot, extra], ignore_index=True)
        self.instance.reservations = pd.DataFrame([
            ("B2", "booked", "economy", "S2", "S2", pd.Timestamp("2027-03-02"), pd.Timestamp("2027-03-03"), 0.)
        ], columns=self.instance.reservations.columns)
        result = self.search()
        self.assertEqual(result.totals["new_demand_revenue"], 300)
        self.assertEqual(result.totals["holding_cost"], 60)
        self.assertEqual(result.totals["profit"], 300 - 60 - 25)

    def test_destination_benefit_does_not_justify_larger_origin_loss(self):
        self.demand.loc[self.demand.station_id.eq("S1") & self.demand.category.eq("economy"), "demand_to_come"] = 1.
        self.base.loc[self.base.station_id.eq("S1"), "base_revenue_per_car_day"] = 200.
        trial = pd.DataFrame([("transfer", pd.Timestamp("2027-03-01"), "economy", "S1", "S2", "driver", 1)], columns=PLAN_COLUMNS)
        trial_totals = estimate_profit(instance=self.instance, plan=trial, demand=self.demand, base_revenue=self.base)
        self.assertEqual(trial_totals["new_demand_revenue"], 200)  # destination benefits on two days
        self.assertEqual(trial_totals["profit"], 200 - 30 - 25)
        result = self.search()
        self.assertEqual(result.totals["profit"], 600 - 30)
        pd.testing.assert_frame_equal(result.plan, self.plan)

    def test_driver_cost_can_eliminate_all_destination_gain(self):
        self.instance.transfer_costs.loc[self.instance.transfer_costs["mode"].eq("driver"), "fixed_cost"] = 300.
        result = self.search()
        pd.testing.assert_frame_equal(result.plan, self.plan)
        self.assertEqual(result.totals["profit"], -30)

    def test_two_car_quantity_charges_per_car(self):
        self.fixture.set_cars([2, 0, 0, 0])
        self.demand.loc[self.demand.station_id.eq("S2") & self.demand.category.eq("economy"), "demand_to_come"] = 2.
        result = self.search()
        self.assertEqual(result.plan.cars.sum(), 2)
        self.assertEqual(result.totals["profit"], 400 - 60 - 50)

    def test_after_horizon_arrivals_and_zero_budget_keep_fallback(self):
        before = self.plan.copy(deep=True)
        self.instance.distances["lead_time_days"] = 3
        pd.testing.assert_frame_equal(self.search().plan, before)
        self.instance.distances["lead_time_days"] = 1
        pd.testing.assert_frame_equal(self.search(time_limit_seconds=0).plan, before)
        pd.testing.assert_frame_equal(self.plan, before)

    def test_invalid_fallback_is_rejected(self):
        self.plan = pd.DataFrame([("remove", pd.Timestamp("2027-03-01"), "economy", "S1", None, None, 2)], columns=PLAN_COLUMNS)
        with self.assertRaisesRegex(ValueError, "Invalid plan"):
            self.search()


if __name__ == "__main__":
    unittest.main()
