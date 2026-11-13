"""Local profit estimates under explicit demand and price inputs (R6-R11)."""

import numpy as np
import pandas as pd

from fleetkit import Instance
from solution.fleet import committed_car_days, simulate_fleet, transit_cars, validate_plan

KEY = ["station_id", "category"]
DAY_KEY = [*KEY, "date"]


def revenue_for_served(*, demand: float, served: float, base: float, factors: pd.DataFrame) -> float:
    """Interpolate total revenue, including the line from zero to 60% served.

    Factor points remain in served-share order even if marginal revenue rises.
    No concavity correction or reordering of revenue buckets is applied.
    """
    if demand == 0:
        return 0.0
    points = factors.sort_values("served_share")
    quantities = points["served_share"].to_numpy() * demand
    revenues = quantities * base * points["price_factor"].to_numpy()
    return float(np.interp(served, np.r_[0.0, quantities], np.r_[0.0, revenues]))


def _align(grid: pd.DataFrame, values: pd.DataFrame, keys: list[str], column: str) -> pd.DataFrame:
    """Require exactly one finite nonnegative input value per grid cell."""
    table = values[keys + [column]].copy()
    if "date" in keys:
        table["date"] = pd.to_datetime(table["date"])
    result = grid.merge(table, on=keys, how="outer", validate="one_to_one", indicator=True)
    if not result["_merge"].eq("both").all():
        raise ValueError(f"{column} must cover exactly the complete grid")
    numbers = pd.to_numeric(result[column], errors="raise")
    if not (np.isfinite(numbers) & numbers.ge(0)).all():
        raise ValueError(f"{column} must contain finite nonnegative values")
    result[column] = numbers
    return result.drop(columns="_merge")


def estimate_profit(
    *,
    instance: Instance,
    plan: pd.DataFrame,
    fleet: pd.DataFrame | None = None,
    demand: pd.DataFrame,
    base_revenue: pd.DataFrame,
) -> dict[str, float]:
    """Validate a plan and return six locally estimated monetary totals.

    ``demand`` is the complete station/category/date demand-to-come grid;
    ``base_revenue`` gives a rate for every station/category pair. Optional
    ``fleet`` must match the plan simulation. Invalid plans/inputs raise
    ValueError. These estimates are not official demand-scenario scores.
    """
    validate_plan(instance=instance, plan=plan)
    simulated = simulate_fleet(instance=instance, plan=plan)
    if fleet is not None:
        supplied = _align(simulated[DAY_KEY], fleet, DAY_KEY, "station_fleet")
        check = simulated.merge(supplied, on=DAY_KEY, suffixes=("_actual", "_supplied"), validate="one_to_one")
        if not check["station_fleet_actual"].eq(check["station_fleet_supplied"]).all():
            raise ValueError("Supplied fleet does not match the proposed plan")
    daily = _align(simulated, demand, DAY_KEY, "demand_to_come")
    pairs = simulated[KEY].drop_duplicates()
    prices = _align(pairs, base_revenue, KEY, "base_revenue_per_car_day")
    daily = daily.merge(prices, on=KEY, validate="many_to_one")
    daily = daily.merge(committed_car_days(instance=instance), on=DAY_KEY, validate="one_to_one")
    daily = daily.merge(instance.stations[["station_id", "station_type", "max_utilization"]], on="station_id", validate="many_to_one")
    ranks = instance.categories.set_index("category")["rank"]
    daily["rank"] = daily["category"].map(ranks)
    daily["capacity"] = (daily["max_utilization"] * daily["station_fleet"] - daily["committed_car_days"]).clip(lower=0)
    factor_sets = {key: rows for key, rows in instance.price_factors.groupby(["station_type", "category"])}
    new_revenue = 0.0
    for _, group in daily.groupby(["station_id", "date"], sort=False):
        group = group.sort_values("rank")
        demand_values = group["demand_to_come"].to_numpy()
        capacity = group["capacity"].to_numpy()
        own = np.minimum(demand_values, capacity)
        served = own.copy()
        # Each higher category keeps own service first; only its unused cars
        # can serve unmet demand of the immediately lower category.
        served[:-1] += np.minimum(capacity[1:] - own[1:], demand_values[:-1] - own[:-1])
        for row, quantity in zip(group.itertuples(index=False), served):
            new_revenue += revenue_for_served(
                demand=row.demand_to_come, served=quantity, base=row.base_revenue_per_car_day,
                factors=factor_sets[(row.station_type, row.category)],
            )

    first = instance.horizon["first_date"].iloc[0]
    last = instance.horizon["last_date"].iloc[0]
    rentals = instance.reservations.loc[instance.reservations["status"].isin(["open", "booked"])]
    lengths = (rentals["return_date"] - rentals["pickup_date"]).dt.days + 1
    inside = (rentals["return_date"].clip(upper=last) - rentals["pickup_date"].clip(lower=first)).dt.days + 1
    committed_revenue = float((rentals["revenue"] * inside.clip(lower=0) / lengths).sum())

    categories = instance.categories.set_index("category")
    holding = float((daily["station_fleet"] * daily["category"].map(categories["holding_cost_per_day"])).sum())
    transit = transit_cars(instance=instance, plan=plan)
    holding += float((transit["in_transit"] * transit["category"].map(categories["holding_cost_per_day"])).sum())
    addition_cost = 0.0
    transfer_cost = 0.0
    if not plan.empty:
        additions = plan.loc[plan["action"].eq("add")]
        addition_cost = float((additions["cars"] * additions["category"].map(categories["addition_cost_per_car"])).sum())
        transfers = plan.loc[plan["action"].eq("transfer")].copy()
        transfers["date"] = pd.to_datetime(transfers["date"])
        routes = instance.distances.rename(columns={"from_station_id": "station_id"})
        transfers = transfers.merge(routes[["station_id", "to_station_id", "distance_km"]], on=["station_id", "to_station_id"], validate="many_to_one")
        costs = instance.transfer_costs.set_index("mode")
        drivers = transfers.loc[transfers["mode"].eq("driver")]
        transfer_cost += float((drivers["cars"] * (costs.loc["driver", "fixed_cost"] + costs.loc["driver", "cost_per_km"] * drivers["distance_km"])).sum())
        trucks = transfers.loc[transfers["mode"].eq("truck")].groupby(["station_id", "to_station_id", "date", "distance_km"], as_index=False)["cars"].sum()
        transfer_cost += float((np.ceil(trucks["cars"] / 8) * (costs.loc["truck", "fixed_cost"] + costs.loc["truck", "cost_per_km"] * trucks["distance_km"])).sum())
    return {
        "committed_revenue": committed_revenue,
        "new_demand_revenue": float(new_revenue),
        "holding_cost": holding,
        "addition_cost": addition_cost,
        "transfer_cost": transfer_cost,
        "profit": committed_revenue + new_revenue - holding - addition_cost - transfer_cost,
    }
