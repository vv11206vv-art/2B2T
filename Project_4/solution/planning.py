"""Bounded deterministic Level 1 local search; no global optimality guarantee."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from fleetkit import Instance, baseline_plan
from solution.evaluate import estimate_profit
from solution.fleet import PLAN_COLUMNS, committed_car_days, simulate_fleet, validate_plan
from solution.forecast import demand_to_come, base_revenue as estimate_base_revenue


@dataclass
class SearchResult:
    """Selected official-format plan, exact reference totals and search audit."""

    plan: pd.DataFrame
    references: dict
    totals: dict[str, float]
    statistics: dict


class _Level1Cache:
    """Cache fixed inputs and score only stations affected by each move.

    Counts have shape (action, station, category, day), action 0=add, 1=remove.
    Fleet and occupancy have shape (station, day, category). Every candidate
    checks whole-network caps and every daily commitment. Revenue is evaluated
    jointly across categories with the same R6-R8 equations as estimate_profit.
    """

    def __init__(self, instance, committed, demand, base_revenue):
        self.instance = instance
        self.stations = instance.stations.station_id.tolist()
        self.categories = instance.categories.sort_values("rank").category.tolist()
        self.days = pd.date_range(instance.horizon.first_date.iloc[0], instance.horizon.last_date.iloc[0])
        index = pd.MultiIndex.from_product([self.stations, self.days, self.categories], names=["station_id", "date", "category"])
        shape = (len(self.stations), len(self.days), len(self.categories))

        def array(table, value):
            return table.set_index(index.names)[value].reindex(index).to_numpy(dtype=float).reshape(shape)

        self.start = array(simulate_fleet(instance=instance, plan=pd.DataFrame()), "station_fleet")
        self.committed = array(committed, "committed_car_days")
        self.demand = array(demand, "demand_to_come")
        pairs = pd.MultiIndex.from_product([self.stations, self.categories], names=["station_id", "category"])
        self.base = base_revenue.set_index(pairs.names).base_revenue_per_car_day.reindex(pairs).to_numpy().reshape(len(self.stations), -1)
        categories = instance.categories.set_index("category").loc[self.categories]
        self.holding = categories.holding_cost_per_day.to_numpy()
        self.addition = categories.addition_cost_per_car.to_numpy()
        self.caps = np.stack([categories.addition_cap.to_numpy(), categories.removal_cap.to_numpy()])
        stations = instance.stations.set_index("station_id").loc[self.stations]
        self.utilization = stations.max_utilization.to_numpy()
        self.prices = {}
        for s, station_type in enumerate(stations.station_type):
            for k, category in enumerate(self.categories):
                factors = instance.price_factors.loc[instance.price_factors.station_type.eq(station_type) & instance.price_factors.category.eq(category)].sort_values("served_share")
                self.prices[s, k] = (np.r_[0, factors.served_share.to_numpy()], np.r_[0, (factors.served_share * factors.price_factor).to_numpy()])

    def counts(self, plan):
        result = np.zeros((2, len(self.stations), len(self.categories), len(self.days)), dtype=int)
        for row in plan.itertuples(index=False):
            if row.action not in ("add", "remove"):
                raise ValueError("Level 1 references must contain only additions/removals")
            result[0 if row.action == "add" else 1, self.stations.index(row.station_id), self.categories.index(row.category), self.days.get_loc(pd.Timestamp(row.date))] += row.cars
        return result

    def plan(self, counts):
        rows = [("add" if a == 0 else "remove", self.days[d], self.categories[k], self.stations[s], None, None, int(counts[a, s, k, d])) for a, s, k, d in zip(*np.nonzero(counts))]
        return pd.DataFrame(rows, columns=PLAN_COLUMNS)

    def fleet(self, counts):
        effects = np.cumsum(counts[0] - counts[1], axis=2).transpose(0, 2, 1)
        return self.start + effects

    def station_score(self, station, fleet):
        capacity = np.maximum(0, self.utilization[station] * fleet - self.committed[station])
        demand = self.demand[station]
        own = np.minimum(demand, capacity)
        served = own.copy()
        served[:, :-1] += np.minimum(capacity[:, 1:] - own[:, 1:], demand[:, :-1] - own[:, :-1])
        revenue = 0.0
        for k in range(len(self.categories)):
            shares, totals = self.prices[station, k]
            q = np.divide(served[:, k], demand[:, k], out=np.zeros(len(self.days)), where=demand[:, k] > 0)
            revenue += float((np.interp(q, shares, totals) * demand[:, k] * self.base[station, k]).sum())
        return revenue - float((fleet * self.holding).sum())


def _moves(counts):
    """Stable exhaustive neighborhood: units, cancellations, dates, reallocations."""
    _, stations, categories, days = counts.shape
    totals = counts.sum(axis=(1, 3))
    # Primitive moves precede paired moves so even a short run sees all types.
    for a in range(2):
        for s in range(stations):
            for k in range(categories):
                for d in range(days):
                    yield ((a, s, k, d, 1),)
                    if counts[a, s, k, d]:
                        yield ((a, s, k, d, -1),)
    for a, s, k, d in zip(*np.nonzero(counts)):
        for target_day in range(days):
            if target_day != d:
                yield ((a, s, k, d, -1), (a, s, k, target_day, 1))
        if a == 0 and totals[a, k]:
            for target_station in range(stations):
                if target_station != s:
                    for target_day in range(days):
                        yield ((a, s, k, d, -1), (a, target_station, k, target_day, 1))


def search_level1(*, instance: Instance, committed: pd.DataFrame, demand: pd.DataFrame,
                  base_revenue: pd.DataFrame, time_limit_seconds: float = 180,
                  max_rounds: int = 500) -> SearchResult:
    """Keep the best exact-valid reference or verified local-search improvement.

    Stable move order and strict improvement make the search deterministic for
    a fixed number of completed rounds. A wall-clock cutoff may interrupt a
    round; the accepted move trace is recorded for audit. The time limit covers
    search setup/references and reserves time for final exact evaluation.
    """
    if not np.isfinite(time_limit_seconds) or time_limit_seconds < 0 or max_rounds < 0:
        raise ValueError("Search limits must be finite and nonnegative")
    started = perf_counter()
    references = {}
    reference_plans = {"do_nothing": pd.DataFrame(columns=PLAN_COLUMNS), "baseline": baseline_plan(instance=instance)}
    for name, plan in reference_plans.items():
        try:
            t = perf_counter()
            totals = estimate_profit(instance=instance, plan=plan, demand=demand, base_revenue=base_revenue)
            references[name] = {"valid": True, "totals": totals, "evaluation_seconds": perf_counter() - t}
        except ValueError as error:
            references[name] = {"valid": False, "error": str(error)}
    valid = [name for name in references if references[name]["valid"]]
    if not valid:
        raise ValueError(f"No valid reference plan: {references}")
    best_reference = max(valid, key=lambda name: references[name]["totals"]["profit"])
    fallback = reference_plans[best_reference].copy()
    exact_fallback = references[best_reference]["totals"]
    cache = _Level1Cache(instance, committed, demand, base_revenue)
    counts = cache.counts(fallback)
    fleet = cache.fleet(counts)
    scores = np.array([cache.station_score(s, fleet[s]) for s in range(len(cache.stations))])
    addition_cost = float((counts[0].sum(axis=(0, 2)) * cache.addition).sum())
    constant = exact_fallback["committed_revenue"]
    cached_profit = constant + scores.sum() - addition_cost
    if not np.isclose(cached_profit, exact_fallback["profit"], rtol=0, atol=1e-6):
        raise RuntimeError("Cached score disagrees with the exact reference evaluator")
    benchmark_started = perf_counter()
    for _ in range(100):
        cache.station_score(0, fleet[0])
    station_score_seconds = (perf_counter() - benchmark_started) / 100
    reserve = max(3.0, 3 * max(references[name]["evaluation_seconds"] for name in valid))
    deadline = started + max(0, time_limit_seconds - reserve)
    candidates = 0
    trace = []
    stop_reason = "round_limit"
    for _ in range(max_rounds):
        if perf_counter() >= deadline:
            stop_reason = "time_limit"
            break
        best_gain, best_move = 1e-7, None
        totals = counts.sum(axis=(1, 3))
        for move in _moves(counts):
            if perf_counter() >= deadline:
                stop_reason = "time_limit"
                break
            candidates += 1
            cap_changes = np.zeros_like(totals)
            changed_fleets = {}
            add_cost_delta = 0.0
            for a, s, k, d, delta in move:
                cap_changes[a, k] += delta
                changed_fleets.setdefault(s, fleet[s].copy())
                changed_fleets[s][d:, k] += delta if a == 0 else -delta
                if a == 0:
                    add_cost_delta += delta * cache.addition[k]
            if np.any(totals + cap_changes > cache.caps) or any(np.any(f < cache.committed[s]) for s, f in changed_fleets.items()):
                continue
            gain = -add_cost_delta + sum(cache.station_score(s, f) - scores[s] for s, f in changed_fleets.items())
            if gain > best_gain:
                best_gain, best_move = gain, move
        if best_move is None:
            if stop_reason != "time_limit":
                stop_reason = "local_optimum"
            break
        for a, s, k, d, delta in best_move:
            counts[a, s, k, d] += delta
        fleet = cache.fleet(counts)
        scores = np.array([cache.station_score(s, fleet[s]) for s in range(len(cache.stations))])
        trace.append([[int(v) for v in item] for item in best_move])
        if stop_reason == "time_limit":
            break
    candidate = cache.plan(counts)
    exact = estimate_profit(instance=instance, plan=candidate, demand=demand, base_revenue=base_revenue)
    predicted = constant + scores.sum() - float((counts[0].sum(axis=(0, 2)) * cache.addition).sum())
    if not np.isclose(predicted, exact["profit"], rtol=0, atol=1e-6):
        raise RuntimeError("Cached search score disagrees with the exact final evaluator")
    if exact["profit"] < exact_fallback["profit"]:
        candidate, exact = fallback, exact_fallback
    validate_plan(instance=instance, plan=candidate)
    return SearchResult(candidate, references, exact, {
        "time_limit_seconds": time_limit_seconds, "max_rounds": max_rounds,
        "runtime_seconds": perf_counter() - started, "best_reference": best_reference,
        "candidates_screened": candidates, "accepted_moves": len(trace),
        "stop_reason": stop_reason, "move_trace": trace,
        "cached_station_score_seconds": station_score_seconds,
        "cached_exact_profit_difference": float(predicted - exact["profit"]),
    })


def level1_plan(
    *, instance: Instance, committed: pd.DataFrame, demand: pd.DataFrame, base_revenue: pd.DataFrame,
    time_limit_seconds: float = 180, max_rounds: int = 500,
) -> pd.DataFrame:
    """Build a per-category additions/removals plan without transfers."""
    return search_level1(instance=instance, committed=committed, demand=demand, base_revenue=base_revenue,
                         time_limit_seconds=time_limit_seconds, max_rounds=max_rounds).plan


def add_driver_transfers(*, instance: Instance, plan: pd.DataFrame) -> pd.DataFrame:
    """Return a plan with feasible driver transfers added (later Level 2 work)."""
    raise NotImplementedError("Driver transfer planning is not implemented")


def improve_with_trucks(*, instance: Instance, plan: pd.DataFrame) -> pd.DataFrame:
    """Return a plan with truck-capacity improvements (later Level 3 work)."""
    raise NotImplementedError("Truck planning is not implemented")


def build_plan(*, instance: Instance, time_limit_seconds: float = 180, max_rounds: int = 500) -> pd.DataFrame:
    """Connect fleet, forecast, planning and validation into a final plan."""
    committed = committed_car_days(instance=instance)
    return level1_plan(instance=instance, committed=committed,
                       demand=demand_to_come(instance=instance, committed=committed),
                       base_revenue=estimate_base_revenue(instance=instance),
                       time_limit_seconds=time_limit_seconds, max_rounds=max_rounds)
