"""Bounded Level 1 and driver-transfer searches; no global optimality guarantee."""

from dataclasses import dataclass
from collections import Counter
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


def search_driver_transfers(
    *, instance: Instance, plan: pd.DataFrame, demand: pd.DataFrame,
    base_revenue: pd.DataFrame, time_limit_seconds: float = 60,
    max_rounds: int = 100, max_cars_per_transfer: int = 8,
) -> SearchResult:
    """Append improving driver transfers to an exact, valid Level 1 fallback.

    Search every supplied route, category and departure whose arrival is inside
    the horizon, with quantities 1..max_cars_per_transfer (also capped by all
    future origin commitments). This quantity bound is a search setting, not a
    driver-capacity rule. Repeated moves can send more cars. Add/remove rows
    are fixed. Every accepted move is checked with the existing exact evaluator;
    unsuccessful or exhausted searches retain the unchanged input plan.
    """
    if not np.isfinite(time_limit_seconds) or time_limit_seconds < 0:
        raise ValueError("Time limit must be finite and nonnegative")
    if not isinstance(max_rounds, int) or max_rounds < 0 or not isinstance(max_cars_per_transfer, int) or max_cars_per_transfer < 1:
        raise ValueError("max_rounds must be nonnegative and max_cars_per_transfer positive integers")
    started = perf_counter()
    fallback = plan.copy(deep=True)
    if not set(fallback["action"]) <= {"add", "remove"}:
        raise ValueError("Driver search must start from an add/remove Level 1 plan")
    exact = estimate_profit(instance=instance, plan=fallback, demand=demand, base_revenue=base_revenue)
    reference_seconds = perf_counter() - started
    references = {"saved_level1": {"valid": True, "totals": exact.copy(), "evaluation_seconds": reference_seconds}}
    committed = committed_car_days(instance=instance)
    cache = _Level1Cache(instance, committed, demand, base_revenue)
    fleet = cache.fleet(cache.counts(fallback))
    scores = np.array([cache.station_score(s, fleet[s]) for s in range(len(cache.stations))])
    driver = instance.transfer_costs.set_index("mode").loc["driver"]
    routes = sorted(
        (cache.stations.index(row.from_station_id), cache.stations.index(row.to_station_id),
         int(row.lead_time_days), float(driver.fixed_cost + driver.cost_per_km * row.distance_km))
        for row in instance.distances.itertuples(index=False)
        if row.from_station_id != row.to_station_id
    )
    deadline = started + max(0, time_limit_seconds - max(3.0, 3 * reference_seconds))
    current = fallback.copy(deep=True)
    trace = []
    screened = 0
    exact_rejections = 0
    stop_reason = "round_limit"
    for _ in range(max_rounds):
        if perf_counter() >= deadline:
            stop_reason = "time_limit"
            break
        best_gain, best = 1e-7, None
        timed_out = False
        for origin, destination, lead, unit_cost in routes:
            if timed_out:
                break
            for day in range(len(cache.days) - lead):
                if timed_out:
                    break
                arrival = day + lead
                for category in range(len(cache.categories)):
                    slack = fleet[origin, day:, category] - cache.committed[origin, day:, category]
                    bound = min(max_cars_per_transfer, int(np.floor(slack.min())))
                    for cars in range(1, bound + 1):
                        if perf_counter() >= deadline:
                            timed_out = True
                            stop_reason = "time_limit"
                            break
                        screened += 1
                        departing = fleet[origin].copy()
                        arriving = fleet[destination].copy()
                        departing[day:, category] -= cars
                        arriving[arrival:, category] += cars
                        if np.any(departing < cache.committed[origin]) or np.any(arriving < cache.committed[destination]):
                            continue
                        # station_score includes station holding costs. Restore
                        # holding cost for the cars absent from both stations
                        # during transit, so each car is charged once daily.
                        transit_holding = cars * lead * cache.holding[category]
                        gain = (cache.station_score(origin, departing) - scores[origin]
                                + cache.station_score(destination, arriving) - scores[destination]
                                - transit_holding - cars * unit_cost)
                        if gain > best_gain:
                            best_gain = float(gain)
                            best = (origin, destination, category, day, arrival, cars, unit_cost)
                    if timed_out:
                        break
        if best is None:
            if not timed_out:
                stop_reason = "no_improving_transfer"
            break
        origin, destination, category, day, arrival, cars, unit_cost = best
        row = {"action": "transfer", "date": cache.days[day], "category": cache.categories[category],
               "station_id": cache.stations[origin], "to_station_id": cache.stations[destination],
               "mode": "driver", "cars": cars}
        trial = pd.concat([current, pd.DataFrame([row], columns=PLAN_COLUMNS)], ignore_index=True)
        try:
            checked = estimate_profit(instance=instance, plan=trial, demand=demand, base_revenue=base_revenue)
        except ValueError:
            exact_rejections += 1
            stop_reason = "exact_validation_rejected_move"
            break
        improvement = checked["profit"] - exact["profit"]
        if not np.isclose(improvement, best_gain, rtol=0, atol=1e-6):
            raise RuntimeError("Cached transfer gain disagrees with the exact evaluator")
        if improvement <= 1e-7:
            stop_reason = "no_exact_improvement"
            break
        current, exact = trial, checked
        fleet[origin, day:, category] -= cars
        fleet[destination, arrival:, category] += cars
        scores[origin] = cache.station_score(origin, fleet[origin])
        scores[destination] = cache.station_score(destination, fleet[destination])
        trace.append({**row, "date": row["date"].strftime("%Y-%m-%d"),
                      "arrival": cache.days[arrival].strftime("%Y-%m-%d"),
                      "driver_cost": float(cars * unit_cost), "local_profit_gain": float(improvement)})
        if timed_out:
            break
    validate_plan(instance=instance, plan=current)
    final = estimate_profit(instance=instance, plan=current, demand=demand, base_revenue=base_revenue)
    if final["profit"] < references["saved_level1"]["totals"]["profit"]:
        current, final = fallback, references["saved_level1"]["totals"].copy()
    return SearchResult(current, references, final, {
        "time_limit_seconds": time_limit_seconds, "max_rounds": max_rounds,
        "max_cars_per_transfer": max_cars_per_transfer,
        "runtime_seconds": perf_counter() - started, "best_reference": "saved_level1",
        "candidates_screened": screened, "accepted_moves": len(trace), "move_trace": trace,
        "exact_rejections": exact_rejections, "stop_reason": stop_reason,
        "search_scope": "fixed Level 1 actions; appended driver transfers arriving within horizon",
    })


def add_driver_transfers(
    *, instance: Instance, plan: pd.DataFrame, demand: pd.DataFrame | None = None,
    base_revenue: pd.DataFrame | None = None, time_limit_seconds: float = 60,
    max_rounds: int = 100, max_cars_per_transfer: int = 8,
) -> pd.DataFrame:
    """Return the verified driver-transfer plan, or unchanged valid Level 1 input."""
    if demand is None:
        committed = committed_car_days(instance=instance)
        demand = demand_to_come(instance=instance, committed=committed)
    if base_revenue is None:
        base_revenue = estimate_base_revenue(instance=instance)
    return search_driver_transfers(instance=instance, plan=plan, demand=demand, base_revenue=base_revenue,
                                   time_limit_seconds=time_limit_seconds, max_rounds=max_rounds,
                                   max_cars_per_transfer=max_cars_per_transfer).plan


def _level3_moves(counts, routes, categories, days, max_cars):
    """Stable neighborhood; transfer keys are origin, destination, day, category, mode.

    Groups change mode together across categories. Other edits change one
    aggregate transfer: cancel, resize, change mode/origin/destination/date.
    New transfers try 1, 8, 9 and the configured maximum (within that maximum).
    """
    active = sorted(key for key, cars in counts.items() if cars > 0)
    for origin, destination, day in sorted({key[:3] for key in active}):
        for mode in ("driver", "truck"):
            members = [key for key in active if key[:3] == (origin, destination, day) and key[4] == mode]
            edits = []
            for key in members:
                edits += [(key, -counts[key]), (key[:4] + ("truck" if mode == "driver" else "driver",), counts[key])]
            if edits:
                yield "group_mode", edits
    for key in active:
        origin, destination, day, category, mode = key
        cars = counts[key]
        other = key[:4] + ("truck" if mode == "driver" else "driver",)
        yield "mode", [(key, -cars), (other, cars)]
        yield "remove", [(key, -cars)]
        for delta in sorted({-8, -1, 1, 8}):
            if cars + delta > 0 and abs(delta) <= max_cars:
                yield "resize", [(key, delta)]
        for new_day in range(days):
            if new_day != day:
                yield "date", [(key, -cars), ((origin, destination, new_day, category, mode), cars)]
        for new_origin, new_destination in sorted(routes):
            if new_destination == destination and new_origin != origin:
                yield "origin", [(key, -cars), ((new_origin, destination, day, category, mode), cars)]
            if new_origin == origin and new_destination != destination:
                yield "destination", [(key, -cars), ((origin, new_destination, day, category, mode), cars)]
    quantities = sorted({q for q in (1, 8, 9, max_cars) if q <= max_cars})
    for origin, destination in sorted(routes):
        for day in range(days):
            for category in range(categories):
                for mode in ("driver", "truck"):
                    for cars in quantities:
                        yield "add", [((origin, destination, day, category, mode), cars)]


def search_level3(
    *, instance: Instance, plan: pd.DataFrame, demand: pd.DataFrame,
    base_revenue: pd.DataFrame, time_limit_seconds: float = 120,
    max_rounds: int = 100, max_cars_per_move: int = 16,
) -> SearchResult:
    """Best-improvement transfer edits around a fixed Level 2 add/remove plan.

    All candidates receive a whole-plan estimated profit, using cached fixed
    inputs with full-category station scoring and all-category truck grouping.
    Field validity and unchanged add/remove caps follow by construction; daily
    coverage is checked at every affected station. Accepted improvements and
    final selection must pass the existing exact evaluator and full checker.
    This finite neighborhood and wall-clock/round limits do not prove optimality.
    """
    if not np.isfinite(time_limit_seconds) or time_limit_seconds < 0:
        raise ValueError("Time limit must be finite and nonnegative")
    if not isinstance(max_rounds, int) or max_rounds < 0 or not isinstance(max_cars_per_move, int) or max_cars_per_move < 1:
        raise ValueError("Round limit must be nonnegative and move quantity limit positive integers")
    started = perf_counter()
    fallback = plan.copy(deep=True)
    exact = estimate_profit(instance=instance, plan=fallback, demand=demand, base_revenue=base_revenue)
    reference_seconds = perf_counter() - started
    reference = exact.copy()
    cache = _Level1Cache(instance, committed_car_days(instance=instance), demand, base_revenue)
    index = pd.MultiIndex.from_product([cache.stations, cache.days, cache.categories], names=["station_id", "date", "category"])
    fleet = simulate_fleet(instance=instance, plan=fallback).set_index(index.names).station_fleet.reindex(index).to_numpy(dtype=float).reshape(cache.start.shape)
    scores = np.array([cache.station_score(s, fleet[s]) for s in range(len(cache.stations))])
    routes = {(cache.stations.index(row.from_station_id), cache.stations.index(row.to_station_id)): (int(row.lead_time_days), float(row.distance_km)) for row in instance.distances.itertuples(index=False)}
    costs = instance.transfer_costs.set_index("mode")
    unit_cost = {(o, d, mode): float(costs.loc[mode, "fixed_cost"] + costs.loc[mode, "cost_per_km"] * distance)
                 for (o, d), (_, distance) in routes.items() for mode in ("driver", "truck")}
    fixed = fallback.loc[~fallback.action.eq("transfer")].copy()
    counts = Counter()
    for row in fallback.loc[fallback.action.eq("transfer")].itertuples(index=False):
        counts[(cache.stations.index(row.station_id), cache.stations.index(row.to_station_id), cache.days.get_loc(pd.Timestamp(row.date)), cache.categories.index(row.category), row.mode)] += int(row.cars)

    def transfer_cost(state):
        trucks = Counter()
        total = 0.0
        for (o, d, day, k, mode), cars in state.items():
            if mode == "driver":
                total += cars * unit_cost[o, d, mode]
            else:
                trucks[o, d, day] += cars  # Deliberately omit category.
        return total + sum(((cars + 7) // 8) * unit_cost[o, d, "truck"] for (o, d, day), cars in trucks.items())

    def as_plan(state):
        rows = [("transfer", cache.days[day], cache.categories[k], cache.stations[o], cache.stations[d], mode, cars)
                for (o, d, day, k, mode), cars in sorted(state.items()) if cars > 0]
        return pd.concat([fixed, pd.DataFrame(rows, columns=PLAN_COLUMNS)], ignore_index=True)

    old_transfer_cost = transfer_cost(counts)
    if not np.isclose(old_transfer_cost, exact["transfer_cost"], rtol=0, atol=1e-6):
        raise RuntimeError("Cached grouped transfer cost disagrees with exact fallback")
    deadline = started + max(0, time_limit_seconds - max(3.0, 3 * reference_seconds))
    screened, invalid, trace = Counter(), Counter(), []
    current = fallback.copy(deep=True)
    stop_reason = "round_limit"
    for _ in range(max_rounds):
        if perf_counter() >= deadline:
            stop_reason = "time_limit"
            break
        best_gain, best = 1e-7, None
        timed_out = False
        for kind, edits in _level3_moves(counts, routes, len(cache.categories), len(cache.days), max_cars_per_move):
            if perf_counter() >= deadline:
                stop_reason, timed_out = "time_limit", True
                break
            screened[kind] += 1
            state = counts.copy()
            changed = {}
            transit_delta = 0.0
            for (o, d, day, k, mode), delta in edits:
                key = (o, d, day, k, mode)
                state[key] += delta
                if state[key] < 0:
                    break
                if not state[key]:
                    del state[key]
                lead = routes[o, d][0]
                arrival = day + lead
                if o not in changed:
                    changed[o] = fleet[o].copy()
                if d not in changed:
                    changed[d] = fleet[d].copy()
                changed[o][day:, k] -= delta
                changed[d][arrival:, k] += delta
                transit_delta += delta * min(lead, len(cache.days) - day) * cache.holding[k]
            else:
                if any(np.any(values < cache.committed[s]) for s, values in changed.items()):
                    invalid[kind] += 1
                    continue
                new_cost = transfer_cost(state)
                gain = sum(cache.station_score(s, values) - scores[s] for s, values in changed.items()) - transit_delta - new_cost + old_transfer_cost
                if gain > best_gain:
                    best_gain = float(gain)
                    best = (kind, edits, state, changed, new_cost)
                continue
            invalid[kind] += 1
        if best is None:
            if not timed_out:
                stop_reason = "local_optimum"
            break
        kind, edits, state, changed, new_cost = best
        trial = as_plan(state)
        checked = estimate_profit(instance=instance, plan=trial, demand=demand, base_revenue=base_revenue)
        improvement = checked["profit"] - exact["profit"]
        if not np.isclose(improvement, best_gain, rtol=0, atol=1e-6):
            raise RuntimeError("Cached whole-plan Level 3 gain disagrees with exact evaluator")
        if improvement <= 1e-7:
            stop_reason = "no_exact_improvement"
            break
        current, exact, counts, old_transfer_cost = trial, checked, state, new_cost
        for s, values in changed.items():
            fleet[s] = values
            scores[s] = cache.station_score(s, values)
        trace.append({"kind": kind, "local_profit_gain": float(improvement), "local_profit_after": float(exact["profit"]),
                      "edits": [{"station_id": cache.stations[o], "to_station_id": cache.stations[d], "date": cache.days[day].strftime("%Y-%m-%d"),
                                 "category": cache.categories[k], "mode": mode, "cars_delta": int(delta)} for (o, d, day, k, mode), delta in edits]})
        if timed_out:
            break
    validate_plan(instance=instance, plan=current)
    final = estimate_profit(instance=instance, plan=current, demand=demand, base_revenue=base_revenue)
    if final["profit"] < reference["profit"]:
        current, final = fallback, reference.copy()
    return SearchResult(current, {"saved_level2": {"valid": True, "totals": reference, "evaluation_seconds": reference_seconds}}, final, {
        "time_limit_seconds": time_limit_seconds, "max_rounds": max_rounds, "max_cars_per_move": max_cars_per_move,
        "runtime_seconds": perf_counter() - started, "best_reference": "saved_level2", "stop_reason": stop_reason,
        "candidates_screened": sum(screened.values()), "screened_by_type": dict(screened), "invalid_by_type": dict(invalid),
        "accepted_moves": len(trace), "move_trace": trace,
        "search_scope": "fixed additions/removals; transfer group modes, mode, remove, resize, date, origin, destination, add",
    })


def improve_with_trucks(
    *, instance: Instance, plan: pd.DataFrame, demand: pd.DataFrame | None = None,
    base_revenue: pd.DataFrame | None = None, time_limit_seconds: float = 120,
    max_rounds: int = 100, max_cars_per_move: int = 16,
) -> pd.DataFrame:
    """Return a locally verified Level 3 candidate or the unchanged fallback."""
    if demand is None:
        demand = demand_to_come(instance=instance, committed=committed_car_days(instance=instance))
    if base_revenue is None:
        base_revenue = estimate_base_revenue(instance=instance)
    return search_level3(instance=instance, plan=plan, demand=demand, base_revenue=base_revenue,
                         time_limit_seconds=time_limit_seconds, max_rounds=max_rounds,
                         max_cars_per_move=max_cars_per_move).plan


def build_plan(*, instance: Instance, time_limit_seconds: float = 180, max_rounds: int = 500) -> pd.DataFrame:
    """Connect fleet, forecast, planning and validation into a final plan."""
    committed = committed_car_days(instance=instance)
    return level1_plan(instance=instance, committed=committed,
                       demand=demand_to_come(instance=instance, committed=committed),
                       base_revenue=estimate_base_revenue(instance=instance),
                       time_limit_seconds=time_limit_seconds, max_rounds=max_rounds)
