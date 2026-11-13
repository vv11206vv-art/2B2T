"""Fleet accounting and plan validity (README Appendix A, R1-R5)."""

import numbers
from pathlib import Path

import pandas as pd

from fleetkit import Instance, load_instance

KEY = ["station_id", "category"]
DAY_KEY = [*KEY, "date"]
PLAN_COLUMNS = ["action", "date", "category", "station_id", "to_station_id", "mode", "cars"]


def _grid(instance: Instance) -> pd.DataFrame:
    """Every station, category and horizon date, even if its count is zero."""
    first = instance.horizon["first_date"].iloc[0]
    last = instance.horizon["last_date"].iloc[0]
    dates = pd.DataFrame({"date": pd.date_range(first, last)})
    return (
        instance.stations[["station_id"]]
        .merge(instance.categories[["category"]], how="cross")
        .merge(dates, how="cross")
    )


def _committed(instance: Instance) -> pd.DataFrame:
    return instance.reservations.loc[instance.reservations["status"].isin(["open", "booked"])]


def _fill_counts(grid: pd.DataFrame, counts: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    result = grid.merge(counts, on=keys, how="left", validate="one_to_one")
    result[columns] = result[columns].fillna(0).astype("int64")
    return result


def initial_fleet(*, instance: Instance) -> pd.DataFrame:
    """Count each snapshot car once at its idle or rental pickup station.

    Return ``station_id, category, cars`` for all station/category pairs.
    """
    snapshot = instance.fleet_snapshot
    on_rent = snapshot.loc[snapshot["status"].eq("on_rent")]
    broken = on_rent["reservation_id"].isna() | ~on_rent["reservation_id"].isin(instance.reservations["reservation_id"])
    if broken.any():
        details = on_rent.loc[broken, ["vehicle_id", "reservation_id"]].to_dict("records")
        raise ValueError(f"On-rent snapshot cars have missing or unknown reservation links: {details}")
    idle = snapshot.loc[snapshot["status"].eq("idle"), KEY]
    rented = snapshot.loc[snapshot["status"].eq("on_rent"), ["reservation_id", "category"]]
    rented = rented.merge(
        instance.reservations[["reservation_id", "pickup_station_id"]],
        on="reservation_id",
        validate="many_to_one",
    ).rename(columns={"pickup_station_id": "station_id"})[KEY]
    counts = (
        pd.concat([idle, rented], ignore_index=True)
        .groupby(KEY, as_index=False)
        .size()
        .rename(columns={"size": "cars"})
    )
    result = _fill_counts(_grid(instance)[KEY].drop_duplicates(), counts, KEY, ["cars"])
    if result["cars"].sum() != len(snapshot):
        raise ValueError(
            f"Initial fleet count mismatch: counted {result['cars'].sum()} cars "
            f"from {len(snapshot)} snapshot cars"
        )
    return result


def committed_car_days(*, instance: Instance) -> pd.DataFrame:
    """Count open/booked rentals on each day from pickup through return, inclusive."""
    grid = _grid(instance)
    rentals = _committed(instance)
    days = rentals.merge(grid[["date"]].drop_duplicates(), how="cross")
    days = days.loc[days["date"].between(days["pickup_date"], days["return_date"])]
    counts = (
        days.groupby(["pickup_station_id", "category", "date"], as_index=False)
        .size()
        .rename(columns={"pickup_station_id": "station_id", "size": "committed_car_days"})
    )
    return _fill_counts(grid, counts, DAY_KEY, ["committed_car_days"])


def oneway_flows(*, instance: Instance) -> pd.DataFrame:
    """Move one-way cars on the day after return, if inside the horizon."""
    grid = _grid(instance)
    rentals = _committed(instance)
    one_way = rentals.loc[rentals["pickup_station_id"].ne(rentals["return_station_id"])].copy()
    one_way["date"] = one_way["return_date"] + pd.Timedelta(days=1)
    one_way = one_way.loc[one_way["date"].between(grid["date"].min(), grid["date"].max())]
    departures = (
        one_way.groupby(["pickup_station_id", "category", "date"], as_index=False)
        .size()
        .rename(columns={"pickup_station_id": "station_id", "size": "oneway_departures"})
    )
    arrivals = (
        one_way.groupby(["return_station_id", "category", "date"], as_index=False)
        .size()
        .rename(columns={"return_station_id": "station_id", "size": "oneway_arrivals"})
    )
    return _fill_counts(
        grid.merge(arrivals, on=DAY_KEY, how="left", validate="one_to_one"),
        departures,
        DAY_KEY,
        ["oneway_arrivals", "oneway_departures"],
    )


def _prepare_plan(instance: Instance, plan: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Normalize action rows and collect format/route violations before simulation."""
    if plan.empty and len(plan.columns) == 0:
        return [], []  # Existing do-nothing callers use a bare empty DataFrame.
    missing = [column for column in PLAN_COLUMNS if column not in plan.columns]
    if missing:
        return [], [{"type": "missing_fields", "message": f"Missing required columns: {missing}"}]
    first = pd.Timestamp(instance.horizon["first_date"].iloc[0])
    last = pd.Timestamp(instance.horizon["last_date"].iloc[0])
    stations = set(instance.stations["station_id"])
    categories = set(instance.categories["category"])
    modes = set(instance.transfer_costs["mode"])
    routes = instance.distances.set_index(["from_station_id", "to_station_id"])["lead_time_days"].to_dict()
    actions, violations = [], []
    for position, (_, row) in enumerate(plan.iterrows(), start=1):
        errors = []
        action = row["action"]
        category = row["category"]
        origin = row["station_id"]
        destination = row["to_station_id"]
        mode = row["mode"]
        cars = row["cars"]
        if action not in ("add", "remove", "transfer"):
            errors.append("action must be add, remove or transfer")
        if category not in categories:
            errors.append("unknown category")
        if origin not in stations:
            errors.append("unknown station_id")
        if not isinstance(cars, numbers.Integral) or isinstance(cars, bool) or cars < 1:
            errors.append("cars must be a positive integer")
        try:
            date = pd.Timestamp(row["date"])
            if pd.isna(date) or date.time() != pd.Timestamp(0).time() or not first <= date <= last:
                errors.append("date must be a horizon date")
        except (TypeError, ValueError):
            date = pd.NaT
            errors.append("date must be a horizon date")
        if action == "transfer":
            if destination not in stations:
                errors.append("unknown to_station_id")
            if destination == origin:
                errors.append("transfer destination must differ from origin")
            if mode not in modes:
                errors.append("unknown transfer mode")
            if origin in stations and destination in stations and origin != destination and (origin, destination) not in routes:
                errors.append("transfer route has no lead time")
        elif action in ("add", "remove"):
            if pd.notna(destination) or pd.notna(mode):
                errors.append("addition/removal must have empty to_station_id and mode")
        if errors:
            violations.append({"type": "action_row", "row": position,
                               "action_row": row[PLAN_COLUMNS].to_dict(),
                               "message": "; ".join(errors)})
        else:
            actions.append({"action": action, "date": date, "category": category,
                            "station_id": origin, "to_station_id": destination, "mode": mode,
                            "cars": int(cars), "lead_time_days": int(routes[(origin, destination)]) if action == "transfer" else 0})
    return actions, violations


def _fleet_from_actions(instance: Instance, actions: list[dict]) -> pd.DataFrame:
    """Apply daily deltas to the verified do-nothing fleet."""
    flows = oneway_flows(instance=instance)
    start = initial_fleet(instance=instance)
    daily = flows.merge(start, on=KEY, validate="many_to_one")
    daily["net"] = daily["oneway_arrivals"] - daily["oneway_departures"]
    last = daily["date"].max()
    deltas = []
    for row in actions:
        action, date, category, origin, cars = (row[key] for key in ("action", "date", "category", "station_id", "cars"))
        sign = 1 if action == "add" else -1
        deltas.append((origin, category, date, sign * cars))
        if action == "transfer":
            arrival = date + pd.Timedelta(days=row["lead_time_days"])
            if arrival <= last:
                deltas.append((row["to_station_id"], category, arrival, cars))
    if deltas:
        effects = pd.DataFrame(deltas, columns=DAY_KEY + ["effect"])
        effects = effects.groupby(DAY_KEY, as_index=False)["effect"].sum()
        daily = daily.merge(effects, on=DAY_KEY, how="left", validate="one_to_one")
        daily["net"] += daily["effect"].fillna(0).astype("int64")
    daily["station_fleet"] = daily["cars"] + daily.groupby(KEY, sort=False)["net"].cumsum()
    return daily[DAY_KEY + ["station_fleet"]]


def simulate_fleet(*, instance: Instance, plan: pd.DataFrame) -> pd.DataFrame:
    """Return every station/category/date fleet cell after all valid-format actions.

    This computes even an infeasible plan; use ``check_plan_validity`` to test
    commitments and caps. Badly formatted action rows raise ValueError.
    """
    actions, errors = _prepare_plan(instance, plan)
    if errors:
        raise ValueError(f"Invalid plan format: {errors}")
    return _fleet_from_actions(instance, actions)


def transit_cars(*, instance: Instance, plan: pd.DataFrame) -> pd.DataFrame:
    """Return ``category, date, in_transit`` for every category and horizon day.

    Cars count from departure through the day before arrival, including a
    transfer whose arrival lies beyond the horizon.
    """
    actions, errors = _prepare_plan(instance, plan)
    if errors:
        raise ValueError(f"Invalid plan format: {errors}")
    dates = pd.DataFrame({"date": pd.date_range(instance.horizon["first_date"].iloc[0], instance.horizon["last_date"].iloc[0])})
    grid = instance.categories[["category"]].merge(dates, how="cross")
    last = grid["date"].max()
    events = []
    for row in actions:
        if row["action"] == "transfer":
            arrival = row["date"] + pd.Timedelta(days=row["lead_time_days"])
            events.append((row["category"], row["date"], row["cars"]))
            if arrival <= last:
                events.append((row["category"], arrival, -row["cars"]))
    if events:
        changes = pd.DataFrame(events, columns=["category", "date", "change"])
        changes = changes.groupby(["category", "date"], as_index=False)["change"].sum()
        grid = grid.merge(changes, on=["category", "date"], how="left", validate="one_to_one")
        grid["in_transit"] = grid["change"].fillna(0).astype("int64").groupby(grid["category"], sort=False).cumsum()
    else:
        grid["in_transit"] = 0
    return grid[["category", "date", "in_transit"]]


def check_plan_validity(*, instance: Instance, plan: pd.DataFrame) -> list[dict]:
    """Return action, category-cap and daily committed-fleet violations.

    Row numbers are one-based plan data rows. If action rows are malformed,
    fleet shortages are skipped because their fleet effects are undefined.
    """
    actions, violations = _prepare_plan(instance, plan)
    if violations:
        return violations
    for category_row in instance.categories.itertuples(index=False):
        category = category_row.category
        for action, cap in (("add", category_row.addition_cap), ("remove", category_row.removal_cap)):
            total = sum(row["cars"] for row in actions if row["category"] == category and row["action"] == action)
            if total > cap:
                violations.append({"type": "category_cap", "category": category, "action": action,
                                   "total": total, "cap": int(cap), "excess": total - int(cap)})
    fleet = _fleet_from_actions(instance, actions)
    daily = fleet.merge(committed_car_days(instance=instance), on=DAY_KEY, validate="one_to_one")
    for row in daily.loc[daily["station_fleet"] < daily["committed_car_days"]].itertuples(index=False):
        violations.append({"type": "fleet_shortage", "station_id": row.station_id,
                           "category": row.category, "date": row.date,
                           "station_fleet": int(row.station_fleet),
                           "committed_car_days": int(row.committed_car_days),
                           "shortage": int(row.committed_car_days - row.station_fleet)})
    return violations


def compare_check_values(*, instance: Instance) -> pd.DataFrame:
    """Return every supplied Level 0 mismatch with expected/calculated values."""
    actual = (
        committed_car_days(instance=instance)
        .merge(oneway_flows(instance=instance), on=DAY_KEY, validate="one_to_one")
        .merge(simulate_fleet(instance=instance, plan=pd.DataFrame()), on=DAY_KEY, validate="one_to_one")
    )
    checked = instance.check_values.merge(actual, on=DAY_KEY, how="left", suffixes=("_expected", "_calculated"), validate="one_to_one")
    mismatches = []
    for row in checked.to_dict("records"):
        for metric in ("committed_car_days", "oneway_arrivals", "oneway_departures", "station_fleet"):
            expected = row[f"{metric}_expected"]
            calculated = row[f"{metric}_calculated"]
            if pd.isna(calculated) or expected != calculated:
                mismatches.append({**{key: row[key] for key in DAY_KEY}, "metric": metric, "expected": expected, "calculated": calculated})
    return pd.DataFrame(mismatches, columns=DAY_KEY + ["metric", "expected", "calculated"])


def validate_plan(*, instance: Instance, plan: pd.DataFrame) -> None:
    """Raise ValueError containing violations; return None for a valid plan."""
    violations = check_plan_validity(instance=instance, plan=plan)
    if violations:
        raise ValueError(f"Invalid plan: {violations}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    data = load_instance(data_dir=project_dir / "data")
    differences = compare_check_values(instance=data)
    if not differences.empty:
        print(differences.to_string(index=False))
        raise SystemExit(1)
    print(f"All {len(data.check_values)} check-value rows match across all four metrics.")
