"""Simple Level 0 estimates from completed rentals in the prior eight weeks."""

import numpy as np
import pandas as pd

from fleetkit import Instance

KEY = ["station_id", "category"]
DAY_KEY = [*KEY, "date"]


def _completed(instance: Instance) -> pd.DataFrame:
    return instance.reservations.loc[instance.reservations["status"].eq("completed")]


def _station_categories(instance: Instance) -> pd.DataFrame:
    return instance.stations[["station_id"]].merge(instance.categories[["category"]], how="cross")


def _validate_commitments(committed: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Require a complete, unique grid with finite nonnegative commitments."""
    required = DAY_KEY + ["committed_car_days"]
    missing_columns = [column for column in required if column not in committed.columns]
    if missing_columns:
        raise ValueError(f"Commitments missing required columns: {missing_columns}")
    if committed[required].isna().any().any():
        raise ValueError("Commitments contain null keys or committed_car_days values")
    duplicates = committed.duplicated(DAY_KEY, keep=False)
    if duplicates.any():
        raise ValueError(f"Commitments contain duplicate keys: {committed.loc[duplicates, DAY_KEY].to_dict('records')[:5]}")
    actual_keys = pd.MultiIndex.from_frame(committed[DAY_KEY])
    expected_keys = pd.MultiIndex.from_frame(expected[DAY_KEY])
    missing = expected_keys.difference(actual_keys)
    extra = actual_keys.difference(expected_keys)
    if len(missing) or len(extra):
        raise ValueError(f"Commitments key mismatch: missing {len(missing)} keys {list(missing[:5])}; extra {len(extra)} keys {list(extra[:5])}")
    values = committed["committed_car_days"]
    if not pd.api.types.is_numeric_dtype(values):
        raise ValueError("Commitments committed_car_days must be numeric")
    invalid = ~np.isfinite(values) | values.lt(0)
    if invalid.any():
        raise ValueError(f"Commitments must be finite and nonnegative: {committed.loc[invalid, required].to_dict('records')[:5]}")


def demand_to_come(*, instance: Instance, committed: pd.DataFrame) -> pd.DataFrame:
    """Forecast weekday occupancy minus committed car-days, floored at zero.

    Historical means cover all 56 calendar dates, including zero-occupancy
    cells. Completed rentals occupy both their pickup and return dates.
    ``committed`` must be the full output of ``fleet.committed_car_days``.
    """
    first = instance.horizon["first_date"].iloc[0]
    last = instance.horizon["last_date"].iloc[0]
    future_dates = pd.DataFrame({"date": pd.date_range(first, last)})
    future = _station_categories(instance).merge(future_dates, how="cross")
    _validate_commitments(committed, future)
    historical_dates = pd.DataFrame({"date": pd.date_range(first - pd.Timedelta(days=56), periods=56)})
    history_grid = _station_categories(instance).merge(historical_dates, how="cross")
    rentals = _completed(instance)
    occupied = rentals.merge(historical_dates, how="cross")
    occupied = occupied.loc[occupied["date"].between(occupied["pickup_date"], occupied["return_date"])]
    counts = (
        occupied.groupby(["pickup_station_id", "category", "date"], as_index=False)
        .size()
        .rename(columns={"pickup_station_id": "station_id", "size": "cars_on_rent"})
    )
    history = history_grid.merge(counts, on=DAY_KEY, how="left", validate="one_to_one")
    history["cars_on_rent"] = history["cars_on_rent"].fillna(0)
    history["weekday"] = history["date"].dt.day_of_week
    weekday_means = (
        history.groupby(KEY + ["weekday"], as_index=False)["cars_on_rent"]
        .mean()
        .rename(columns={"cars_on_rent": "mean_occupancy"})
    )

    future["weekday"] = future["date"].dt.day_of_week
    future = future.merge(weekday_means, on=KEY + ["weekday"], validate="many_to_one")
    future = future.merge(committed[DAY_KEY + ["committed_car_days"]], on=DAY_KEY, validate="one_to_one")
    future["demand_to_come"] = (future["mean_occupancy"] - future["committed_car_days"]).clip(lower=0)
    return future[DAY_KEY + ["demand_to_come"]]


def base_revenue(*, instance: Instance) -> pd.DataFrame:
    """Return revenue divided by inclusive rental days for every pair.

    Missing station/category history falls back to the category-wide weighted
    rate, then the overall weighted rate if that category has no history.
    ``estimate_source`` identifies the level used. With no completed rentals
    anywhere, no data-backed rate exists and this function raises ValueError.
    """
    rentals = _completed(instance).copy()
    if rentals.empty:
        raise ValueError("Cannot estimate base revenue: no completed rentals")
    rentals["rental_days"] = (rentals["return_date"] - rentals["pickup_date"]).dt.days + 1
    if (rentals["rental_days"] <= 0).any():
        raise ValueError("Completed rental has nonpositive inclusive length")

    def rates(group_cols: list[str], rate_col: str) -> pd.DataFrame:
        totals = rentals.groupby(group_cols, as_index=False).agg(
            revenue=("revenue", "sum"), rental_days=("rental_days", "sum")
        )
        totals[rate_col] = totals["revenue"] / totals["rental_days"]
        return totals[group_cols + [rate_col]]

    pair_rates = rates(["pickup_station_id", "category"], "pair_rate").rename(
        columns={"pickup_station_id": "station_id"}
    )
    category_rates = rates(["category"], "category_rate")
    overall_rate = rentals["revenue"].sum() / rentals["rental_days"].sum()
    result = _station_categories(instance).merge(pair_rates, on=KEY, how="left", validate="one_to_one")
    result = result.merge(category_rates, on="category", how="left", validate="many_to_one")
    result["estimate_source"] = "station_category"
    result.loc[result["pair_rate"].isna(), "estimate_source"] = "category"
    result.loc[result["category_rate"].isna(), "estimate_source"] = "overall"
    result["base_revenue_per_car_day"] = result["pair_rate"].fillna(result["category_rate"]).fillna(overall_rate)
    return result[KEY + ["base_revenue_per_car_day", "estimate_source"]]
