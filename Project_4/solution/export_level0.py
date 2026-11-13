"""Export Level 0 inspection CSVs, without changing existing different files.

From Project_4: .venv/Scripts/python.exe -B -m solution.export_level0
Demand to come and base revenue are estimates. These are not action plans.
"""

import difflib
from io import StringIO
from pathlib import Path

import pandas as pd

from fleetkit import Instance, load_instance
from solution.fleet import initial_fleet, committed_car_days, oneway_flows, simulate_fleet
from solution.forecast import demand_to_come, base_revenue

KEY = ["station_id", "category"]
DAY_KEY = KEY + ["date"]


def _check_keys(table: pd.DataFrame, expected: pd.DataFrame, keys: list[str]) -> None:
    """Reject nulls, duplicates, missing keys or extra keys before any write."""
    if table.isna().any().any():
        raise ValueError("Level 0 export contains null values")
    coverage = expected[keys].merge(table[keys], on=keys, how="outer", validate="one_to_one", indicator=True)
    if not coverage["_merge"].eq("both").all():
        raise ValueError(f"Level 0 export key mismatch: {coverage.loc[coverage['_merge'].ne('both')].to_dict('records')}")


def _sorted(table: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return table.sort_values(keys).reset_index(drop=True)


def _read_csv(source, table: pd.DataFrame) -> pd.DataFrame:
    """Parse floating columns with Python's correctly rounded float conversion."""
    return pd.read_csv(
        source,
        parse_dates=["date"] if "date" in table.columns else [],
        converters={column: float for column in table if pd.api.types.is_float_dtype(table[column])},
    )


def build_exports(*, instance: Instance) -> dict[str, pd.DataFrame]:
    """Build and check all three tables using only the existing Level 0 functions."""
    pairs = instance.stations[["station_id"]].merge(instance.categories[["category"]], how="cross")
    dates = pd.DataFrame({"date": pd.date_range(instance.horizon.first_date.iloc[0], instance.horizon.last_date.iloc[0])})
    grid = pairs.merge(dates, how="cross")
    initial = initial_fleet(instance=instance)
    committed = committed_car_days(instance=instance)
    components = [
        committed,
        oneway_flows(instance=instance),
        simulate_fleet(instance=instance, plan=pd.DataFrame()),
        demand_to_come(instance=instance, committed=committed),
    ]
    daily = grid.copy()
    for component in components:
        _check_keys(component, grid, DAY_KEY)
        daily = daily.merge(component, on=DAY_KEY, validate="one_to_one")
        _check_keys(daily, grid, DAY_KEY)
        # Confirm joining did not change even one source value.
        pd.testing.assert_frame_equal(
            _sorted(daily[component.columns], DAY_KEY), _sorted(component, DAY_KEY), check_exact=True
        )
    rates = base_revenue(instance=instance)
    _check_keys(initial, pairs, KEY)
    _check_keys(rates, pairs, KEY)
    checks = instance.check_values
    metrics = ["committed_car_days", "oneway_arrivals", "oneway_departures", "station_fleet"]
    compared = checks.merge(daily[DAY_KEY + metrics], on=DAY_KEY, how="left", validate="one_to_one", suffixes=("_expected", "_calculated"))
    for metric in metrics:
        mismatch = compared[f"{metric}_expected"].ne(compared[f"{metric}_calculated"])
        if mismatch.any():
            raise ValueError(f"Check-value mismatch for {metric}: {compared.loc[mismatch].to_dict('records')}")
    return {
        "initial_fleet.csv": _sorted(initial, KEY),
        "daily_picture.csv": _sorted(daily, DAY_KEY),
        "base_revenue.csv": _sorted(rates, KEY),
    }


def export_level0(*, instance: Instance, output_dir: Path) -> dict[str, int]:
    """Write absent files; leave identical files untouched; refuse any conflict.

    All conflicts are reported before writing any CSV. Values are checked
    exactly after serialization and after reading the actual exported files.
    """
    tables = build_exports(instance=instance)
    payloads = {}
    conflicts = []
    for name, table in tables.items():
        payload = table.to_csv(index=False, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n").encode("utf-8")
        decoded = _read_csv(StringIO(payload.decode()), table)
        pd.testing.assert_frame_equal(decoded, table, check_exact=True, check_dtype=False)
        payloads[name] = payload
        path = output_dir / name
        if path.exists() and path.read_bytes() != payload:
            diff = list(difflib.unified_diff(path.read_bytes().decode("utf-8", errors="replace").splitlines(), payload.decode().splitlines(), fromfile=str(path), tofile="calculated " + name, lineterm=""))
            conflicts.append(str(path) + ": existing contents differ (not overwritten)\n" + "\n".join(diff[:80]))
    if conflicts:
        raise FileExistsError("\n".join(conflicts))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        path = output_dir / name
        if not path.exists():
            with path.open("xb") as output:
                output.write(payload)
        table = tables[name]
        readback = _read_csv(path, table)
        pd.testing.assert_frame_equal(readback, table, check_exact=True, check_dtype=False)
    return {name: len(table) for name, table in tables.items()}


def main() -> None:
    project = Path(__file__).resolve().parent.parent
    instance = load_instance(data_dir=project / "data")
    output_dir = project / "results" / "level0"
    try:
        counts = export_level0(instance=instance, output_dir=output_dir)
    except FileExistsError as error:
        raise SystemExit(str(error)) from error
    for name, rows in counts.items():
        print(f"{output_dir / name}: {rows} rows")
    print(f"All {len(instance.check_values) * 4} supplied check values match; exported values equal function outputs exactly.")
    print("Demand and base revenue are estimates. Inspection exports are separate from the submitted action-plan CSV.")


if __name__ == "__main__":
    main()
