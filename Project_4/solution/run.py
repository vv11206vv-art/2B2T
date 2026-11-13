"""Run the team plan pipeline from the Project_4 directory.

Usage: .venv/Scripts/python.exe -m solution.run --time-limit 180
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import pandas as pd

from fleetkit import baseline_plan, load_instance, write_plan
from solution.evaluate import estimate_profit
from solution.fleet import committed_car_days, validate_plan
from solution.forecast import base_revenue, demand_to_come
from solution.planning import search_level1


def action_counts(plan: pd.DataFrame) -> dict[str, int]:
    """Count action rows and cars, including a zero transfer count for Level 1."""
    return {"action_rows": len(plan), **{
        f"cars_{action}": int(plan.loc[plan.action.eq(action), "cars"].sum())
        for action in ("add", "remove", "transfer")
    }}


def main() -> None:
    """Write fresh candidate/comparison artifacts and verify a CSV round trip."""
    parser = argparse.ArgumentParser(description="Level 1 local forecast estimates; no official submission")
    parser.add_argument("--time-limit", type=float, default=180, help="search budget in seconds (default 180)")
    parser.add_argument("--max-rounds", type=int, default=500, help="deterministic accepted-move limit")
    args = parser.parse_args()
    started = perf_counter()
    project_dir = Path(__file__).resolve().parent.parent
    instance = load_instance(data_dir=project_dir / "data")
    committed = committed_car_days(instance=instance)
    demand = demand_to_come(instance=instance, committed=committed)
    base = base_revenue(instance=instance)
    print(f"Local forecast estimates: {len(demand)} demand cells, {len(base)} base rates.", flush=True)
    result = search_level1(instance=instance, committed=committed, demand=demand, base_revenue=base,
                           time_limit_seconds=args.time_limit, max_rounds=args.max_rounds)
    validate_plan(instance=instance, plan=result.plan)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"level1_{stamp}"
    plan_path = project_dir / "plans" / f"{stem}.csv"
    report_path = project_dir / "results" / f"{stem}.md"
    audit_path = project_dir / "results" / f"{stem}.json"
    # Exclusive reservation prevents accidental overwrite, including repeated runs.
    with plan_path.open("x", encoding="utf-8"):
        pass
    write_plan(df_plan=result.plan, path=plan_path)
    reloaded = pd.read_csv(plan_path, parse_dates=["date"])
    validate_plan(instance=instance, plan=reloaded)
    replay = estimate_profit(instance=instance, plan=reloaded, demand=demand, base_revenue=base)
    if any(abs(replay[key] - result.totals[key]) > 1e-6 for key in replay):
        raise RuntimeError("Written candidate does not reproduce the reported totals")
    reference_plans = {"do_nothing": pd.DataFrame(columns=result.plan.columns), "baseline": baseline_plan(instance=instance)}
    comparison = {}
    for name, reference in result.references.items():
        if reference["valid"]:
            comparison[name] = {**reference["totals"], **action_counts(reference_plans[name])}
            if replay["profit"] + 1e-6 < reference["totals"]["profit"]:
                raise RuntimeError("Candidate is worse than a valid reference")
    comparison["candidate"] = {**replay, **action_counts(reloaded)}
    runtime = perf_counter() - started
    audit = {
        "label": "Local forecast estimates, not official scores",
        "candidate_csv": str(plan_path.relative_to(project_dir)),
        "locally_valid": True, "csv_round_trip_verified": True,
        "total_runtime_seconds": runtime, "search": result.statistics,
        "references": result.references, "comparison": comparison,
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((project_dir / "data").glob("*.csv"))},
        "candidate_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
    }
    with audit_path.open("x", encoding="utf-8") as output:
        json.dump(audit, output, indent=2, allow_nan=False)
    names = list(comparison)
    metrics = list(comparison["candidate"])
    lines = ["# Level 1 local forecast estimates", "", "These are local forecast estimates, not official scorer results.", "",
             "| Metric | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
    for metric in metrics:
        lines.append("| " + metric + " | " + " | ".join(f"{comparison[name][metric]:,.2f}" for name in names) + " |")
    lines += ["", f"Candidate: `{plan_path.relative_to(project_dir).as_posix()}`.",
              "Local validity and CSV round-trip profit reproduction: passed.",
              f"Total runtime: {runtime:.3f} seconds; search runtime: {result.statistics['runtime_seconds']:.3f} seconds.",
              f"Settings: time limit {args.time_limit}s; max rounds {args.max_rounds}; deterministic move order; no random seed needed.",
              f"Screened {result.statistics['candidates_screened']} moves; accepted {result.statistics['accepted_moves']}; stopped: {result.statistics['stop_reason']}.",
              f"Best valid fallback: {result.statistics['best_reference']}.",
              f"Exact reference evaluation timings: { {name: ref.get('evaluation_seconds') for name, ref in result.references.items()} }.",
              f"Cached station scoring benchmark: {result.statistics['cached_station_score_seconds']:.8f}s.",
              "", "The heuristic does not prove global optimality. Candidate screening uses cached full-horizon capacity, pricing, upgrades, costs and validity constraints; final totals use the existing exact evaluator.",
              "Wall-clock cutoffs can select different last moves on different machines. Use the JSON move trace for an exact audit; fixed completed rounds with a generous time limit reproduce the deterministic search.",
              "Forecasts retain the historical-window edge effects; hidden demand scenarios are unavailable.",
              f"Settings, reference validity, input hashes and accepted moves: `{audit_path.name}`."]
    with report_path.open("x", encoding="utf-8") as output:
        output.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"\nAudit: {audit_path}\nReport: {report_path}", flush=True)


if __name__ == "__main__":
    main()
