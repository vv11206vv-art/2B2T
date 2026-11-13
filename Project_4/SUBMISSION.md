# Submission: Project 4 — Fleet Allocation Challenge — Team 2B2T

## Authors

**Team name:** 2B2T

**Project:** Project 4: Fleet Allocation Challenge

## Problem and approach

We choose additions, removals and transfers across 10 stations and four car categories for March 1–14, 2027. The objective is profit after holding, addition and transfer costs. Every committed rental must retain its car; category caps, transfer lead times and daily fleet coverage are hard constraints. Future new demand and revenue rates are uncertain.

We reconstruct fleet and commitments from supplied data, estimate weekday demand and revenue per car-day from eight weeks of history, and evaluate candidate actions using the README's exact rules. The evaluator includes adjacent-category upgrades, holding costs in transit, and truck sharing across categories on the same route/day. Bounded local search improves valid plans; scenario comparisons reduce dependence on one forecast. Public scores were not used to calibrate the scenarios or search objective.

Two useful findings: switching seven full-load driver transfers to trucks saved **12,545.50** with identical fleet movements, so that saving is independent of demand; downtown midsize has a **non-concave** revenue curve, which we retain exactly rather than smoothing. CSV hashes, accepted-edit traces and reload checks connect plans to their local reports.

## Models and why

The submitted pipeline runs locally without LLM inference. GPT models assisted development and review; we did not conduct a controlled comparison of their coding performance or measure their usage cost. Numerical candidates were compared on identical scenario draws.

| Model | Open-weight or closed | Where it runs | Main metric(s) | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Historical weekday demand and weighted revenue-rate estimates | N/A: numerical method | Local Python | Two rolling proxy tests: demand MAE 6.27 / 6.18 car-days per cell; rate MAE 8.31 / 6.82 per car-day | Runtime not measured separately; no API cost | Yes; transparent starting forecast |
| Scenario-mean local search | N/A: numerical method | Local Python | Final fresh-check mean profit 3,745,297.36; paired gain 95,617.18 over incumbent | Final pipeline 257.16 s; search 154.34 s | Yes; improved all 32 fresh check draws |
| GPT-6.1 Sol and GPT-6 Astra | Closed | Codex development environment | No controlled LLM benchmark; generated code checked by tests and exact evaluation | Usage cost and per-task time not recorded | Development/review assistance only |

Settings: 16 equally weighted selection scenarios (seed 5201), then 32 fresh check scenarios (seed 62013), used once for acceptance. Scenarios combine four interior historical weeks, shared network effects, demand sensitivity U(0.8, 1.2), and revenue-rate sensitivity U(0.9, 1.1). These ranges are assumptions. Search budget: 300 seconds and 20 accepted rounds. An independently GPT-generated `plans/Level4robust.csv` was one candidate source; its model was not recorded. We verified it, reduced transfer costs, and used the resulting plan as the final search's incumbent.

## Results

**Organizer feedback on the public scenario:** all plans below were feasible.

| Plan | Official profit |
| --- | ---: |
| Supplied baseline | 3,111,408.11 |
| Level 3 | 3,929,026.32 |
| Independent GPT candidate | 3,940,286.13 |
| **Selected final plan** | **4,069,442.37** |

The final plan improves the baseline by **958,034.26 (30.79%)**. It has 49 action rows: **39 cars added, 233 removed and 556 transferred, all by truck**. Its official components are committed revenue 2,851,079.81; new revenue 3,785,190.07; holding cost 2,475,618.00; addition cost 17,745.00; transfer cost 73,464.50. Genuine final feedback: `final_selection_20261113T145218373087Z.result.md`.

On the same 32 fresh simulated check scenarios, the final plan's mean profit was **3,745,297.36**, versus **3,649,680.18** for the all-truck incumbent. Every paired gain was positive; the minimum was 84,374.45. Mean profit in the worst four draws rose from 3,010,002.61 to 3,095,156.02, while overall profit spread increased slightly. These are local sensitivity results, not official hidden scores.

Verification: **75 tests passed**, all **40 supplied Level 0 values** matched, and the reloaded CSV reproduced every component across all 48 selection/check scenarios. Details and hashes are in `results/final_selection_20261113T145218373087Z.md` and its matching JSON.

Limits: historical completion/window boundaries affect occupancy estimates; four interior weeks cannot establish reliable tail probabilities. The README states past fleet was ample and no demand was lost. Scenario distributions remain assumptions, and fresh draws from them are not independent real-world evidence. Search stopped at its round limit and does not prove global optimality. Final ranking uses 30 hidden scenarios; public success does not establish the hidden score.

## How to run

Use **Python 3.14+**. From a fresh clone, in Windows PowerShell:

```powershell
git clone https://github.com/vv11206vv-art/2B2T.git
cd 2B2T/Project_4
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -B -m unittest discover -s tests
& .\.venv\Scripts\python.exe -B -m solution.fleet
& .\.venv\Scripts\python.exe -B -m solution.final_search --seconds 300 --rounds 20
```

Data is in `data/`; no environment variables, secrets or API keys are required. On macOS/Linux, use `.venv/bin/python` for the Python commands. The final search writes new timestamped CSVs under `plans/`, reports under `results/`, and a local `submission/final.csv`, backing up an existing copy. It does not upload to the organizers.

Time limits can change the candidate on another computer. The selected recorded plan is `plans/final_selection_20261113T145218373087Z.csv`; its identical copy is `submission/final.csv`. Both have SHA-256:

```text
929ab8810425985eabba875a453cdcaaab675963fd0f8543253e9f356b50699f
```

The audit preserves the run's scenarios, inputs, edits and component totals. Official profit requires the organizers' scorer and cannot be regenerated from the local forecasts.

## Demo

`submission/slides.pdf` — five slides. For organizer submission, use the selected plan as `final.csv` in the team's shared folder.

## Next steps

SIXT could use this prototype to compare fleet actions while protecting bookings and making movement costs explicit. Separate fleet, forecast, evaluator and planner modules support maintenance and adaptation. With more time, we would validate on longer historical windows, calibrate scenario variation, add operational driver/truck availability, and connect updated booking and fleet data. Before real deployment, decisions and forecasts would require operational validation.
