# Team 2B2T — Fleet Allocation Challenge

## Approach

We plan fleet actions for 10 stations, four car categories and March 1–14, 2027.
The objective is profit: committed rental revenue plus revenue from new demand,
less holding, addition and transfer costs. We treat committed bookings and open
rentals as hard obligations. Every candidate must respect the category-wide
addition and removal caps and keep each station/category/day fleet at least as
large as its commitments.

Our pipeline has four parts:

1. `solution/fleet.py` reconstructs the initial fleet, committed car-days and
   next-day one-way movements. It simulates actions and checks plan validity.
2. `solution/forecast.py` estimates new demand by historical weekday occupancy
   and base revenue per car-day from completed rentals. These are forecasts, not
   known future values.
3. `solution/evaluate.py` estimates profit using the supplied price factors,
   adjacent-category upgrades, in-transit holding costs, and driver/truck costs.
4. `solution/planning.py` searches valid add/remove and transfer actions.
   `solution/robustness.py` compares plans under reproducible demand and revenue
   scenarios. The searches are bounded heuristics and do not prove global
   optimality.

The Level 0 calculations matched all 40 supplied values in `data/check_values.csv`
(10 selected cells, four metrics each). Those values check deterministic fleet
accounting, not forecast accuracy. Tests also cover plan timing, validity,
pricing, truck capacity and scenario reproducibility.

## Results

The organizer returned **feasible** public-scenario results for these saved plans:

| Plan | Official public profit | Action summary | Evidence |
| --- | ---: | --- | --- |
| Baseline | 3,111,408.11 | 96 added | `results/2B2T_baseline_trial_20261113.result.md` |
| Level 1 | 3,405,286.17 | 115 added, 233 removed | `results/level1_20261113T020936672087Z.comparison.md` |
| Level 2 | 3,569,957.06 | Level 1 actions plus 96 driver transfers | `results/level2_20261113T123921953877Z.comparison.md` |
| Level 3 | **3,929,026.32** | 115 added, 233 removed, 400 transferred | `results/level3_20261113T130545853303Z.comparison.md` |

Level 3 improved the official public result by **817,618.21** over the
baseline. Its exact saved CSV is `plans/level3_20261113T130545853303Z.csv`.
The official feedback, local estimates and input/plan hashes are preserved in
`results/`. Local forecast estimates and official public scores use different
demand inputs and should not be compared directly.

The Level 4 study in `results/level4_20261113T134857657528Z.md` tested 24
independent *local* sensitivity scenarios. Its candidate improved mean local
profit by 5,882.40 over the saved Level 3 plan. These scenarios are assumptions,
not the organizers' hidden scenarios. Historical proxy checks cannot observe
rejected demand, cancellations or past fleet constraints. At the time of this
draft, `plans/level4_20261113T134857657528Z.csv` was awaiting official feedback.
Update this paragraph and the final-plan decision when the genuine result arrives.

The final ranking uses mean profit over 30 hidden scenarios. One public result
does not establish that a plan will lead on the hidden ranking.

## How to run

From `Project_4`, use Python 3.14 or newer. Create a virtual environment and
install `requirements.txt`. For example, on Windows PowerShell:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -B -m unittest discover -s tests
& .\.venv\Scripts\python.exe -B -m solution.fleet
```

On macOS/Linux, `bash setup.sh` provides the supplied setup path. The last
command above compares the deterministic Level 0 calculations with
`check_values.csv`.

To generate new, timestamped local candidate files and reports:

```powershell
& .\.venv\Scripts\python.exe -B -m solution.run --level 3 --time-limit 120 --max-rounds 100 --max-cars-per-move 16
& .\.venv\Scripts\python.exe -B -m solution.robustness --seconds 120 --rounds 2
```

These commands write new local plans under `plans/` and reports under
`results/`; they do not submit to the organizer. Runtime-limited searches may
produce a different candidate on another computer. The saved CSVs and matching
reports above are the evidence for the stated scores.

## Deliverables and decision

- The short presentation is `submission/slides.pdf` (five slides).
- The exact final plan must be uploaded separately to the team's
  `Fleet-Challenge-Submissions/2B2T` folder as `final.csv` before the deadline.
- Choose the final CSV only after checking the Level 4 scorer result. Retain
  the officially feasible Level 3 CSV as fallback. Record which exact saved
  plan became `final.csv` and its hash here before the repository deadline.
