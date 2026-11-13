# Cost-only improvement to the team's public best plan

The original `plans/Level4robust.csv` is a byte-for-byte copy of the team-supplied
CSV from `results/Level4robust.csv` (SHA-256
`75506341d90a795d295ef96fa47ffce73b4745ad197ca1c2b25dc439b842163b`).
The genuine `results/Level4robust.result.md` says the plan is feasible on the
public scenario with profit **3,940,286.13**. Its action totals are 109 added,
233 removed and 400 transferred.

`plans/level4robust_all_trucks_20261113.csv` changes only the seven driver
transfer rows to truck. All seven rows move multiples of eight cars. Every
station, category, departure date, destination and car quantity is unchanged.
The candidate SHA-256 is
`88f13c62e4d89ee3375c6f9b68b534864f4e84b4f478f2366d1f3912783c5044`.

The full local validity checker accepts both plans. Reloading the candidate
from its written CSV gives the same 44 action rows and the same action totals.
The exact local evaluator reports:

| Component | Original | Truck-only candidate | Difference |
| --- | ---: | ---: | ---: |
| Transfer cost | 61,500.00 | 48,954.50 | -12,545.50 |
| Estimated profit | 3,713,771.95 | 3,726,317.45 | +12,545.50 |

Committed revenue, new-demand revenue, holding cost and addition cost are
identical in the two local evaluations. Under the README's shared route lead
times and truck cost rule, this is a cost-only improvement for *any* demand
scenario: the fleet trajectory and revenues do not change. Applying the same
12,545.50 cost difference to the original official public score predicts
**3,952,831.63** for the candidate. This is a prediction, not official
feedback; submit the candidate under its new filename and retain the genuine
scorer result before selecting `final.csv`.
