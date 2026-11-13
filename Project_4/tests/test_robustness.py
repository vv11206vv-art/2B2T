"""Focused robustness safeguards and exact-scorer comparisons."""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fleetkit import load_instance
from solution.robustness import history_audit, available_history, scenarios, price_audit, ScenarioScore, search, summary
from solution.forecast import demand_to_come, base_revenue
from solution.fleet import committed_car_days, validate_plan
from solution.evaluate import estimate_profit

class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.x=load_instance(data_dir=root/'data')
        cls.plan=pd.read_csv(root/'plans/level3_20261113T130545853303Z.csv',parse_dates=['date'])
        cls.d=demand_to_come(instance=cls.x,committed=committed_car_days(instance=cls.x))
        cls.b=base_revenue(instance=cls.x)
        cls.effects=[[.8,.9],[.9,1],[1.1,1],[1.2,1.1]]
    def test_asof_excludes_future_outcomes_and_bookings(self):
        cutoff=pd.Timestamp('2027-02-01')
        original=available_history(self.x.reservations,cutoff)
        changed=self.x.reservations.copy()
        future=(changed.return_date>=cutoff)|(changed.booking_date>=cutoff)
        changed.loc[future,'revenue']=1e12
        pd.testing.assert_frame_equal(original,available_history(changed,cutoff))
        self.assertTrue((original.return_date<cutoff).all())
        self.assertTrue((original.booking_date<cutoff).all())
    def test_reproducibility_and_shared_week_effects(self):
        a=scenarios(self.d,self.b,self.effects,1,3);b=scenarios(self.d,self.b,self.effects,1,3)
        for (d,r,m),(dd,rr,mm) in zip(a,b):
            pd.testing.assert_frame_equal(d,dd);pd.testing.assert_frame_equal(r,rr);self.assertEqual(m,mm)
            ratio=d.demand_to_come/self.d.demand_to_come.replace(0,np.nan)
            for day in d.date.unique():
                values=ratio[d.date.eq(day)].dropna()
                self.assertLess(values.max()-values.min(),1e-12)
    def test_nonnegative_finite_and_invalid_effects(self):
        for d,b,_ in scenarios(self.d,self.b,self.effects,2,10):
            for v in [d.demand_to_come,b.base_revenue_per_car_day]:self.assertTrue((np.isfinite(v)&v.ge(0)).all())
        with self.assertRaises(ValueError):scenarios(self.d,self.b,[[float('nan'),1]]*4,2,1)
    def test_all_prices_and_irregular_curve(self):
        rows=price_audit(self.x)
        self.assertEqual(len(rows),8)
        odd=next(p for p in rows if p['station_type']=='downtown' and p['category']=='midsize')
        self.assertTrue(odd['nonconcave']);self.assertFalse(odd['nonmonotone'])
        self.assertAlmostEqual(odd['total_revenue_per_demand_base'][3],.8*.990)
    def test_scenario_profits_match_exact_with_upgrades(self):
        samples=scenarios(self.d,self.b,self.effects,3,2)
        fast=ScenarioScore(self.x,self.d,self.b,samples).score(self.plan)
        for v,(d,b,_) in zip(fast,samples):
            actual=estimate_profit(instance=self.x,plan=self.plan,demand=d,base_revenue=b)
            self.assertAlmostEqual(v,actual['profit'],places=6)
    def test_valid_fallback_and_invalid_plan_rejected(self):
        scorer=ScenarioScore(self.x,self.d,self.b,scenarios(self.d,self.b,self.effects,4,2))
        result,log=search(self.x,self.plan,scorer,seconds=0,rounds=1)
        pd.testing.assert_frame_equal(result,self.plan);validate_plan(instance=self.x,plan=result)
        broken=self.plan.copy();broken.loc[0,'cars']=-1
        with self.assertRaises(ValueError):scorer.score(broken)
    def test_historical_audit_has_complete_proxy_cells(self):
        audit=history_audit(self.x)
        self.assertEqual(len(audit["folds"]),2)
        self.assertTrue(all(f["test_cells"]==280 and f["rate_pairs"]==40 for f in audit["folds"]))

    def test_summary_manual(self):
        self.assertEqual(summary([1,2,3,4])['mean'],2.5)
        self.assertEqual(summary([1,2,3,4])['lower_tail_mean'],1)

if __name__=='__main__':unittest.main()
