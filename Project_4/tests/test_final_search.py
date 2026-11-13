"""Independent exact checks for coordinated search and final-file preservation."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import test_evaluate as fixtures
from solution.final_search import PortfolioScore, edited, neighborhoods, bounded_search, install_local_final
from solution.evaluate import estimate_profit
from solution.fleet import PLAN_COLUMNS, validate_plan


class FinalSearchTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.EvaluatorTests();fixture.setUp();fixture.set_cars([20,20,20,20])
        self.x=fixture.instance
        pairs=self.x.stations[['station_id']].merge(self.x.categories[['category']],how='cross')
        self.base=pairs.copy();self.base['base_revenue_per_car_day']=100.
        self.demand=pairs.merge(pd.DataFrame({'date':pd.date_range('2027-03-01',periods=3)}),how='cross')
        self.demand['demand_to_come']=15.
        low=self.demand.copy();low['demand_to_come']*=.5
        self.samples=[(self.demand,self.base,{}),(low,self.base,{})]
        self.score=PortfolioScore(self.x,self.demand,self.base,self.samples)

    def plan(self,rows):
        p=pd.DataFrame(rows,columns=PLAN_COLUMNS);p['date']=pd.to_datetime(p.date);return p

    def test_all_components_match_exact_for_coordinated_changes(self):
        p=self.plan([('add','2027-03-02','economy','S2',None,None,2),
                     ('remove','2027-03-03','premium','S1',None,None,3),
                     ('transfer','2027-03-01','economy','S1','S2','truck',4),
                     ('transfer','2027-03-01','compact','S1','S2','truck',5),
                     ('transfer','2027-03-03','midsize','S1','S2','driver',2)])
        values=self.score.score_state(self.score.state(p))
        for v,(d,b,_) in zip(values,self.samples):
            exact=estimate_profit(instance=self.x,plan=p,demand=d,base_revenue=b)
            self.assertAlmostEqual(v,exact['profit'],places=7)
        # Includes 9 truck cars across categories and a transfer arriving after the horizon.
        self.assertEqual(exact['transfer_cost'],310.)

    def test_cache_rejects_caps_and_future_shortages(self):
        with self.assertRaises(ValueError):self.score.score_state({('add',0,0,0,-1,''):101})
        with self.assertRaises(ValueError):self.score.score_state({('remove',2,0,0,-1,''):21})
        with self.assertRaises(ValueError):self.score.score_state({('transfer',0,0,0,0,'truck'):1})

    def test_atomic_reallocation_at_full_addition_cap(self):
        original={('add',0,0,0,-1,''):100}
        trial=edited(CounterLike(original),[(('add',0,0,0,-1,''),-1),(('add',0,0,1,-1,''),1)])
        self.score.score_state(trial)
        self.assertEqual(original,{('add',0,0,0,-1,''):100})

    def test_addition_replacement_preserves_arrival_and_reduces_cost(self):
        p=self.plan([('add','2027-03-03','economy','S2',None,None,1)])
        state=self.score.state(p)
        move=next(edits for kind,edits in neighborhoods(state,self.score) if kind=='replace_addition_transfer')
        trial=edited(state,move);plan=self.score.plan(trial)
        self.assertEqual(plan.action.tolist(),['transfer'])
        self.assertEqual(plan.date.tolist(),[pd.Timestamp('2027-03-01')])
        validate_plan(instance=self.x,plan=plan)
        expected=estimate_profit(instance=self.x,plan=plan,demand=self.demand,base_revenue=self.base)
        self.assertAlmostEqual(self.score.score_state(trial)[0],expected['profit'])

    def test_zero_budget_retains_incumbent_and_limits_enforced(self):
        p=self.plan([('add','2027-03-03','economy','S2',None,None,1)])
        result,log=bounded_search(self.score,p,seconds=0)
        pd.testing.assert_frame_equal(result,p)
        self.assertEqual(log['tested'],{})
        with self.assertRaises(ValueError):bounded_search(self.score,p,seconds=301)

    def test_existing_final_backed_up_byte_for_byte(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            root=Path(folder);source=root/'selected.csv';target=root/'final.csv'
            source.write_bytes(b'new\r\n');target.write_bytes(b'original\n')
            backup=install_local_final(source,target,'test')
            self.assertEqual(backup.read_bytes(),b'original\n')
            self.assertEqual(target.read_bytes(),source.read_bytes())


def CounterLike(values):
    from collections import Counter
    return Counter(values)


if __name__=='__main__':unittest.main()
