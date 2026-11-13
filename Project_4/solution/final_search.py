"""One bounded scenario search and a reviewable local final bundle.

Run from Project_4: python -B -m solution.final_search --seconds 300 --rounds 20
No scorer feedback is used in the objective or scenario calibration.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from fleetkit import load_instance, write_plan
from solution.evaluate import estimate_profit
from solution.fleet import PLAN_COLUMNS, committed_car_days, simulate_fleet, transit_cars, validate_plan, compare_check_values
from solution.forecast import demand_to_come, base_revenue
from solution.robustness import ScenarioScore, scenarios, history_audit, price_audit, summary
from solution.run import action_counts


class PortfolioScore(ScenarioScore):
    """Whole-plan R1-R11 calculation with fixed data cached across candidates.

    Keys are (action, day, category, origin, destination, mode), with integer
    indices. Truck costs aggregate all categories by route/day. Invalid states
    raise ValueError. Exact evaluator independently verifies every accepted plan.
    """
    def __init__(self, instance, demand, base, samples):
        super().__init__(instance, demand, base, samples)
        c = self.cache
        self.routes = {(c.stations.index(r.from_station_id), c.stations.index(r.to_station_id)):
                       (int(r.lead_time_days), float(r.distance_km)) for r in instance.distances.itertuples()}
        costs = instance.transfer_costs.set_index('mode')
        self.costs = {(o, d, m): float(costs.loc[m, 'fixed_cost'] + costs.loc[m, 'cost_per_km'] * distance)
                      for (o, d), (_, distance) in self.routes.items() for m in ('driver', 'truck')}
        self.constant = estimate_profit(instance=instance, plan=pd.DataFrame(columns=PLAN_COLUMNS),
                                        demand=demand, base_revenue=base)['committed_revenue']

    def state(self, plan):
        validate_plan(instance=self.instance, plan=plan)
        c = self.cache
        state = Counter()
        for r in plan.itertuples():
            key = (r.action, c.days.get_loc(pd.Timestamp(r.date)), c.categories.index(r.category),
                   c.stations.index(r.station_id), c.stations.index(r.to_station_id) if r.action == 'transfer' else -1,
                   r.mode if r.action == 'transfer' else '')
            state[key] += int(r.cars)
        return state

    def plan(self, state):
        c = self.cache
        return pd.DataFrame([(a, c.days[t], c.categories[k], c.stations[o], c.stations[d] if a == 'transfer' else None,
                              m if a == 'transfer' else None, int(q)) for (a,t,k,o,d,m),q in sorted(state.items()) if q], columns=PLAN_COLUMNS)

    def score_state(self, state):
        c = self.cache
        fleet = c.start.copy()
        totals = np.zeros_like(c.caps)
        transit_holding = addition = drivers = 0.0
        trucks = Counter()
        for (a,t,k,o,d,m), q in state.items():
            if not isinstance(q, (int,np.integer)) or q < 1 or not (0 <= t < len(c.days) and 0 <= k < len(c.categories) and 0 <= o < len(c.stations)):
                raise ValueError('Invalid state fields')
            if a in ('add','remove'):
                if d != -1 or m != '': raise ValueError('Invalid non-transfer fields')
                index = int(a == 'remove'); totals[index,k] += q
                fleet[o,t:,k] += q if a == 'add' else -q
                if a == 'add': addition += q*c.addition[k]
            elif a == 'transfer' and (o,d) in self.routes and m in ('driver','truck'):
                lead, _ = self.routes[o,d]
                fleet[o,t:,k] -= q; fleet[d,t+lead:,k] += q
                transit_holding += q*min(lead,len(c.days)-t)*c.holding[k]
                if m == 'driver': drivers += q*self.costs[o,d,m]
                else: trucks[o,d,t] += q
            else: raise ValueError('Invalid action or route')
        if np.any(totals > c.caps) or np.any(fleet < c.committed):
            raise ValueError('Category cap or daily commitment shortage')
        holding = float((fleet*c.holding).sum()) + transit_holding
        transfer = drivers + sum(((q+7)//8)*self.costs[o,d,'truck'] for (o,d,t),q in trucks.items())
        capacity = np.maximum(0,c.utilization[:,None,None]*fleet-c.committed)
        own = np.minimum(self.dem,capacity); served = own.copy()
        served[:,:,:,:-1] += np.minimum(capacity[None,:,:,1:]-own[:,:,:,1:],self.dem[:,:,:,:-1]-own[:,:,:,:-1])
        revenue = np.zeros(len(self.dem))
        for s in range(len(c.stations)):
            for k in range(len(c.categories)):
                demand = self.dem[:,s,:,k]
                share = np.divide(served[:,s,:,k],demand,out=np.zeros_like(demand),where=demand>0)
                x,y = c.prices[s,k]
                revenue += (np.interp(share,x,y)*demand*self.base[:,s,k,None]).sum(axis=1)
        return revenue+self.constant-holding-addition-transfer


def edited(state, edits):
    """Apply a coordinated move atomically; do not mutate the incumbent."""
    result = state.copy()
    for key,delta in edits:
        result[key] += delta
        if result[key] < 0: raise ValueError('Negative action quantity')
        if result[key] == 0: del result[key]
    return result


def neighborhoods(state, score):
    """Round-robin families avoid starving coordinated changes at a time cutoff.

    Quantities 1/8, new transfers 8/16, date endpoints and +/-1; additions can
    move between stations even at a full cap. Replacement moves cancel an
    addition and arrange a transfer to arrive by its action date. Group moves
    align route departures or add two categories into one eight-car truck.
    """
    c = score.cache; keys = sorted(state); days = len(c.days)
    departures = sorted({0, min(3,days-1), min(7,days-1), min(10,days-1), days-1})
    def unary():
        for key in keys:
            a,t,k,o,d,m = key; q=state[key]
            yield 'cancel', [(key,-q)]
            for delta in (-8,-1,1,8):
                if q+delta>0: yield 'resize_'+a, [(key,delta)]
            for date in sorted({0,days-1,t-1,t+1}):
                if 0<=date<days and date!=t:
                    yield 'date_'+a, [(key,-q),((a,date,k,o,d,m),q)]
            if a=='transfer':
                yield 'mode',[(key,-q),((a,t,k,o,d,'truck' if m=='driver' else 'driver'),q)]
                for target in range(len(c.stations)):
                    if (o,target) in score.routes and target!=d:
                        yield 'destination',[(key,-q),((a,t,k,o,target,m),q)]
                    if (target,d) in score.routes and target!=o:
                        yield 'origin',[(key,-q),((a,t,k,target,d,m),q)]
            else:
                for target in range(len(c.stations)):
                    if target!=o:
                        for n in sorted({1,q}):
                            yield 'reallocate_'+a,[(key,-n),((a,t,k,target,-1,''),n)]
    def replace_additions():
        for key in keys:
            a,t,k,d,_,_ = key
            if a!='add':continue
            for (o,dest),(lead,_) in sorted(score.routes.items()):
                if dest==d and t>=lead:
                    for n in sorted({1,min(8,state[key]),state[key]}):
                        yield 'replace_addition_transfer',[(key,-n),(('transfer',t-lead,k,o,d,'truck'),n)]
    def new_actions():
        for t in departures:
            for k in range(len(c.categories)):
                for o in range(len(c.stations)):
                    for a in ('add','remove'):
                        for q in (1,8):yield 'new_'+a,[((a,t,k,o,-1,''),q)]
                for o,d in sorted(score.routes):
                    for q in (8,16):yield 'new_transfer',[(('transfer',t,k,o,d,'truck'),q)]
    def group_moves():
        transfers=[key for key in keys if key[0]=='transfer']
        for o,d,t in sorted({(key[3],key[4],key[1]) for key in transfers}):
            members=[key for key in transfers if (key[3],key[4],key[1])==(o,d,t)]
            for day in sorted({key[1] for key in transfers if (key[3],key[4])==(o,d)}):
                if day!=t:
                    edits=[]
                    for key in members:
                        a,_,k,_,_,m=key
                        edits.extend([(key,-state[key]),((a,day,k,o,d,'truck'),state[key])])
                    yield 'align_truck_group',edits
            drivers=[key for key in members if key[5]=='driver']
            if drivers:
                edits=[]
                for key in drivers:edits.extend([(key,-state[key]),(key[:5]+('truck',),state[key])])
                yield 'truck_group_mode',edits
        for t in departures:
            for o,d in sorted(score.routes):
                for k in range(len(c.categories)-1):
                    yield 'shared_truck_add',[(('transfer',t,k,o,d,'truck'),4),(('transfer',t,k+1,o,d,'truck'),4)]
    families=[iter(unary()),iter(replace_additions()),iter(new_actions()),iter(group_moves())]
    while families:
        remaining=[]
        for family in families:
            try:yield next(family);remaining.append(family)
            except StopIteration:pass
        families=remaining


def bounded_search(score, incumbent, seconds=300, rounds=20):
    """One best-improvement run, with five seconds reserved before hard budget.

    Whole-plan scenario mean is primary. Fixed ordering and round bounds are
    reproducible; a wall-clock cutoff can truncate the last neighborhood.
    """
    if not np.isfinite(seconds) or not 0<=seconds<=300 or not isinstance(rounds,int) or rounds<0:
        raise ValueError('Search budget must be 0..300 seconds; rounds nonnegative')
    start=perf_counter(); deadline=start+max(0,seconds-5)
    state=score.state(incumbent); values=score.score_state(state)
    tested=Counter();invalid=Counter();trace=[];reason='round_limit'
    for round_index in range(rounds):
        best=float(values.mean());winner=None;timed_out=False
        for kind,edits in neighborhoods(state,score):
            if perf_counter()>=deadline:timed_out=True;reason='time_limit';break
            tested[kind]+=1
            try:
                trial=edited(state,edits); v=score.score_state(trial)
            except ValueError:invalid[kind]+=1;continue
            if v.mean()>best+1e-6:best=float(v.mean());winner=(trial,v,kind,edits)
        if winner is None:
            if not timed_out:reason='local_neighborhood_exhausted'
            break
        state,values,kind,edits=winner
        plan=score.plan(state);validate_plan(instance=score.instance,plan=plan)
        # Existing exact evaluator plus existing scenario scorer cross-check.
        checked=score.score(plan)
        if not np.allclose(values,checked,rtol=0,atol=1e-6):raise RuntimeError('Cached score mismatch')
        trace.append(dict(round=round_index+1,kind=kind,selection_mean=best,edits=[dict(key=list(k),delta=int(v)) for k,v in edits]))
        print(f'Accepted round {round_index+1}: {kind}, local mean {best:.2f}',flush=True)
        if timed_out:break
    return score.plan(state),dict(tested=dict(tested),invalid=dict(invalid),trace=trace,stop=reason,
                                 runtime_seconds=perf_counter()-start,budget_seconds=seconds,max_rounds=rounds)


def exact_comparison(instance, plans, samples, score):
    """Audit every plan/scenario component with the existing exact evaluator."""
    audit={}; stats={}
    for name,plan in plans.items():
        totals=[estimate_profit(instance=instance,plan=plan,demand=d,base_revenue=b) for d,b,_ in samples]
        values=[t['profit'] for t in totals]
        if not np.allclose(score.score_state(score.state(plan)),values,rtol=0,atol=1e-6):raise RuntimeError('Exact scenario discrepancy')
        audit[name]=totals;stats[name]=summary(values)
    return audit,stats


def install_local_final(selected, target, stamp):
    """Copy identical bytes; preserve any previous final before replacement."""
    data=selected.read_bytes();backup=None
    if target.exists():
        backup=target.with_name(f'final_backup_{stamp}.csv')
        with backup.open('xb') as f:f.write(target.read_bytes())
        if backup.read_bytes()!=target.read_bytes():raise RuntimeError('Backup verification failed')
    with target.open('wb') as f:f.write(data)
    if target.read_bytes()!=data:raise RuntimeError('Local final copy mismatch')
    return backup


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=300);parser.add_argument('--rounds',type=int,default=20)
    args=parser.parse_args();started=perf_counter();root=Path(__file__).resolve().parents[1]
    paths=['plans/level3_20261113T130545853303Z.csv','plans/level4_20261113T134857657528Z.csv','plans/Level4robust.csv','plans/level4robust_all_trucks_20261113.csv']
    originals={p:(root/p).read_bytes() for p in paths if (root/p).exists()}
    plans={p:pd.read_csv(root/p,parse_dates=['date']) for p in originals}
    instance=load_instance(data_dir=root/'data')
    d=demand_to_come(instance=instance,committed=committed_car_days(instance=instance));b=base_revenue(instance=instance)
    history=history_audit(instance);prices=price_audit(instance)
    # Seeds/counts fixed before seeing any final-check results.
    selection=scenarios(d,b,history['weekly_effects'],5201,16)
    score=PortfolioScore(instance,d,b,selection)
    cost_check={}
    external='plans/Level4robust.csv';trucks='plans/level4robust_all_trucks_20261113.csv'
    if external in plans and trucks in plans:
        left,right=plans[external],plans[trucks]
        pd.testing.assert_frame_equal(left.drop(columns='mode'),right.drop(columns='mode'))
        changed=left['mode'].fillna('')!=right['mode'].fillna('')
        assert changed.sum()==7 and left.loc[changed,'mode'].eq('driver').all() and right.loc[changed,'mode'].eq('truck').all()
        pd.testing.assert_frame_equal(simulate_fleet(instance=instance,plan=left),simulate_fleet(instance=instance,plan=right))
        pd.testing.assert_frame_equal(transit_cars(instance=instance,plan=left),transit_cars(instance=instance,plan=right))
        lt=estimate_profit(instance=instance,plan=left,demand=d,base_revenue=b);rt=estimate_profit(instance=instance,plan=right,demand=d,base_revenue=b)
        assert abs(lt['transfer_cost']-61500)<1e-6 and abs(rt['transfer_cost']-48954.5)<1e-6
        for key in ('committed_revenue','new_demand_revenue','holding_cost','addition_cost'):assert lt[key]==rt[key]
        cost_check=dict(changed_modes=7,before=lt['transfer_cost'],after=rt['transfer_cost'],saving=12545.5,fleet_and_transit_identical=True)
    selection_totals,selection_stats=exact_comparison(instance,plans,selection,score)
    incumbent=max(selection_stats,key=lambda n:selection_stats[n]['mean'])
    print('Selection comparison:',json.dumps(selection_stats),flush=True)
    print('Incumbent:',incumbent,flush=True)
    candidate,search=bounded_search(score,plans[incumbent],args.seconds,args.rounds)
    selection_totals_new,selection_stats_new=exact_comparison(instance,{'searched':candidate},selection,score)
    selection_totals.update(selection_totals_new);selection_stats.update(selection_stats_new)
    holdout=scenarios(d,b,history['weekly_effects'],62013,32)
    check_score=PortfolioScore(instance,d,b,holdout)
    check_totals,check_stats=exact_comparison(instance,{'incumbent':plans[incumbent],'searched':candidate},holdout,check_score)
    gains=np.array([t['profit'] for t in check_totals['searched']])-np.array([t['profit'] for t in check_totals['incumbent']])
    promote=bool(gains.mean()>1e-6 and selection_stats['searched']['mean']>selection_stats[incumbent]['mean']+1e-6)
    selected=candidate if promote else plans[incumbent]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');stem='final_selection_'+stamp
    output=root/'plans'/(stem+'.csv')
    with output.open('x'):pass
    write_plan(df_plan=selected,path=output)
    reloaded=pd.read_csv(output,parse_dates=['date']);validate_plan(instance=instance,plan=reloaded)
    # Reproduce every component after CSV serialization on BOTH scenario sets.
    for samples,expected in [(selection,selection_totals['searched' if promote else incumbent]),(holdout,check_totals['searched' if promote else 'incumbent'])]:
        for (dd,bb,_),totals in zip(samples,expected):
            actual=estimate_profit(instance=instance,plan=reloaded,demand=dd,base_revenue=bb)
            assert all(abs(actual[k]-totals[k])<1e-6 for k in actual)
    assert compare_check_values(instance=instance).empty
    for p,data in originals.items():assert (root/p).read_bytes()==data
    backup=install_local_final(output,root/'submission/final.csv',stamp)
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    result=dict(selected=output.relative_to(root).as_posix(),sha256=digest,local_final='submission/final.csv',backup=str(backup) if backup else None,
                incumbent=incumbent,accepted_after_independent_check=promote,selection=selection_stats,independent_check=check_stats,paired_gain=summary(gains),
                paired_gain_standard_error=float(gains.std(ddof=1)/np.sqrt(len(gains))),counts=action_counts(reloaded),cost_only_check=cost_check,search=search,
                selection_components=selection_totals,check_components=check_totals,selection_seed=5201,check_seed=62013,
                scenario_draws={'selection':[m for _,_,m in selection],'independent_check':[m for _,_,m in holdout]},weekly_effects=history['weekly_effects'],
                prices=prices,source_hashes={p:hashlib.sha256(data).hexdigest() for p,data in originals.items()},
                input_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data').glob('*.csv')},
                implementation_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'solution').glob('*.py')},
                provenance='Final scenario search seeded by '+incumbent+'. Level4robust.csv is an independent GPT-generated external candidate, not our original search output.',
                official_status='Unknown for the newly written final selection. New official trial recommended before final upload.',runtime_seconds=perf_counter()-started)
    with (root/'results'/(stem+'.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    lines=['# Final local selection for review','',result['provenance'],'',
           'Objective: average estimated profit, aligned with the 60% hidden-rank judging component. Reproducibility, constraints and honest evidence support the 40% jury assessment. No public score is used to calibrate forecasts, probabilities or search gains.', '',
           f'Incumbent: `{incumbent}`; highest selection mean among all available valid references. Search promoted after independent check: **{promote}**.', '',
           '## Same-scenario selection comparison','', '| Plan | Mean | Std deviation | P10 | Worst 10% mean |','|---|---:|---:|---:|---:|']
    for name,v in selection_stats.items():lines.append(f"| {name} | {v['mean']:.2f} | {v['std']:.2f} | {v['p10']:.2f} | {v['lower_tail_mean']:.2f} |")
    lines+=['','## Independent final check','', '| Plan | Mean | Std deviation | P10 | Worst 10% mean |','|---|---:|---:|---:|---:|']
    for name,v in check_stats.items():lines.append(f"| {name} | {v['mean']:.2f} | {v['std']:.2f} | {v['p10']:.2f} | {v['lower_tail_mean']:.2f} |")
    lines += ['',f"Paired gains (searched minus incumbent): {result['paired_gain']}. Paired simulation standard error: {result['paired_gain_standard_error']:.2f}.",'',
              'Scenarios: 16 equally weighted selection draws (seed 5201), then 32 fresh check draws (seed 62013), used once for acceptance. Same existing Level 4 generator: paired empirical draws from four interior weeks; shared weekly network effects; global demand U(0.8,1.2) and rate U(0.9,1.1) sensitivity shocks. These ranges are assumptions. New random draws from the same assumed model are not independent real-world evidence. Spatial redistribution, rare shocks and unknown organizer scenario probabilities remain untested. Simulation gains cannot establish hidden ranking or the official two-standard-error tie threshold. No global optimality claim.', '',
              'README Appendix B explicitly states historical fleet was ample and no demand was lost. This supersedes the earlier Level 4 report\'s lost-demand caveat. Pickup-window boundaries and completed-only selection still affect historical occupancy estimates.', '',
              'All eight supplied price curves are retained exactly, including non-concave downtown midsize. Own demand is served before adjacent-category upgrades. Category plans are used directly; no split_total_plan.', '',
              f'Cost-only check: {cost_check}. Identical station fleet and transit imply the 12,545.50 saving applies under every demand/revenue scenario. This is a rule-based consequence, not official feedback.', '',
              'Genuine public evidence: Level 3 feasible, 3,929,026.32 (`results/level3_20261113T130545853303Z.official_candidate.result.md`); external Level4robust feasible, 3,940,286.13 (`results/Level4robust.result.md`). No genuine all-truck feedback was found locally. Official status of the new selection is unknown. **Run a new official feedback trial before uploading final.csv.** Nothing was uploaded by this command.', '',
              f"Search: budget {args.seconds}s, at most {args.rounds} rounds; actual {search['runtime_seconds']:.3f}s; stop `{search['stop']}`. Tested move counts: {search['tested']}. Invalid counts: {search['invalid']}. Accepted changes are recorded in JSON.", '',
              'Neighborhood: additions/removals/transfers resized, cancelled, moved in date or station; new actions; coordinated addition replacement by an arriving transfer; truck route/day grouping across categories; destination/origin changes and mode changes. Round-robin enumeration gives each family opportunities before the time cutoff. Full daily coverage and separate network category caps are checked. Existing exact evaluator validates accepted plans and every reported final scenario component.', '',
              f"Selected CSV: `{result['selected']}`. Identical local copy: `submission/final.csv`. SHA-256 of both: `{digest}`. Prior-final backup: `{result['backup']}`.",
              f"Action totals: {result['counts']}. Reloaded CSV: full local validity passed; all 48 scenario component breakdowns reproduced. All 40 Level 0 values matched.", '',
              f"Total pipeline runtime: {result['runtime_seconds']:.2f}s (includes verification outside the search budget). Audit: `{stem}.json`.", '',
              'Reproduce with `.venv/Scripts/python.exe -B -m solution.final_search --seconds 300 --rounds 20`. This writes a fresh plan/report and backs up any existing local final.csv. Wall-clock cutoffs may change the last accepted move; the JSON edit trace preserves this run.']
    with (root/'results'/(stem+'.md')).open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('selected','sha256','incumbent','accepted_after_independent_check','independent_check','paired_gain','counts','runtime_seconds')},indent=2),flush=True)


if __name__=='__main__':main()
