"""Level 4 sensitivity study. Run: python -m solution.robustness --seconds 120.

Recorded rentals are a censored demand proxy, not unconstrained demand truth.
Public feedback is deliberately never read. Existing Level 0 formulas stay intact.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from fleetkit import load_instance, write_plan
from solution.forecast import base_revenue, demand_to_come
from solution.fleet import committed_car_days, simulate_fleet, validate_plan, compare_check_values
from solution.evaluate import estimate_profit, revenue_for_served
from solution.planning import _Level1Cache


def available_history(reservations, cutoff):
    """Only completed outcomes observable strictly before the prediction date."""
    return reservations.loc[(reservations.return_date < cutoff) & (reservations.booking_date < cutoff) & reservations.status.eq('completed')].copy()


def occupancy(rentals, grid):
    """Inclusive occupancy on a complete grid, including zero cells."""
    out = np.zeros(len(grid))
    for (station, category), group in rentals.groupby(['pickup_station_id', 'category']):
        mask = grid.station_id.eq(station) & grid.category.eq(category)
        days = grid.loc[mask, 'date'].to_numpy()
        out[mask] = ((days[:, None] >= group.pickup_date.to_numpy()) & (days[:, None] <= group.return_date.to_numpy())).sum(axis=1)
    return out


def history_audit(instance):
    """Two rolling seven-day proxy tests; train outcomes are strictly as-of.

    Ten-day burn-in is a sensitivity assumption motivated by observed lengths,
    not evidence that unobserved earlier rentals could not last longer.
    """
    r = instance.reservations
    first = instance.horizon.first_date.iloc[0]
    start = first - pd.Timedelta(days=56)
    pairs = instance.stations[['station_id']].merge(instance.categories[['category']], how='cross')
    folds = []
    for offset in (28, 21):
        cutoff = first - pd.Timedelta(days=offset)
        train = available_history(r, cutoff)
        dates = pd.date_range(start + pd.Timedelta(days=10), cutoff - pd.Timedelta(days=1))
        grid = pairs.merge(pd.DataFrame({'date': dates}), how='cross')
        grid['occupancy'] = occupancy(train, grid)
        grid['weekday'] = grid.date.dt.day_of_week
        means = grid.groupby(['station_id','category','weekday']).occupancy.mean()
        target = pairs.merge(pd.DataFrame({'date': pd.date_range(cutoff, periods=7)}), how='cross')
        target['weekday'] = target.date.dt.day_of_week
        pred = means.reindex(pd.MultiIndex.from_frame(target[['station_id','category','weekday']])).to_numpy()
        # Reservation dates represent recorded bookings; historical amendments and cancellations are unavailable.
        known = r.loc[r.booking_date < cutoff]
        pred = np.maximum(0, pred - occupancy(known, target))
        truth = occupancy(r.loc[r.booking_date >= cutoff], target)
        model = replace(instance, reservations=train)
        rates = base_revenue(instance=model)
        test = r.loc[r.pickup_date.between(cutoff, cutoff + pd.Timedelta(days=6)) & r.status.eq('completed')].copy()
        test['days'] = (test.return_date-test.pickup_date).dt.days + 1
        actual = test.groupby(['pickup_station_id','category']).agg(revenue=('revenue','sum'),days=('days','sum')).reset_index().rename(columns={'pickup_station_id':'station_id'})
        joined = actual.merge(rates,on=['station_id','category'])
        err = joined.base_revenue_per_car_day - joined.revenue/joined.days
        folds.append(dict(cutoff=str(cutoff.date()), train_rentals=len(train), train_calendar_days=len(dates), test_cells=len(target), test_rentals=len(test), rate_pairs=len(joined), demand_mae=float(np.abs(pred-truth).mean()), demand_rmse=float(np.sqrt(np.mean((pred-truth)**2))), demand_bias=float((pred-truth).mean()), demand_wape=float(np.abs(pred-truth).sum()/truth.sum()), rate_mae=float(err.abs().mean()), rate_day_weighted_mae=float(np.average(err.abs(),weights=joined.days))))
    # Four whole interior weeks, away from both selection edges; paired aggregate occupancy/rate effects.
    blocks = []
    for week in range(4):
        begin = start + pd.Timedelta(days=14+7*week)
        grid = pairs.merge(pd.DataFrame({'date':pd.date_range(begin,periods=7)}),how='cross')
        group = r.loc[r.pickup_date.between(begin,begin+pd.Timedelta(days=6)) & r.status.eq('completed')]
        days = (group.return_date-group.pickup_date).dt.days+1
        blocks.append([float(occupancy(r,grid).sum()),float(group.revenue.sum()/days.sum())])
    blocks = np.asarray(blocks); effects = blocks/blocks.mean(axis=0)
    return dict(folds=folds, weekly_effects=effects.tolist(), observed_max_length=int(((r.return_date-r.pickup_date).dt.days+1).max()), missing=r.isna().sum().astype(int).to_dict(), status_counts=r.status.value_counts().astype(int).to_dict(), pickup_min=str(r.pickup_date.min().date()), booking_min=str(r.booking_date.min().date()), booking_max=str(r.booking_date.max().date()), completed_last_return=str(r.loc[r.status.eq('completed'),'return_date'].max().date()))


def scenarios(demand, base, effects, seed, count):
    """Equal-weight samples: two paired empirical week draws plus sensitivity shocks.

    Global demand U(.8,1.2), global rate U(.9,1.1), all shared across pairs.
    Demand week effects shared across all stations/categories on each day.
    Rates use the mean of the two paired week rate effects. These ranges are
    assumptions, not estimated confidence intervals or organizer probabilities.
    """
    rng = np.random.default_rng(seed)
    effects = np.asarray(effects)
    if effects.shape != (4,2) or not np.isfinite(effects).all() or (effects < 0).any():
        raise ValueError('Expected four finite nonnegative paired week effects')
    week = ((demand.date-demand.date.min()).dt.days//7).clip(upper=1).to_numpy()
    result = []
    for i in range(count):
        chosen = rng.integers(0,4,2)
        dm = effects[chosen,0] * rng.uniform(.8,1.2)
        bm = float(effects[chosen,1].mean()*rng.uniform(.9,1.1))
        d,b = demand.copy(),base.copy()
        d['demand_to_come'] *= dm[week]
        b['base_revenue_per_car_day'] *= bm
        for values in (d.demand_to_come,b.base_revenue_per_car_day):
            if not (np.isfinite(values)&values.ge(0)).all():
                raise ValueError('Invalid scenario values')
        result.append((d,b,dict(name=f'{seed}_{i}',weeks=chosen.tolist(),demand_multipliers=dm.tolist(),rate_multiplier=bm)))
    return result


def price_audit(instance):
    result=[]
    for (station_type,category),group in instance.price_factors.groupby(['station_type','category']):
        group=group.sort_values('served_share')
        x=np.r_[0,group.served_share]; y=np.r_[0,group.served_share*group.price_factor]
        slopes=np.diff(y)/np.diff(x)
        for q in np.r_[x,(x[:-1]+x[1:])/2]:
            assert abs(revenue_for_served(demand=100,served=100*q,base=10,factors=group)-1000*np.interp(q,x,y))<1e-8
        result.append(dict(station_type=station_type,category=category,shares=x.tolist(),total_revenue_per_demand_base=y.tolist(),slopes=slopes.tolist(),nonmonotone=bool((slopes < -1e-9).any()),nonconcave=bool((np.diff(slopes)>1e-9).any())))
    return result


def summary(values):
    a=np.asarray(values)
    return dict(mean=float(a.mean()),std=float(a.std(ddof=1)),minimum=float(a.min()),p10=float(np.quantile(a,.1)),lower_tail_mean=float(np.sort(a)[:max(1,int(np.ceil(.1*len(a))))].mean()))


class ScenarioScore:
    """Reuse exact plan costs; vectorized scenario revenues follow R7-R8 unchanged."""
    def __init__(self,instance,central_d,central_b,samples):
        self.instance=instance; self.d=central_d; self.b=central_b
        self.cache=_Level1Cache(instance,committed_car_days(instance=instance),central_d,central_b)
        c=self.cache
        idx=pd.MultiIndex.from_product([c.stations,c.days,c.categories],names=['station_id','date','category'])
        pairs=pd.MultiIndex.from_product([c.stations,c.categories],names=['station_id','category'])
        self.dem=np.stack([d.set_index(idx.names).demand_to_come.reindex(idx).to_numpy().reshape(c.start.shape) for d,b,m in samples])
        self.base=np.stack([b.set_index(pairs.names).base_revenue_per_car_day.reindex(pairs).to_numpy().reshape(len(c.stations),-1) for d,b,m in samples])
        self.idx=idx
    def score(self,plan):
        exact=estimate_profit(instance=self.instance,plan=plan,demand=self.d,base_revenue=self.b)
        c=self.cache
        f=simulate_fleet(instance=self.instance,plan=plan).set_index(self.idx.names).station_fleet.reindex(self.idx).to_numpy().reshape(c.start.shape)
        capacity=np.maximum(0,c.utilization[:,None,None]*f-c.committed)
        own=np.minimum(self.dem,capacity); served=own.copy()
        served[:,:,:,:-1]+=np.minimum(capacity[None,:,:,1:]-own[:,:,:,1:],self.dem[:,:,:,:-1]-own[:,:,:,:-1])
        revenue=np.zeros(len(self.dem))
        for s in range(len(c.stations)):
            for k in range(len(c.categories)):
                d=self.dem[:,s,:,k]; q=np.divide(served[:,s,:,k],d,out=np.zeros_like(d),where=d>0)
                x,y=c.prices[s,k]
                revenue+=(np.interp(q,x,y)*d*self.base[:,s,k,None]).sum(axis=1)
        return revenue+exact['profit']-exact['new_demand_revenue']


def moves(plan):
    """Fixed add/removes; transfer +/-8, +/-1, delete, mode, departure +/-1 day."""
    for idx,row in plan.loc[plan.action.eq('transfer')].iterrows():
        for quantity in sorted({0,max(0,row.cars-8),row.cars-1,row.cars+1,row.cars+8}):
            p=plan.copy(); p.loc[idx,'cars']=quantity
            yield f'quantity:{idx}:{quantity}',p.loc[p.cars>0].reset_index(drop=True)
        p=plan.copy();p.loc[idx,'mode']='truck' if row['mode']=='driver' else 'driver'
        yield f'mode:{idx}',p
        for delta in (-1,1):
            p=plan.copy();p.loc[idx,'date']+=pd.Timedelta(days=delta)
            yield f'date:{idx}:{delta}',p


def search(instance,plan,score,seconds=120,rounds=2):
    """Best mean-profit move per bounded round; holdout never enters selection."""
    if not np.isfinite(seconds) or seconds<0 or rounds<0:
        raise ValueError('Invalid limits')
    started=perf_counter(); current=plan.copy(); best=float(score.score(current).mean()); trace=[]; tested=invalid=0; reason='round_limit'
    for _ in range(rounds):
        winner=None
        for name,trial in moves(current):
            if perf_counter()-started>=seconds:
                reason='time_limit';break
            tested+=1
            try:
                validate_plan(instance=instance,plan=trial)
                value=float(score.score(trial).mean())
            except ValueError:
                invalid+=1;continue
            if value>best+1e-6:
                best=value;winner=(name,trial)
        if winner is None:
            if reason!='time_limit':reason='no_improvement'
            break
        name,current=winner;trace.append(dict(move=name,training_mean=best))
        if reason=='time_limit':break
    return current,dict(tested=tested,invalid=invalid,trace=trace,stop=reason,seconds=perf_counter()-started,limit_seconds=seconds,max_rounds=rounds)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=120);parser.add_argument('--rounds',type=int,default=2);args=parser.parse_args()
    start=perf_counter();root=Path(__file__).resolve().parents[1]
    source=root/'plans/level3_20261113T130545853303Z.csv'; original=source.read_bytes()
    instance=load_instance(data_dir=root/'data'); fallback=pd.read_csv(source,parse_dates=['date'])
    validate_plan(instance=instance,plan=fallback)
    audit=history_audit(instance); prices=price_audit(instance)
    d=demand_to_come(instance=instance,committed=committed_car_days(instance=instance));b=base_revenue(instance=instance)
    train=scenarios(d,b,audit['weekly_effects'],4101,12);hold=scenarios(d,b,audit['weekly_effects'],4102,24)
    stress=[]
    for name,dm,bm in [('low',.7,.85),('central',1.,1.),('high',1.3,1.15)]:
        dd,bb=d.copy(),b.copy();dd['demand_to_come']*=dm;bb['base_revenue_per_car_day']*=bm
        stress.append((dd,bb,dict(name=name,demand_multiplier=dm,rate_multiplier=bm)))
    scorer=ScenarioScore(instance,d,b,train)
    candidate,search_log=search(instance,fallback,scorer,args.seconds,args.rounds)
    print('Search finished:',search_log,flush=True)
    comparisons={}; exact_audit={}
    for label,samples in [('selection',train),('independent_check',hold),('stress',stress)]:
        fast=ScenarioScore(instance,d,b,samples)
        exact_audit[label]={}
        for name,plan in [('level3',fallback),('level4',candidate)]:
            totals=[estimate_profit(instance=instance,plan=plan,demand=dd,base_revenue=bb) for dd,bb,_ in samples]
            values=[t['profit'] for t in totals]
            assert np.allclose(fast.score(plan),values,rtol=0,atol=1e-6)
            exact_audit[label][name]=totals
            comparisons.setdefault(label,{})[name]=summary(values)
        gains=np.array([t['profit'] for t in exact_audit[label]['level4']])-np.array([t['profit'] for t in exact_audit[label]['level3']])
        comparisons[label]['paired_gain']=summary(gains)
    promising=comparisons['independent_check']['paired_gain']['mean']>1e-6 and bool(search_log['trace'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');stem='level4_'+stamp
    candidate_path=None
    if promising:
        path=root/'plans'/(stem+'.csv')
        with path.open('x'):pass
        write_plan(df_plan=candidate,path=path)
        loaded=pd.read_csv(path,parse_dates=['date']);validate_plan(instance=instance,plan=loaded)
        for label,samples in [('selection',train),('independent_check',hold),('stress',stress)]:
            for (dd,bb,_),expected in zip(samples,exact_audit[label]['level4']):
                actual=estimate_profit(instance=instance,plan=loaded,demand=dd,base_revenue=bb)
                assert all(abs(actual[k]-expected[k])<1e-6 for k in actual)
        candidate_path=path.relative_to(root).as_posix()
    assert source.read_bytes()==original
    assert compare_check_values(instance=instance).empty
    result=dict(history=audit,price_curves=prices,search=search_log,comparison=comparisons,exact_totals=exact_audit,scenario_settings={'selection_seed':4101,'check_seed':4102,'selection_n':12,'check_n':24,'description':scenarios.__doc__},scenario_draws={k:[m for _,_,m in v] for k,v in [('selection',train),('independent_check',hold),('stress',stress)]},fallback=str(source.relative_to(root)),fallback_sha256=hashlib.sha256(original).hexdigest(),candidate=candidate_path,recommendation=candidate_path or str(source.relative_to(root)),runtime_seconds=perf_counter()-start,input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'data').glob('*.csv')})
    if candidate_path:result['candidate_sha256']=hashlib.sha256((root/candidate_path).read_bytes()).hexdigest()
    with (root/'results'/(stem+'.json')).open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    lines=['# Level 4 robustness study','', 'All profits below are local sensitivity estimates. Public scorer feedback was not read or used.', '', 'Regenerate: `.venv/Scripts/python.exe -B -m solution.robustness --seconds 120 --rounds 2`.', '', '## Historical evidence', '', 'Two rolling seven-day recorded-rental proxy tests; training uses only rentals returned and booked strictly before each cutoff. Ten-day left burn-in; targets end at least 14 days before the horizon. Zero cells are included. Historical booking amendments, cancellations, rejected demand and fleet constraints are unavailable. This is not a validated backtest of latent unconstrained demand. Completed-only training still has right-edge completion bias. Observed maximum length 10 days does not establish a bound on missing rentals. Four interior weekly blocks are too few to validate probabilities or tails.', '', '| Cutoff | Train rentals | Train days | Test cells | Test rentals | Rate pairs | Demand MAE | RMSE | Bias | WAPE | Rate MAE |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for f in audit['folds']:lines.append(f"| {f['cutoff']} | {f['train_rentals']} | {f['train_calendar_days']} | {f['test_cells']} | {f['test_rentals']} | {f['rate_pairs']} | {f['demand_mae']:.3f} | {f['demand_rmse']:.3f} | {f['demand_bias']:.3f} | {f['demand_wape']:.3%} | {f['rate_mae']:.3f} |")
    lines+=['','Demand errors are cars on rent per station/category/day; rate errors are revenue per inclusive car-day. Rates compare weighted training estimates with fully observed test pickup cohorts, not daily realized fares. Splits use different available history lengths from the final eight-week model.','', '## Scenarios and prices','',scenarios.__doc__.strip(),'', 'Seeds 4101 (12 selection draws) and 4102 (24 independent draws). All scenarios are equally weighted within their set. Stress cases (demand/rate): low .70/.85, central 1/1, high 1.30/1.15; stress cases are reported separately, not included in the objective. Fixed bookings, fleet, routes, costs and supplied price factors do not change. Four empirical week effects preserve shared network variation; global ranges are documented sensitivity assumptions. Spatial shifts, seasonality, supply censoring and rare shocks are not captured. These draws do not reproduce the organizers\' 30 hidden scenarios.','', 'Every price set was checked at breakpoints and midpoints against exact revenue interpolation. Curves remain in served-share order; no smoothing.']
    for p in prices:lines.append(f"- {p['station_type']}/{p['category']}: non-monotone={p['nonmonotone']}, non-concave={p['nonconcave']}; slopes {np.round(p['slopes'],4).tolist()}.")
    lines+=['','## Plan comparison','', '| Set | Plan | Mean | Std deviation | Minimum | P10 | Worst 10% mean |','|---|---|---:|---:|---:|---:|---:|']
    for label,c in comparisons.items():
        for name,v in c.items():lines.append(f"| {label} | {name} | {v['mean']:.2f} | {v['std']:.2f} | {v['minimum']:.2f} | {v['p10']:.2f} | {v['lower_tail_mean']:.2f} |")
    lines+=['',f"Recommendation for next feedback: `{result['recommendation']}`. Candidate saved: `{candidate_path}`.", '', 'Search holds add/remove actions fixed, trying transfer deletion, quantities +/-1 or +/-8, mode changes and departure +/-1 day. It accepts the best positive mean improvement each round. Validity is enforced; every final scenario total is cross-checked with estimate_profit, including the reloaded CSV. Holdout is used once for acceptance, not iterative tuning; it is independent draws from the same assumed model, not independent real data. No global optimality or hidden-score significance is claimed. README ranks mean profit over 30 hidden scenarios and treats differences under two standard errors as ties; our simulated spread cannot establish that official threshold.', '', f"Search: {search_log}. Total runtime: {result['runtime_seconds']:.2f}s. Exact component totals, scenario draws and hashes are in `{stem}.json`."]
    with (root/'results'/(stem+'.md')).open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(candidate=candidate_path,comparison=comparisons,runtime=result['runtime_seconds'],report=stem),indent=2))

if __name__=='__main__':main()
