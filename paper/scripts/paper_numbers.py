"""Calculate descriptive statistics for routing certificates, delay spread and loss.

Usage: python paper_numbers.py RESULTS OUTPUT_JSON"""
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

INF = float('inf')
BASES = ('SP', 'ECMP', 'PCDR', 'MLU')


def v(x):
    return INF if x is None else float(x)


def ratio(c, b):
    if b == 0:
        return INF if c > 0 else 1.0
    if math.isinf(b):
        return 1.0 if math.isinf(c) else 0.0
    return c / b


def q(vals, p):
    s = sorted(vals)
    k = min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))
    return s[k]


def enc(x):
    if isinstance(x, float) and math.isinf(x):
        return 'inf'
    return x


def ts_stats(d):
    files = sorted(glob.glob(os.path.join(d, 'ts', 'ts_*.json')), key=lambda f: int(os.path.basename(f)[3:-5]))
    groups = defaultdict(list)
    lp = to = 0
    isl = []
    for f in files:
        r = json.load(open(f))
        lp += r.get('lp', 0)
        to += r.get('lp_timeouts', 0)
        t = int(os.path.basename(f)[3:-5])
        for x in r['rows']:
            x = dict(x, ts=t)
            groups[(x['load'], x['cap'], x['rho'], x['delta'])].append(x)
            if x.get('CAC_isl_max_util') is not None:
                isl.append((x['CAC_isl_max_util'], x['MLU_isl_max_util']))
    out = {'n_snapshots': len(files), 'lp_total': lp, 'lp_timeouts': to,
           'isl_max_util_cac': max(a for a, _ in isl), 'isl_max_util_mlu': max(b for _, b in isl), 'configs': []}
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda x: x['ts'])
        ok = [x for x in g if x.get('CAC_status') == 'ok']
        rec = {'load': key[0], 'cap': key[1], 'rho': key[2], 'delta': key[3], 'n': len(g), 'nofit': len(g) - len(ok)}
        for name in BASES + ('CAC',):
            vals = [v(x[f'{name}_cert_bots']) for x in ok]
            rec[name] = {'median': enc(statistics.median(vals)), 'p5': enc(q(vals, 0.05)), 'p95': enc(q(vals, 0.95)),
                         'min': enc(min(vals)), 'max_finite': enc(max([z for z in vals if not math.isinf(z)], default=float('nan'))),
                         'share_inf': sum(math.isinf(z) for z in vals) / len(vals),
                         'share_zero': sum(z == 0 for z in vals) / len(vals),
                         'share_le20': sum(z <= 20 for z in vals) / len(vals),
                         'mean_hops': statistics.mean(x[f'{name}_mean_hops'] for x in ok),
                         'median_max_util': statistics.median(x[f'{name}_max_util'] for x in ok)}
        # baselines that overload a GSL before any attack, outside the premise D_l <= c of Proposition 1(b)
        rec['overload'] = {}
        for b in BASES:
            ov = [x for x in ok if x[f'{b}_max_util'] > 1 + 1e-9]
            rec['overload'][b] = {'n_overloaded': len(ov),
                                  'n_positive_cert': sum(v(x[f'{b}_cert_bots']) > 0 for x in ov),
                                  'n_above_cac': sum(v(x[f'{b}_cert_bots']) > v(x['CAC_cert_bots']) for x in ov)}
        rec['both_uncongestible'] = sum(math.isinf(v(x['CAC_cert_bots'])) and
                                        math.isinf(max(v(x[f'{b}_cert_bots']) for b in BASES)) for x in ok)
        rs = [ratio(v(x['CAC_cert_bots']), max(v(x[f'{b}_cert_bots']) for b in BASES)) for x in ok]
        rm = [ratio(v(x['CAC_cert_bots']), v(x['MLU_cert_bots'])) for x in ok]
        rec['ratio_best'] = {'p5': enc(q(rs, 0.05)), 'median': enc(statistics.median(rs)), 'p95': enc(q(rs, 0.95)),
                             'min': enc(min(rs)), 'share_ge1': sum(r >= 1 - 1e-9 for r in rs) / len(rs),
                             'share_ge1_5': sum(r >= 1.5 for r in rs) / len(rs), 'share_ge2': sum(r >= 2 for r in rs) / len(rs),
                             'share_ge3': sum(r >= 3 for r in rs) / len(rs), 'share_inf': sum(math.isinf(r) for r in rs) / len(rs)}
        rec['ratio_mlu'] = {'p5': enc(q(rm, 0.05)), 'median': enc(statistics.median(rm))}
        secs = [x['CAC_seconds'] for x in ok]
        msecs = [x['MLU_seconds'] for x in ok]
        rec['cac_seconds'] = {'median': statistics.median(secs), 'p95': q(secs, 0.95), 'max': max(secs)}
        rec['mlu_seconds'] = {'median': statistics.median(msecs), 'max': max(msecs)}
        rec['cert_ge_T'] = all(v(x['CAC_cert_bots']) >= x['CAC_T_bots'] * (1 - 1e-6) for x in ok)
        out['configs'].append(rec)
    return out


def attack_stats(d):
    files = sorted(glob.glob(os.path.join(d, 'attack', 'atk2_*.json')))
    agg = defaultdict(list)
    meta = defaultdict(list)
    snaps = set()
    for f in files:
        r = json.load(open(f))
        snap = int(os.path.basename(f).split('_')[1])
        snaps.add(snap)
        for x in r['rows']:
            agg[(r['load'], r['cap'], x['attacker'], x['rho_t'], x['routing'], x['budget_bots'])].append((snap, x['loss_frac'] * 100))
            if x['attacker'] == 'A' and x['rho_t'] == 0.9 and x['budget_bots'] == 100:
                meta[(r['load'], r['cap'], x['routing'])].append({'snap': snap, 'mean_hops': x['mean_hops'],
                                                                  'cert': enc(v(x['cert_min_bots'])),
                                                                  'isl': x['isl_max_util'], 'base_loss': x['base_loss_frac'] * 100})
    rows = []
    for k, vals in sorted(agg.items(), key=lambda t: str(t[0])):
        losses = [l for _, l in sorted(vals)]
        rows.append({'load': k[0], 'cap': k[1], 'attacker': k[2], 'rho_t': k[3], 'routing': k[4], 'budget': k[5],
                     'mean': statistics.mean(losses), 'max': max(losses), 'per_snapshot': [[s, l] for s, l in sorted(vals)]})
    m = [{'load': k[0], 'cap': k[1], 'routing': k[2], 'mean_hops': statistics.mean(z['mean_hops'] for z in vals),
          'cert_median': enc(statistics.median(v(z['cert']) if z['cert'] != 'inf' else INF for z in vals)),
          'isl_max': max(z['isl'] for z in vals), 'base_loss_mean': statistics.mean(z['base_loss'] for z in vals)}
         for k, vals in sorted(meta.items(), key=lambda t: str(t[0]))]
    return {'snapshots': sorted(snaps), 'rows': rows, 'routing_meta': m}


def lp_size(d):
    f = os.path.join(d, 'lp_size_stats.json')
    if not os.path.exists(f):
        return None
    rows = json.load(open(f))
    out = {}
    for k in rows[0]:
        if k == 'ts':
            continue
        vals = [r[k] for r in rows]
        out[k] = {'min': min(vals), 'median': statistics.median(vals), 'max': max(vals)}
    for dl in (0, 1, 2):
        nv = [2 * r[f'P{dl}'] + r['n_land'] for r in rows]
        nc = [r['n_src'] + 2 * r['n_land'] + r[f'P{dl}'] for r in rows]
        out[f'lp_vars_D{dl}'] = {'median': statistics.median(nv), 'max': max(nv)}
        out[f'lp_rows_D{dl}'] = {'median': statistics.median(nc), 'max': max(nc)}
    return out


def reorder_stats(d):
    """Delay spread of per-packet splitting (probe/reorder_spread.py, results/step6_reorder, Delta=2).

    Shares are weighted by legitimate demand over all source satellites and snapshots.  A landing
    satellite counts as used if its fraction exceeds 1e-6.  Spread = slowest minus fastest one-way
    propagation (ISL path + GSL slant) among the landing satellites a source uses; quantiles are the
    upper edges of 0.1 ms bins, over the split traffic only."""
    rdir = os.path.join(os.path.dirname(os.path.abspath(d)), 'step6_reorder')
    files = glob.glob(os.path.join(rdir, 'rs_*.json'))
    if not files:
        return None
    th = '1e-06'
    agg = {}
    hop_ok = True
    for f in files:
        r = json.load(open(f))
        hop_ok = hop_ok and r['hop_check']
        for x in r['rows']:
            if x['routing'] == 'CAC' and x.get('status') != 'ok':
                continue
            s = x[th]
            a = agg.setdefault((x['routing'], x['load'], x.get('cap'), x.get('rho')),
                               {'hist': [0.0] * len(s['hist']), 'tot': 0.0, 'sl': 0.0, 'sg': 0.0, 'mx': 0.0, 'n': 0})
            a['hist'] = [p + q for p, q in zip(a['hist'], s['hist'])]
            a['tot'] += s['tot']
            a['sl'] += s['single_land']
            a['sg'] += s['single_gw']
            a['mx'] = max(a['mx'], s['max_ms'])
            a['n'] += 1

    def wq(h, q):
        tot, c = sum(h), 0.0
        for i, w in enumerate(h):
            c += w
            if c >= q * tot:
                return round((i + 1) * 0.1, 1)
        return round(len(h) * 0.1, 1)

    rows = []
    for k in sorted(agg, key=str):
        a = agg[k]
        hs = list(a['hist'])
        hs[0] -= a['sl']                     # drop unsplit traffic (spread 0)
        rows.append({'routing': k[0], 'load': k[1], 'cap': k[2], 'rho': k[3], 'snapshots': a['n'],
                     'share_unsplit': a['sl'] / a['tot'], 'share_single_gw': a['sg'] / a['tot'],
                     'split_multi_gw': 1 - (a['sg'] - a['sl']) / (a['tot'] - a['sl']),
                     'split_p50_ms': wq(hs, 0.5), 'split_p95_ms': wq(hs, 0.95), 'max_ms': round(a['mx'], 1)})
    return {'threshold': float(th), 'bin_ms': 0.1, 'hop_check': hop_ok, 'rows': rows}


def infcheck_stats(d):
    """Tolerance check of the uncongestible classification (probe/inf_margin_check.py, results/step7_infcheck).

    'current' is the rule of cac_core.certified_cost (A > R(1+1e-7)+1e-6 Mbps), 'strict' has no tolerance.
    The script rebuilds every routing of the time series; cert_mismatch counts disagreements with the
    certificates reported from ts_*.json."""
    cdir = os.path.join(os.path.dirname(os.path.abspath(d)), 'step7_infcheck')
    files = glob.glob(os.path.join(cdir, 'ic_*.json'))
    if not files:
        return None
    rows, mism = [], 0
    for f in files:
        t = int(os.path.basename(f)[3:-5])
        ref = {(y['load'], y['cap'], y['rho'], y['delta']): y
               for y in json.load(open(os.path.join(d, 'ts', f'ts_{t}.json')))['rows']}
        for x in json.load(open(f))['rows']:
            r = ref[(x['load'], x['cap'], x['rho'], x['delta'])].get(f"{x['routing']}_cert_bots")
            a = x['cert_cur']
            mism += not ((a is None and r is None) or
                         (a is not None and r is not None and abs(a - r) <= 1e-9 * max(1.0, abs(r))))
            rows.append(x)
    cac_unc = [x for x in rows if x['routing'] == 'CAC' and x['unc_cur']]
    flips = [x for x in rows if x['n_flip_gsl'] > 0]
    # Compare finite certificates with and without the numerical overload margin.
    cac_fin = [x for x in rows if x['routing'] == 'CAC' and x['cert_cur'] is not None]
    lower = [x['cert_str'] / x['cert_cur'] for x in cac_fin
             if x['cert_str'] is not None and x['cert_str'] < x['cert_cur'] * (1 - 1e-9)]
    lower.sort()
    return {'n_snapshots': len(files), 'n_routings': len(rows), 'cert_mismatch': mism,
            'cac_finite': len(cac_fin), 'cac_finite_lower_without_tolerance': len(lower),
            'cac_finite_lower_ratio_median': lower[len(lower) // 2] if lower else None,
            'cac_finite_lower_ratio_min': lower[0] if lower else None,
            'cac_uncongestible': len(cac_unc),
            'cac_uncongestible_at_threshold_1e-9': sum(abs(x['min_slack_mbps']) <= 1e-9 for x in cac_unc),
            'max_overshoot_within_tolerance_mbps': max((x['max_overshoot_mbps'] for x in flips), default=0.0),
            'routings_with_such_gsl': sorted({x['routing'] for x in flips}),
            'uncongestible_changes_without_tolerance': sum(x['unc_cur'] != x['unc_str'] for x in rows),
            'gsl_with_zero_spare_capacity': sum(x['n_R_zero'] for x in rows)}


def main():
    d, out = sys.argv[1], sys.argv[2]
    res = {'timeseries': ts_stats(d), 'attack': attack_stats(d), 'lp_size': lp_size(d), 'reorder': reorder_stats(d),
           'infcheck': infcheck_stats(d)}
    json.dump(res, open(out, 'w'), indent=1)
    ts = res['timeseries']
    print('snapshots', ts['n_snapshots'], 'LPs', ts['lp_total'], 'timeouts', ts['lp_timeouts'],
          'ISL max cac/mlu %.3f/%.3f' % (ts['isl_max_util_cac'], ts['isl_max_util_mlu']))
    for c in ts['configs']:
        print(f"L={c['load']} cap={c['cap']} rho={c['rho']} D={c['delta']} nofit={c['nofit']} | "
              + ' '.join(f"{n}:{c[n]['median'] if isinstance(c[n]['median'], str) else round(c[n]['median'], 1)}"
                         f"/h{c[n]['mean_hops']:.2f}/inf{c[n]['share_inf']:.2f}/0:{c[n]['share_zero']:.2f}"
                         for n in BASES + ('CAC',))
              + f" | ratio p5 {c['ratio_best']['p5']} med {c['ratio_best']['median']} >=1.5 {c['ratio_best']['share_ge1_5']:.3f}"
              f" >=2 {c['ratio_best']['share_ge2']:.3f} >=1 {c['ratio_best']['share_ge1']:.3f}"
              f" | t {c['cac_seconds']['median']:.2f}/{c['cac_seconds']['max']:.1f} mlu {c['mlu_seconds']['median']:.3f}"
              f" | util SP {c['SP']['median_max_util']:.2f} MLU {c['MLU']['median_max_util']:.2f} CAC {c['CAC']['median_max_util']:.2f}"
              f" | certgeT {c['cert_ge_T']}")
    print('lp_size', json.dumps(res['lp_size']))


if __name__ == '__main__':
    main()
