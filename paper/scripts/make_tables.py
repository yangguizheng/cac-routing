"""Generate LaTeX tables from time-series and loss summaries.

Usage: python make_tables.py RESULTS OUTPUT_DIRECTORY"""
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
    return s[min(len(s) - 1, max(0, int(round(p * (len(s) - 1)))))]


def f0(x):
    return r'$\infty$' if math.isinf(x) else f'{x:.0f}'


def f2(x):
    return r'$\infty$' if math.isinf(x) else f'{x:.2f}'


def load_rows(d):
    rows = []
    for f in glob.glob(os.path.join(d, 'ts', 'ts_*.json')):
        t = int(os.path.basename(f)[3:-5])
        for x in json.load(open(f))['rows']:
            rows.append(dict(x, ts=t))
    return rows


def cert_table(rows, out):
    lines = [r'\begin{tabular}{@{}rrr rrrrr r rrr@{}}', r'\toprule',
             r' & & & \multicolumn{5}{c}{Median certified cost (terminals)} & CAC & \multicolumn{3}{c}{CAC / strongest baseline} \\',
             r'\cmidrule(lr){4-8}\cmidrule(l){10-12}',
             r'$L$ (Gb/s) & $\rho$ & $\Delta$ & SP & ECMP & PCDR-like & MLU-TE & CAC & uncong. (\%) & p5 & median & $\geq 1.5\times$ (\%) \\',
             r'\midrule']
    for cap in (50, 20):
        lines.append(r'\multicolumn{12}{@{}l}{\emph{At most %d terminals per source satellite}} \\' % cap)
        deltas = (1, 2) if cap == 50 else (2,)
        for load in (0.05, 0.10):
            for rho in (1.0, 0.9):
                for delta in deltas:
                    g = [x for x in rows if x['load'] == load and x['cap'] == cap and x['rho'] == rho and x['delta'] == delta
                         and x.get('CAC_status') == 'ok']
                    med = {k: statistics.median([v(x[f'{k}_cert_bots']) for x in g]) for k in BASES + ('CAC',)}
                    rs = [ratio(v(x['CAC_cert_bots']), max(v(x[f'{b}_cert_bots']) for b in BASES)) for x in g]
                    unc = sum(x['CAC_cert_bots'] is None for x in g) / len(g) * 100
                    share = sum(r >= 1.5 for r in rs) / len(rs) * 100
                    lines.append(f'{int(load * 1000)} & {rho:g} & {delta} & {f0(med["SP"])} & {f0(med["ECMP"])} & {f0(med["PCDR"])} & '
                                 f'{f0(med["MLU"])} & {f0(med["CAC"])} & {unc:.1f} & {f2(q(rs, 0.05))} & {f2(statistics.median(rs))} & '
                                 f'{share:.1f} \\\\')
        if cap == 50:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(out, 'w').write('\n'.join(lines) + '\n')


def cost_table(rows, out):
    """Path length, peak legitimate GSL utilization and computation time, n=50, rho=1."""
    def stats(load, delta, key):
        g = [x for x in rows if x['load'] == load and x['cap'] == 50 and x['rho'] == 1.0 and x['delta'] == delta
             and x.get('CAC_status') == 'ok']
        hops = statistics.mean(x[f'{key}_mean_hops'] for x in g)
        util = statistics.median(x[f'{key}_max_util'] for x in g)
        secs = [x[f'{key}_seconds'] for x in g] if key in ('MLU', 'CAC') else None
        return hops, util, secs
    lines = [r'\begin{tabular}{@{}l rr rr rr@{}}', r'\toprule',
             r' & \multicolumn{2}{c}{Mean path length (hops)} & \multicolumn{2}{c}{Peak GSL utilization} & \multicolumn{2}{c}{Time per snapshot (s)} \\',
             r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}',
             r'Routing & 50 Gb/s & 100 Gb/s & 50 Gb/s & 100 Gb/s & median & max \\', r'\midrule']
    entries = [('SP', 'SP', 2), ('ECMP', 'ECMP', 2), ('PCDR-like', 'PCDR', 2), (r'MLU-TE, $\Delta=1$', 'MLU', 1),
               (r'MLU-TE, $\Delta=2$', 'MLU', 2), (r'CAC, $\Delta=1$', 'CAC', 1), (r'CAC, $\Delta=2$', 'CAC', 2)]
    for name, key, delta in entries:
        h5, u5, s5 = stats(0.05, delta, key)
        h10, u10, s10 = stats(0.10, delta, key)
        if s5 is not None:
            allsec = s5 + s10
            mx = max(allsec)
            tcols = f'{statistics.median(allsec):.2f} & ' + (f'{mx:.2f}' if mx < 1 else f'{mx:.1f}')
        else:
            tcols = r'-- & --'
        lines.append(f'{name} & {h5:.2f} & {h10:.2f} & {u5 * 100:.0f}\\% & {u10 * 100:.0f}\\% & {tcols} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(out, 'w').write('\n'.join(lines) + '\n')


def attack_table(d, out, att):
    agg = defaultdict(list)
    for f in glob.glob(os.path.join(d, 'attack', 'atk2_*.json')):
        r = json.load(open(f))
        for x in r['rows']:
            agg[(r['load'], r['cap'], x['attacker'], x['rho_t'], x['routing'], x['budget_bots'])].append(x['loss_frac'] * 100)
    budgets = sorted({k[5] for k in agg})
    lines = [r'\begin{tabular}{@{}l ' + 'r' * len(budgets) + r'@{}}', r'\toprule',
             r'Routing & ' + ' & '.join(f'{b} terminals' for b in budgets) + r' \\', r'\midrule']
    blocks = ((0.05, 20, r'50 Gb/s, at most 20 terminals per satellite'), (0.05, 50, r'50 Gb/s, at most 50 terminals per satellite'),
              (0.10, 50, r'100 Gb/s, at most 50 terminals per satellite'))
    names = (('SP', 'SP'), ('PCDR-like', 'PCDR-like'), ('MLU-TE[D2]', r'MLU-TE, $\Delta=2$'), ('CAC[D2]', r'CAC, $\Delta=2$'))
    for i, (load, cap, title) in enumerate(blocks):
        if i:
            lines.append(r'\midrule')
        lines.append(r'\multicolumn{%d}{@{}l}{\emph{%s}} \\' % (len(budgets) + 1, title))
        for key, name in names:
            cells = []
            for b in budgets:
                vals = agg[(load, cap, att, 0.9, key, b)]
                cells.append(f'{statistics.mean(vals):.2f} [{max(vals):.2f}]')
            lines.append(name + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(out, 'w').write('\n'.join(lines) + '\n')


def main():
    d, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    rows = load_rows(d)
    cert_table(rows, os.path.join(outdir, 'tab_certificates.tex'))
    cost_table(rows, os.path.join(outdir, 'tab_costs.tex'))
    attack_table(d, os.path.join(outdir, 'tab_attack_adaptive.tex'), 'A')
    attack_table(d, os.path.join(outdir, 'tab_attack_skyfall.tex'), 'S')
    for f in ('tab_certificates.tex', 'tab_costs.tex', 'tab_attack_adaptive.tex'):
        print(open(os.path.join(outdir, f)).read())


if __name__ == '__main__':
    main()
