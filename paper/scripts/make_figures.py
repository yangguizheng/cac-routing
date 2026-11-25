"""Generate PDF figures from the time-series and loss summaries.

Usage: python make_figures.py RESULTS OUTPUT_DIRECTORY"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

INF = float('inf')
COL = {'CAC': '#2a78d6', 'MLU': '#eb6834', 'PCDR': '#1baf7a', 'ECMP': '#eda100', 'SP': '#e87ba4', 'Lex': '#008300'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LABEL = {'CAC': 'CAC (ours)', 'MLU': 'MLU-TE', 'PCDR': 'PCDR-like', 'ECMP': 'ECMP', 'SP': 'SP (default)', 'Lex': 'LexCAC (ours)'}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIX Two Text', 'STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5, 'legend.fontsize': 7.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42})


def v(x):
    return INF if x is None else float(x)


def grid(ax, axis='y'):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def load_ts(d):
    files = sorted(glob.glob(os.path.join(d, 'ts', 'ts_*.json')), key=lambda f: int(os.path.basename(f)[3:-5]))
    rows = []
    for f in files:
        t = int(os.path.basename(f)[3:-5])
        for x in json.load(open(f))['rows']:
            rows.append(dict(x, ts=t))
    return rows


def sel(rows, load, cap, rho, delta):
    return sorted([x for x in rows if x['load'] == load and x['cap'] == cap and x['rho'] == rho and x['delta'] == delta],
                  key=lambda x: x['ts'])


def fig_timeseries(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    ymax = 0.0
    for load in (0.05, 0.10):
        for x in sel(rows, load, 50, 1.0, 2):
            for k in ('SP', 'ECMP', 'PCDR', 'MLU', 'CAC'):
                if x.get(f'{k}_cert_bots') is not None:
                    ymax = max(ymax, x[f'{k}_cert_bots'])
    top = math.ceil(ymax / 100.0) * 100 + 100
    band = (top, top * 1.14)
    for ax, load, title in zip(axes, (0.05, 0.10), ('(a) 50 Gb/s offered load', '(b) 100 Gb/s offered load')):
        g = sel(rows, load, 50, 1.0, 2)
        t = np.array([x['ts'] for x in g])
        for k in ('SP', 'ECMP', 'PCDR', 'MLU'):
            y = np.array([v(x[f'{k}_cert_bots']) for x in g])
            y = np.where(np.isinf(y), np.nan, y)
            ax.plot(t, y, color=COL[k], lw=0.8 if k != 'MLU' else 1.2, label=LABEL[k] + (r' ($\Delta$=2)' if k == 'MLU' else ''))
        y = np.array([v(x['CAC_cert_bots']) for x in g])
        fin = np.isfinite(y)
        ax.plot(t[fin], y[fin], ls='none', marker='o', ms=2.4, color=COL['CAC'], markeredgewidth=0,
                label=LABEL['CAC'] + r' ($\Delta$=2)')
        unc = t[np.isinf(y)]
        ax.plot(unc, np.full(len(unc), 0.5 * (band[0] + band[1])), ls='none', marker='o', ms=2.4, color=COL['CAC'],
                markeredgewidth=0)
        ax.axhspan(band[0], band[1], color='#f1f0ec', lw=0, zorder=0)
        ax.set_ylim(0, band[1])
        ax.set_xlim(0, 3600)
        ax.set_xticks(range(0, 3601, 600))
        ax.set_xlabel('Time (s)')
        ax.set_title(title, loc='left', color=INK)
        grid(ax)
        share = len(unc) / len(t)
        ax.text(3540, band[0] - 0.03 * top, f'CAC uncongestible in {share * 100:.0f}% of snapshots', ha='right', va='top',
                fontsize=6.8, color=INK2)
    axes[0].set_ylabel('Certified attack cost\n(compromised terminals)')
    yt = [y for y in range(0, int(top) + 1, 200)]
    axes[0].set_yticks(yt + [0.5 * (band[0] + band[1])])
    axes[0].set_yticklabels([str(y) for y in yt] + [r'$\infty$'])
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=1.8, columnspacing=1.2)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out)
    plt.close(fig)
    return top


def ratio(c, b):
    if b == 0:
        return INF if c > 0 else 1.0
    if math.isinf(b):
        return 1.0 if math.isinf(c) else 0.0
    return c / b


def fig_ratio(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=True)
    xs = np.logspace(0, 1, 400)
    for ax, rho, title in zip(axes, (1.0, 0.9), (r'(a) Congestion target $\rho=1$', r'(b) Congestion target $\rho=0.9$')):
        for delta, ls in ((2, '-'), (1, '--')):
            for load, shade in ((0.05, 0), (0.10, 1)):
                g = [x for x in sel(rows, load, 50, rho, delta) if x.get('CAC_status') == 'ok']
                rs = np.array([ratio(v(x['CAC_cert_bots']), max(v(x[f'{b}_cert_bots']) for b in ('SP', 'ECMP', 'PCDR', 'MLU')))
                               for x in g])
                surv = [(rs >= xx * (1 - 1e-9)).mean() * 100 for xx in xs]
                col = COL['CAC'] if load == 0.05 else '#123f75'
                ax.plot(xs, surv, ls=ls, color=col, lw=1.4,
                        label=f'{int(load * 1000)} Gb/s, ' + r'$\Delta$=' + str(delta))
        ax.axvline(1.5, color=INK2, lw=0.7, ls=':')
        ax.text(1.53, 4, '1.5×', fontsize=6.8, color=INK2)
        ax.set_xscale('log')
        ax.set_xlim(1, 10)
        ax.set_xticks([1, 1.5, 2, 3, 5, 10])
        ax.set_xticklabels(['1', '1.5', '2', '3', '5', '10'])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(0, 102)
        ax.set_xlabel('Improvement factor $x$ over the strongest baseline')
        ax.set_title(title, loc='left', color=INK)
        grid(ax, 'both')
    axes[0].set_ylabel('Snapshots with CAC $\\geq x\\times$\nstrongest baseline (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.2)
    fig.subplots_adjust(wspace=0.08)
    fig.savefig(out)
    plt.close(fig)


def load_attack(d):
    agg = defaultdict(list)
    for f in glob.glob(os.path.join(d, 'attack', 'atk2_*.json')):
        r = json.load(open(f))
        for x in r['rows']:
            agg[(r['load'], r['cap'], x['attacker'], x['rho_t'], x['routing'], x['budget_bots'])].append(x['loss_frac'] * 100)
    return agg


def curve(agg, load, cap, att, rho, routing):
    bs = sorted({k[5] for k in agg})
    return np.array(bs), np.array([np.mean(agg[(load, cap, att, rho, routing, b)]) for b in bs]), \
        np.array([np.max(agg[(load, cap, att, rho, routing, b)]) for b in bs])


def fig_attack(agg, out, att='A', rho=0.9):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35), sharey=False)
    panels = ((0.05, 20, '(a) 50 Gb/s, at most 20 per satellite'), (0.05, 50, '(b) 50 Gb/s, at most 50 per satellite'),
              (0.10, 50, '(c) 100 Gb/s, at most 50 per satellite'))
    series = (('SP', 'SP', 'o'), ('PCDR-like', 'PCDR', 's'), ('MLU-TE[D2]', 'MLU', '^'), ('CAC[D2]', 'CAC', 'D'))
    for ax, (load, cap, title) in zip(axes, panels):
        for routing, key, mk in series:
            b, m, _ = curve(agg, load, cap, att, rho, routing)
            lab = LABEL[key] + (r' ($\Delta$=2)' if key in ('MLU', 'CAC') else '')
            ax.plot(b, m, color=COL[key], marker=mk, ms=3.6, lw=1.3 if key != 'CAC' else 1.7, label=lab,
                    markeredgecolor='white', markeredgewidth=0.5)
        ax.set_xscale('log')
        ax.set_xticks([100, 200, 460, 1000, 2000])
        ax.set_xticklabels(['100', '200', '460', '1000', '2000'])
        ax.minorticks_off()
        ax.set_xlabel('Attacker budget (terminals)')
        ax.set_title(title, loc='left', color=INK, fontsize=7.8)
        ax.set_ylim(bottom=-0.05 * ax.get_ylim()[1])
        grid(ax)
    axes[0].set_ylabel('Legitimate traffic lost (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(out)
    plt.close(fig)


def fig_delta1(agg, out, att='A', rho=0.9):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    panels = ((0.05, 50, '(a) 50 Gb/s, at most 50 per satellite'), (0.10, 50, '(b) 100 Gb/s, at most 50 per satellite'))
    series = (('MLU-TE[D1]', 'MLU', '^', '-', r'MLU-TE ($\Delta$=1)'), ('CAC[D1]', 'CAC', 'D', '-', r'CAC ($\Delta$=1)'),
              ('LexCAC[D1]', 'Lex', 'v', '-', r'LexCAC ($\Delta$=1)'), ('CAC[D2]', 'CAC', 'D', ':', r'CAC ($\Delta$=2), reference'))
    for ax, (load, cap, title) in zip(axes, panels):
        for routing, key, mk, ls, lab in series:
            b, m, _ = curve(agg, load, cap, att, rho, routing)
            ax.plot(b, m, color=COL[key], marker=mk, ms=3.6, ls=ls, lw=1.4, label=lab, markeredgecolor='white',
                    markeredgewidth=0.5, alpha=1.0 if ls == '-' else 0.8)
        ax.set_xscale('log')
        ax.set_xticks([100, 200, 460, 1000, 2000])
        ax.set_xticklabels(['100', '200', '460', '1000', '2000'])
        ax.minorticks_off()
        ax.set_xlabel('Attacker budget (terminals)')
        ax.set_title(title, loc='left', color=INK)
        ax.set_ylim(bottom=-0.1)
        grid(ax)
    axes[0].set_ylabel('Legitimate traffic lost (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.2)
    fig.savefig(out)
    plt.close(fig)


def fig_time(rows, out):
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    for delta, ls in ((1, '--'), (2, '-')):
        s = np.sort([x['CAC_seconds'] for x in rows if x['cap'] == 50 and x['delta'] == delta and x.get('CAC_status') == 'ok'])
        ax.plot(s, np.arange(1, len(s) + 1) / len(s) * 100, color=COL['CAC'], ls=ls, label=r'CAC, $\Delta$=' + str(delta))
        m = np.sort([x['MLU_seconds'] for x in rows if x['cap'] == 50 and x['delta'] == delta])
        ax.plot(m, np.arange(1, len(m) + 1) / len(m) * 100, color=COL['MLU'], ls=ls, label=r'MLU-TE, $\Delta$=' + str(delta))
    ax.axvline(10, color=INK2, lw=0.7, ls=':')
    ax.text(9.2, 8, 'snapshot\ninterval', fontsize=6.5, color=INK2, ha='right')
    ax.set_xscale('log')
    ax.set_xlim(0.008, 15)
    ax.set_ylim(0, 101)
    ax.set_xlabel('Computation time per snapshot (s)')
    ax.set_ylabel('Cumulative share of runs (%)')
    grid(ax, 'both')
    ax.legend(frameon=False, loc='center right', fontsize=6.8)
    fig.savefig(out)
    plt.close(fig)


def main():
    d, figdir = sys.argv[1], sys.argv[2]
    os.makedirs(figdir, exist_ok=True)
    rows = load_ts(d)
    top = fig_timeseries(rows, os.path.join(figdir, 'fig_timeseries.pdf'))
    fig_ratio(rows, os.path.join(figdir, 'fig_ratio.pdf'))
    agg = load_attack(d)
    fig_attack(agg, os.path.join(figdir, 'fig_attack.pdf'))
    fig_attack(agg, os.path.join(figdir, 'fig_attack_skyfall.pdf'), att='S')
    fig_delta1(agg, os.path.join(figdir, 'fig_delta1.pdf'))
    fig_time(rows, os.path.join(figdir, 'fig_time.pdf'))
    print('timeseries y-top', top)


if __name__ == '__main__':
    main()
