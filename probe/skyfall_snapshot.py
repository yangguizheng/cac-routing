"""Rebuild one SKYFALL (NDSS 2025) +Grid snapshot and keep the accepted legal demand per block.

Port of SKYFALL/skyfall/generate_flow_grid.py (MIT, commit 17c0888).  The logic, constants and
routing are kept as in the original; differences: the sampler is seeded, the per-block accepted
traffic is saved (the original keeps it only in memory), and the per-source landing path is
precomputed once instead of being walked for every sampled flow.

usage: python skyfall_snapshot.py <skyfall_dir> <timeslot> <out_dir> [seed]
"""
import json
import math
import os
import random
import sys

import numpy as np

RADIUS = 6371.0
FLOW = 0.5            # Mbps per sampled flow
N_SAMPLES = 2_000_000
CAP_GSL = 4096.0      # GSL up/down capacity (Mbps)
CAP_ISL = 20480.0
BOUND = 3.743333 * 299792.458 * 0.001   # max slant range (km), as in SKYFALL


def car(lat, lng, h):
    lat = np.radians(lat)
    lng = np.radians(lng)
    r = RADIUS + h
    return np.stack([r * np.cos(lat) * np.cos(lng), r * np.cos(lat) * np.sin(lng), r * np.sin(lat)], axis=-1)


def main():
    skydir, ts, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261123
    nosample = len(sys.argv) > 5 and sys.argv[5] == 'nosample'   # expected-demand runs need no sampling
    cfg = json.load(open(os.path.join(skydir, 'config.json')))
    n_orb, n_per = cfg['# of orbit'], cfg['# of satellites']
    inc = cfg['Inclination']
    n_sat = n_orb * n_per

    lla = np.loadtxt(os.path.join(skydir, cfg['Name'], 'sat_lla', f'{ts}.txt'), delimiter=',')
    sat = car(lla[:, 0], lla[:, 1], lla[:, 2])

    lats, lons = [], []
    for lat in range(inc, -inc, -1):
        for lon in range(-180, 180):
            lats.append(lat - 0.5)
            lons.append(lon + 0.5)
    user = car(np.array(lats), np.array(lons), 0.0)
    n_blk = len(lats)

    pop = []
    with open(os.path.join(skydir, 'skyfall', 'starlink_count.txt')) as fr:
        lines = fr.readlines()
        for row in range(90 - inc, 90 + inc):
            pop.extend([float(x) for x in lines[row].split(' ')[:-1]] + [0])
    pop = np.array(pop)
    assert len(pop) == n_blk, (len(pop), n_blk)

    # block -> access satellite (nearest within BOUND)
    ucs = np.full(n_blk, -1, dtype=int)
    for b in range(n_blk):
        d = np.sqrt(((sat - user[b]) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] <= BOUND:
            ucs[b] = j

    gs_info = json.load(open(os.path.join(skydir, 'skyfall', 'GS.json')))
    gs_keys = list(gs_info.keys())
    gs = car(np.array([float(gs_info[k]['lat']) for k in gs_keys]),
             np.array([float(gs_info[k]['lng']) for k in gs_keys]), 0.0)
    scg = np.full(n_sat, -1, dtype=int)
    for s in range(n_sat):
        d = np.sqrt(((gs - sat[s]) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] <= BOUND:
            scg[s] = j

    # SKYFALL next-hop rule (find_next_sat): toward the nearest landing satellite in grid hops,
    # inter-orbit move first when orbits differ, ties keep the first landing satellite found.
    landing = [i for i in range(n_sat) if scg[i] != -1]
    nxt = np.full(n_sat, -1, dtype=int)
    for o in range(n_orb):
        for k in range(n_per):
            si = o * n_per + k
            if scg[si] != -1:
                nxt[si] = si
                continue
            best, nh = int(n_per / 2) + int(n_orb / 2), -1
            for it in landing:
                io, ik = it // n_per, it % n_per
                od = abs(io - o) if abs(io - o) <= n_orb // 2 else n_orb - abs(io - o)
                kd = abs(ik - k) if abs(ik - k) <= n_per // 2 else n_per - abs(ik - k)
                if od + kd >= best:
                    continue
                best = od + kd
                if io == o:
                    if ik > k:
                        if (ik - k) <= (n_per - (ik - k)):
                            nh = si + 1
                        elif k == 0:
                            nh = si + n_per - 1
                        else:
                            nh = si - 1
                    else:
                        if (k - ik) <= (n_per - (k - ik)):
                            nh = si - 1
                        elif k == n_per - 1:
                            nh = si - n_per + 1
                        else:
                            nh = si + 1
                else:
                    if io > o:
                        if (io - o) <= (n_orb - (io - o)):
                            nh = si + n_per
                        elif o == 0:
                            nh = si + n_per * (n_orb - 1)
                        else:
                            nh = si - n_per
                    else:
                        if (o - io) <= (n_orb - (o - io)):
                            nh = si - n_per
                        elif o == n_orb - 1:
                            nh = si - n_per * (n_orb - 1)
                        else:
                            nh = si + n_per
            nxt[si] = nh

    def link_seq(i, j):
        oi, oj = i // n_per, j // n_per
        if oi == oj and (j == i + 1 or j == i - (n_per - 1)):
            return i * 6
        if j == (i + n_per) % n_sat:
            return i * 6 + 1
        if oi == oj and (j == i - 1 or j == i + (n_per - 1)):
            return i * 6 + 2
        if j == (i - n_per + n_sat) % n_sat:
            return i * 6 + 3
        return -1

    # precompute the ISL link list and landing satellite of every source satellite
    land_of = np.full(n_sat, -1, dtype=int)
    isl_path = [None] * n_sat
    for s in range(n_sat):
        if nxt[s] == -1:
            continue
        links, cur, guard = [], s, 0
        while nxt[cur] != cur:
            lk = link_seq(cur, nxt[cur])
            assert lk != -1, (cur, nxt[cur])
            links.append(lk)
            cur = nxt[cur]
            guard += 1
            assert guard < 200
        land_of[s] = cur
        isl_path[s] = links

    blocks = [b for b in range(n_blk) if pop[b] != 0]
    weights = [math.ceil(pop[b]) for b in blocks]
    cum = np.cumsum(weights)
    total_w = int(cum[-1])

    rng = random.Random(seed)
    link = np.zeros(n_sat * 6)
    down = np.zeros(n_sat)
    up = np.zeros(n_sat)
    acc_blk = np.zeros(n_blk)
    for _ in range(0 if nosample else N_SAMPLES):
        r = rng.randint(1, total_w)
        idx = int(np.searchsorted(cum, r))          # same index as SKYFALL's bisection
        b = blocks[idx]
        s = ucs[b]
        if s == -1:
            continue
        l = land_of[s]
        if l == -1:
            continue
        if down[l] + FLOW < CAP_GSL:                # SKYFALL accepts iff new downlink < capacity
            down[l] += FLOW
            up[s] += FLOW
            for lk in isl_path[s]:
                link[lk] += FLOW
            acc_blk[b] += FLOW

    os.makedirs(out, exist_ok=True)
    np.savez_compressed(os.path.join(out, f'snap_{ts}.npz'),
                        sat_lla=lla, ucs=ucs, scg=scg, nxt=nxt, land_of=land_of, pop=pop,
                        acc_blk=acc_blk, down=down, up=up, link=link,
                        n_orb=n_orb, n_per=n_per, inc=inc, seed=seed)
    print(f'ts={ts} landing={len(landing)} accepted={acc_blk.sum():.0f} Mbps '
          f'down_max={down.max():.0f} n_full={(down > CAP_GSL - 1).sum()} '
          f'n_half={(down > CAP_GSL / 2).sum()} isl_max={link.max():.0f}')


if __name__ == '__main__':
    main()
