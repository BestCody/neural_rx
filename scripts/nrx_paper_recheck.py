"""Second pass: recompute the evaluation values from raw per-episode files with separate code and compare with the text."""
import glob, json, math, os, re, subprocess
import numpy as np
SW = 'experiments/20261104_tbler_sweep/results'
text = re.sub(r'\s+', ' ', subprocess.run(['pdftotext', 'paper/build/main.pdf', '-'], capture_output=True, text=True).stdout)
# main.tex pulls the sections in with \input{name}; expand them to check the full text.
tex = re.sub(r'\\input\{([^}]+)\}', lambda m: open(f'paper/{m.group(1)}.tex', encoding='utf8').read(),
             open('paper/main.tex', encoding='utf8').read())
bad = 0

def curve(cond, method):
    table = {}
    for directory in ('sweep', 'sweep_high'):
        files = sorted(glob.glob(f'{SW}/{directory}/{cond}/{method}/episode_*.npz'))
        assert len(files) == 32, (cond, method, directory, len(files))
        for path in files:
            with np.load(path) as data:
                mask = data['active'][16:]
                for k, snr in enumerate(data['snr_db']):
                    fails = int((data['errors'][k, 16:] & mask).sum())
                    entry = table.setdefault(float(snr), {})
                    entry.setdefault(directory, [0, 0])
                    entry[directory][0] += fails
                    entry[directory][1] += int(mask.sum())
    snrs = sorted(table)
    return snrs, [table[x].get('sweep', table[x].get('sweep_high'))[0] / table[x].get('sweep', table[x].get('sweep_high'))[1] for x in snrs]

def crossing(snrs, rates, target=.1):
    # monotone envelope by pooled adjacent violators, then log-TBLER interpolation
    levels = []
    for r in rates:
        levels.append([r, 1])
        while len(levels) > 1 and levels[-2][0] < levels[-1][0]:
            v2, n2 = levels.pop(); v1, n1 = levels.pop()
            levels.append([(v1 * n1 + v2 * n2) / (n1 + n2), n1 + n2])
    fit = [v for v, n in levels for _ in range(n)]
    for i in range(len(fit) - 1):
        if fit[i] >= target >= fit[i + 1]:
            if fit[i] == fit[i + 1]:
                return (snrs[i] + snrs[i + 1]) / 2
            f = (math.log(target) - math.log(fit[i])) / (math.log(fit[i + 1]) - math.log(fit[i])) if fit[i + 1] > 0 else (fit[i] - target) / (fit[i] - fit[i + 1])
            return snrs[i] + f * (snrs[i + 1] - snrs[i])
    return None

section = tex[tex.index(r'\section{Evaluation}'):tex.index(r'\bibliography{references}')]
parts = {'abstract': (r'\begin{abstract}', r'\end{abstract}'),
         'introduction': (r'\section{Introduction}', r'\section{Related Work}')}
parts = {name: re.sub(r'\s+', ' ', tex[tex.index(start):tex.index(end)]) for name, (start, end) in parts.items()}
parts['section'] = re.sub(r'\s+', ' ', section)
checked = 0

def expect(string, label):
    global bad, checked
    checked += 1
    if string not in section:
        bad += 1; print('MISSING', label, string)

def claim(phrase, holds, label, where=('section',)):
    """A qualitative statement: it must appear verbatim where stated, and the data must support it."""
    global bad, checked
    for name in where:
        checked += 1
        if phrase not in parts[name]:
            bad += 1; print('MISSING in', name, label, '|', phrase)
    checked += 1
    if not holds:
        bad += 1; print('CLAIM FAILS', label)

final_claims = []  # statements that need values computed further below

core = ['natural_umi', 'controlled_topology_change', 'schedule_gap_reentry']
required = {c: {m: crossing(*curve(c, m)) for m in ['baseline_lslin_lmmse', 'baseline_lmmse_kbest', 'baseline_perf_csi_kbest', 'frozen_cold_k2', 'frozen_cold_k8', 'candidate_reset', 'candidate_persistent']} for c in core}
def gap(c, m):
    return abs(round(required[c][m], 2) - round(required[c]['candidate_persistent'], 2))
k8 = {c: gap(c, 'frozen_cold_k8') for c in core}
memory_gain = {c: round(required[c]['candidate_reset'], 2)-round(required[c]['candidate_persistent'], 2) for c in core}
claim("performs close to the Large NRX with $N_{\\mathrm{it}}=8$, within 0.3\\,dB in all three conditions",
      max(k8.values()) < .3, 'close to N_it=8')
claim("At the same $N_{\\mathrm{it}}=2$, the Large NRX requires more than 1\\,dB more",
      min(gap(c, 'frozen_cold_k2') for c in core) > 1 and
      all(required[c]['candidate_persistent'] < required[c]['frozen_cold_k2'] for c in core), 'N_it=2 more than 1 dB behind')
# "by 0.3 to 0.4 dB ... with and without IDD": checked with the no-IDD results at the end.
claim("The memory alone reduces the required SNR in all three conditions", min(memory_gain.values()) > 0,
      'memory gain in the introduction', ('introduction',))
# Loss of cutting the Large NRX from N_it=8 to 2; the abstract and introduction quote its range (checked below).
cut = {c: abs(round(required[c]['frozen_cold_k2'], 2) - round(required[c]['frozen_cold_k8'], 2)) for c in core}
MATCH_DB = .1  # "matches": required SNRs within this margin
proposed = {c: round(required[c]['candidate_persistent'], 2) for c in core}
baselines = {
    'outperforms LS + LMMSE in all three conditions':
        all(proposed[c] < round(required[c]['baseline_lslin_lmmse'], 2) for c in core),
    'matches LMMSE + K-Best for UMi and channel change':
        all(gap(c, 'baseline_lmmse_kbest') <= MATCH_DB for c in ('natural_umi', 'controlled_topology_change')),
    'ahead of LMMSE + K-Best for scheduling gaps':
        proposed['schedule_gap_reentry'] < round(required['schedule_gap_reentry']['baseline_lmmse_kbest'], 2),
    'Perf.-CSI + K-Best ahead of all other receivers':
        all(required[c]['baseline_perf_csi_kbest'] < required[c][m] for c in core for m in required[c]
            if m != 'baseline_perf_csi_kbest'),
}

SE = {'first': 'experiments/20261021_package_layout_validation/results/v2_confirmation_005000/errors.npz', 'second': 'experiments/20261021_current_code_reproduction/results/confirmation_005000_seed86000000/errors.npz'}
names = ['historical_tdl_b100', 'historical_tdl_c300', 'natural_umi', 'natural_uma', 'controlled_topology_change', 'schedule_gap_reentry', 'close_los', 'near_far', 'ray_traced_munich']
methods = ['candidate_persistent', 'candidate_reset', 'frozen_cold_k8']
rates, pooled = {}, {}
for ds, path in SE.items():
    data = np.load(path)
    for c, name in enumerate(names[:data['active'].shape[0]]):
        mask = data['active'][c, :, 16:]
        for m in methods:
            e = int((data[f'errors_{m}'][c, 0, :, 16:] & mask).sum()); n = int(mask.sum())
            rates[ds, name, m] = 100 * e / n
# The cases the fixed-SNR paragraph names: the three conditions, close LOS, near-far, and Munich (first dataset only).
cases = [(ds, n) for ds in SE for n in core + ['close_los', 'near_far', 'ray_traced_munich'] if (ds, n, methods[0]) in rates]
unchanged = [('second', 'close_los')]
claims = {
    **baselines,
    'memory lowers TBLER in all named fixed-SNR cases but one':
        all(rates[k + ('candidate_persistent',)] < rates[k + ('candidate_reset',)] for k in cases if k not in unchanged),
    'close LOS unchanged on one dataset':
        all(rates[k + ('candidate_persistent',)] == rates[k + ('candidate_reset',)] for k in unchanged),
}
k8_lead = {k: rates[k + ('candidate_persistent',)] - rates[k + ('frozen_cold_k8',)] for k in cases}
largest = sorted(k8_lead, key=k8_lead.get, reverse=True)[:2]
claim("The Large NRX with $N_{\\mathrm{it}}=8$ stays ahead of the proposed NRX in some cases, most clearly for closely "
      "spaced LOS and near-far UEs on one dataset, and falls clearly behind on the ray-traced channels",
      set(largest) == {('first', 'close_los'), ('first', 'near_far')} and k8_lead[('first', 'ray_traced_munich')] < -5,
      'Large NRX N_it=8 on the fixed-SNR datasets')
for statement, holds in claims.items():
    checked += 1
    if not holds:
        bad += 1; print('CLAIM FAILS', statement)
# Paper tables only; the 3GPP citation "Table~5.1.3.1-1" is not one.
if re.search(r'\\begin\{table|tab:', section):
    bad += 1; print('A table reference remains')
print('values and claims recomputed:', checked)

def speed_curve(method, stratum):
    table = {}
    for directory, skip in (('sweep', 0), ('sweep_high', 1)):
        for path in glob.glob(f'{SW}/{directory}/natural_umi/{method}/episode_*.npz'):
            if int(os.path.basename(path)[8:13]) % 3 != stratum:
                continue
            with np.load(path) as data:
                for k, snr in enumerate(data['snr_db'][skip:], skip):
                    for tb in range(16, 64):
                        for ue in range(2):
                            if data['active'][tb, ue]:
                                bin_ = math.floor(snr + 10 * math.log10(data['channel_power'][tb, ue]))
                                if -6 <= bin_ < 13:
                                    entry = table.setdefault(bin_, [0, 0])
                                    entry[0] += int(data['errors'][k, tb, ue]); entry[1] += 1
    centers = [b + .5 for b in sorted(table)]
    return crossing(centers, [table[b][0] / table[b][1] for b in sorted(table)])

sizes = [sum(int(os.path.basename(p)[8:13]) % 3 == s for p in glob.glob(f'{SW}/sweep/natural_umi/candidate_persistent/episode_*.npz')) for s in range(3)]
expect('Each group contains about 10 sequences', 'Doppler group size phrase')
checked += 1
if not all(10 <= n <= 11 for n in sizes):
    bad += 1; print('CLAIM FAILS about 10 sequences per Doppler group', sizes)
doppler_gain = [speed_curve('candidate_reset', s) - speed_curve('candidate_persistent', s) for s in range(3)]
gain = doppler_gain[2]
claim("The gain is small on slow channels and exceeds 1\\,dB at Doppler shifts above 175\\,Hz",
      doppler_gain[0] < .1 and doppler_gain[2] > 1, 'memory gain by Doppler shift')
claim("its gain grows with the Doppler shift", doppler_gain[0] < doppler_gain[1] < doppler_gain[2],
      'memory gain grows with Doppler', ('introduction',))
claim("The memory contributes most on fast-varying channels", doppler_gain[2] == max(doppler_gain),
      'memory gain largest on fast channels', ('abstract',))
# Why the memory helps on fast channels: Jakes correlation between DMRS symbols at the highest Doppler shift,
# with J0 by numerical integration and the DMRS positions and UE speed read from the radio configuration.
cfg = open('config/nrx_large.cfg', encoding='utf8').read()
setting = lambda key: float(re.search(rf'^{key}\s*=\s*([-\d.e]+)', cfg, re.M).group(1))
assert setting('dmrs_additional_position') == 1  # TS 38.211 Table 6.4.1.1.3-3: type A, 14 symbols, pos1 -> l0 and 11
first, last = int(setting('dmrs_type_a_position')), 11
speed, carrier = setting('max_ut_velocity'), setting('carrier_frequency')
doppler = speed*carrier/299792458.
theta = np.linspace(0, np.pi, 200001)
jakes = lambda tau: np.trapz(np.cos(2*np.pi*doppler*tau*np.sin(theta)), theta)/np.pi
symbol_s = .5e-3/14
across, within = 14-last+first, last-first
words = {5: 'five', 9: 'nine'}
claim(f"Its last DMRS symbol lies only {words[across]} OFDM symbols before that symbol, whereas the second DMRS symbol of "
      f"the current slot lies {words[within]} symbols after it, so under isotropic scattering the older pilot is the more "
      "strongly correlated one", jakes(across*symbol_s) > jakes(within*symbol_s), 'DMRS correlation at the highest Doppler')
expect(f"Doppler shifts up to {round(doppler)}\\,Hz, which corresponds to the maximum UE speed of {speed:.0f}\\,m/s",
       'maximum UE speed')
print('memory gain by Doppler group', [round(g, 3) for g in doppler_gain],
      'correlation at 400 Hz', round(jakes(across*symbol_s), 3), round(jakes(within*symbol_s), 3))
slow_gap = speed_curve('candidate_reset', 0) - speed_curve('baseline_perf_csi_kbest', 0)
checked += 1
if slow_gap > .1:
    bad += 1; print('CLAIM FAILS no memory matches Perf.-CSI on slow channels', slow_gap)
print('slow channels: no memory minus Perf.-CSI', round(slow_gap, 3))

with np.load(f'{SW}/latency.npz') as data:
    med = {k[len('latency_ms_'):]: float(np.median(data[k])) for k in data.files if k.startswith('latency_ms_')}
print('latency medians', {k: round(v, 2) for k, v in med.items()})
saving = 100 * (1 - med['candidate_persistent'] / med['frozen_cold_k8'])
added = med['candidate_persistent'] - med['frozen_cold_k2']
print('saving % ', round(saving, 1), 'added ms', round(added, 2))
per_iteration = (med['frozen_cold_k8'] - med['frozen_cold_k2']) / 6
claim("The proposed NRX is only slightly slower than the Large NRX with $N_{\\mathrm{it}}=2$ and clearly faster than with "
      "$N_{\\mathrm{it}}=8$, at a comparable required SNR, as its additions cost about a third of a CGNN iteration",
      0 < added < .05*med['frozen_cold_k2'] and saving > 15 and .29 <= added/per_iteration <= .38, 'latency')
claim("the median latency of the proposed NRX is lower than that of the eight-iteration NRX", saving > 0,
      'latency in the introduction', ('introduction',))
# Training budget: 17,000 parent updates recorded in the recipe plus the selected checkpoint's final-run updates.
recipe = json.load(open('experiments/20261019_learned_blend_joint_training_6000/inputs/training_recipe.json'))
selected = json.load(open('experiments/20261019_learned_blend_joint_training_6000/results/best.json'))['checkpoint']
final_updates = json.load(open(f'{selected}/manifest.json'))['update']
checked += 1
phrase = (f"the receiver was trained for $1.7\\cdot 10^{{4}}$ steps in earlier versions, which did not yet have the "
          f"combining weight $a$, and then for $5\\cdot 10^{{3}}$ more steps")
initialization = json.load(open('models/learned_blend/initialization.json'))
new_blend = 'blend' in initialization['initialized'] and initialization['training_updates'] == 0
if not (recipe['initializer_parent_updates'] == 17000 and final_updates == 5000 and new_blend
        and recipe['initializer_checkpoint'] == 'models/learned_blend/initial_receiver.weights.h5'
        and phrase in re.sub(r'\s+', ' ', tex)):
    bad += 1; print('MISSING training budget', recipe['initializer_parent_updates'], final_updates)
for phrase in ('real-time budget of 1',):
    if phrase not in text.replace('- ', '-') and phrase not in section:
        bad += 1; print('PHRASE MISSING', phrase)


# ---- Network Decisions, recomputed independently of experiments/20261106_crc_fallback and
# experiments/20261112_early_retransmission (own loaders, own logistic fit with L-BFGS instead of Newton steps).
REAL = ['candidate_persistent', 'candidate_reset', 'frozen_cold_k2', 'frozen_cold_k3', 'frozen_cold_k8',
        'baseline_lslin_lmmse', 'baseline_lmmse_kbest']

def sweep_episodes(cond, methods):
    """Per episode: (active, ue_ids, channel power), merged SNR grid, per-method errors [SNR, TB, UE]."""
    out = []
    for name in sorted(os.path.basename(p) for p in glob.glob(f'{SW}/sweep/{cond}/candidate_persistent/episode_*.npz')):
        errs = {m: {} for m in methods}
        for directory in ('sweep', 'sweep_high'):
            for m in methods:
                with np.load(f'{SW}/{directory}/{cond}/{m}/{name}') as data:
                    for k, snr in enumerate(data['snr_db']):
                        errs[m].setdefault(float(snr), data['errors'][k])
                    base = (data['active'], data['ue_ids'], data['channel_power'])
        snrs = sorted(errs[methods[0]])
        out.append((base, snrs, {m: np.stack([errs[m][s] for s in snrs]) for m in methods}))
    return out

def after_failure(active, ids, errors_at_point):
    """Per scored failure: whether the same UE's previous scheduled TB failed."""
    last, flags = {}, []
    for tb in range(active.shape[0]):
        for pos in range(active.shape[1]):
            if not active[tb, pos]:
                continue
            ue = int(ids[tb, pos])
            if tb >= 16 and errors_at_point[tb, pos]:
                flags.append(last.get(ue) is True)
            last[ue] = bool(errors_at_point[tb, pos])
    return flags

kbest_gain, stuck_share, persistent, ues = {}, {}, [], {}
for c in core:
    eps = sweep_episodes(c, REAL)
    snrs = eps[0][1]
    def req(rule):
        fails, n = np.zeros(len(snrs)), 0
        for (active, ids, power), _, e in eps:
            fails += (rule(e)[:, 16:] & active[16:][None]).sum((1, 2)); n += int(active[16:].sum())
        return crossing(snrs, list(fails/n))
    alone = req(lambda e: e['candidate_persistent'])
    assert round(alone, 2) == round(required[c]['candidate_persistent'], 2)
    kbest_gain[c] = round(round(alone, 2)-round(req(lambda e: e['candidate_persistent'] & e['baseline_lmmse_kbest']), 2), 2)
    point = int(np.argmin(np.abs(np.asarray(snrs)-alone)))
    failed = stuck = 0
    ues[c] = set()
    for index, ((active, ids, power), _, e) in enumerate(eps):
        m = active[16:]
        tb, pos = np.nonzero(e['candidate_persistent'][point, 16:] & m)
        ues[c] |= {(index, int(ids[16+t, p])) for t, p in zip(tb, pos)}
        failed += int((e['candidate_persistent'][point, 16:] & m).sum())
        stuck += int((np.logical_and.reduce([e[r][point, 16:] for r in REAL]) & m).sum())
        persistent += after_failure(active, ids, e['candidate_persistent'][point])
    stuck_share[c] = 100*stuck/failed
claim("decoding failed TBs again with LMMSE + K-Best reduces the required SNR only slightly", max(kbest_gain.values()) < .3,
      'fallback to LMMSE + K-Best without power control')
claim("most errors occur on TBs that no receiver without perfect CSI decodes, and almost all follow an error of the same "
      "UE's previous TB", min(stuck_share.values()) > 85 and 100*np.mean(persistent) > 95, 'errors without power control')
counts = [len(ues[c]) for c in core]
with np.load('experiments/20261106_crc_fallback/results/latency.npz') as data:
    kbest_ratio = np.median(data['latency_ms_baseline_lmmse_kbest'])/np.median(data['latency_ms_candidate_persistent'])
checked += 1
if not (round(kbest_ratio) == 7 and 'about seven times the latency' in section):
    bad += 1; print('CLAIM FAILS LMMSE + K-Best about seven times the latency', kbest_ratio)
print('fallback without power control', kbest_gain, {c: round(v, 1) for c, v in stuck_share.items()},
      'persistence', round(100*np.mean(persistent), 2))

# Power control: operating points, persistence, and the early-retransmission controller.
EV = 'experiments/20261112_early_retransmission/results/evidence_pc'
from scipy.optimize import minimize

def pc_files(cond, episodes=range(32)):
    return [f'{EV}/{cond}/episode_{i:05d}.npz' for i in episodes]

pc_point = {}
for c in core:
    fails, n = 0, 0
    for path in pc_files(c):
        with np.load(path) as data:
            a = data['active'][16:]
            fails = fails+(data['errors'][:, 16:] & a[None]).sum((1, 2)); n += int(a.sum()); grid = data['snr_db']
    pc_point[c] = float(grid[np.argmin(np.abs(grid-crossing(list(grid), list(fails/n))))])

def pc_rows(cond, episodes):
    """Controller inputs, failure, and CRC history of every active UE-TB at the operating point."""
    out = []
    for path in pc_files(cond, episodes):
        with np.load(path) as data:
            k = int(np.flatnonzero(data['snr_db'] == pc_point[cond])[0])
            names = list(data['features'])
            ev, act, err, ids = data['evidence'][k], data['active'], data['errors'][k], data['ue_ids']
            n0 = data['noise_variance'][k]
        last = {}
        for tb in range(act.shape[0]):
            for pos in range(act.shape[1]):
                if not act[tb, pos]:
                    continue
                ue, f = int(ids[tb, pos]), lambda name: float(ev[tb, pos, names.index(name)])
                prev = last.get(ue)
                last[ue] = bool(err[tb, pos])
                if tb < 16:
                    continue
                entries = f('memory_entries')
                logit = lambda p: math.log(min(max(p, 1e-12), 1-1e-12)/(1-min(max(p, 1e-12), 1-1e-12)))
                out.append(dict(
                    failed=bool(err[tb, pos]), prev=prev,
                    previous_failed_flag=float(prev is True), no_previous=float(prev is None),
                    noise=10*math.log10(n0[tb]), current_strength=10*math.log10(f('ls_power')/(4*f('ls_noise'))),
                    has_memory=float(entries > 0), entries=entries, newest_age=f('memory_newest_age'),
                    recent_strength=(10*math.log10(max(f('memory_newest_power'), 1e-12)/max(4*f('memory_newest_noise'), 1e-12))
                                     if entries > 0 else 0.),
                    estimate_strength=10*math.log10(f('estimate_power')/(4*f('ls_noise'))),
                    estimate_error=10*math.log10(f('estimate_error')/f('estimate_power')),
                    llr_bit_error=logit(f('llr_bit_error')), unsatisfied_checks=math.log1p(f('unsatisfied_checks')),
                    failed_code_blocks=f('failed_code_blocks'), posterior_bit_error=logit(f('posterior_bit_error'))))
    return out

pc_all = [r for c in core for r in pc_rows(c, range(32))]
pc_fit = [r for c in core for r in pc_rows(c, range(16))]
pc_test = [r for c in core for r in pc_rows(c, range(16, 32))]
pc_persistent = 100*np.mean([r['prev'] is True for r in pc_all if r['failed']])
claim("However, only about three quarters of the errors now follow an error of the same UE's previous TB",
      70 <= pc_persistent <= 80, 'persistence with power control')

# Receivers and CRC fallback with power control: every receiver decoded the same power-controlled episodes.
RX = 'experiments/20261112_early_retransmission/results/receivers_pc'
pc_req, pc_kbest, pc_stuck, pc_ues, pc_changed = {}, {}, {}, {}, {}
fewer = more = 0
steps = []
for c in core:
    eps = []
    changed = with_entry = 0
    for i in range(32):
        name = f'episode_{i:05d}.npz'
        with np.load(f'{EV}/{c}/{name}') as data:
            e = {'candidate_persistent': data['errors']}
            act, grid, fp, ids, gain = (data['active'], data['snr_db'], data['fingerprint'], data['ue_ids'],
                                        data['power_gain_db'])
        for m in REAL[1:]:
            with np.load(f'{RX}/{c}/{m}/{name}') as data:
                assert np.array_equal(data['fingerprint'], fp) and np.array_equal(data['snr_db'], grid), (c, m, name)
                e[m] = data['errors']
        eps.append((act, e, ids))
        # Power change since the UE's newest memory entry (its last TB within the 8-slot expiry), TBs 17-64.
        newest = {}
        for tb in range(act.shape[0]):
            for pos in range(act.shape[1]):
                if not act[tb, pos]:
                    continue
                ue = int(ids[tb, pos])
                if tb >= 16 and ue in newest and tb-newest[ue][0] <= 8:
                    with_entry += 1
                    step = gain[tb, pos]-newest[ue][1]
                    if abs(step) > 1e-6:
                        changed += 1; steps.append(step)
                newest[ue] = (tb, gain[tb, pos])
    pc_changed[c] = 100*changed/with_entry
    k =int(np.flatnonzero(grid == pc_point[c])[0])
    pc_ues[c] = len({(i, int(ids[16+t, p])) for i, (act, e, ids) in enumerate(eps)
                     for t, p in zip(*np.nonzero(e['candidate_persistent'][k, 16:] & act[16:]))})
    for act, e, ids in eps:
        own, large = (int((e[m][k, 16:] & act[16:]).sum()) for m in ('candidate_persistent', 'frozen_cold_k8'))
        fewer += own < large; more += own > large
    eps = [(act, e) for act, e, ids in eps]
    def pc_crossing(rule):
        fails, n = np.zeros(len(grid)), 0
        for act, e in eps:
            fails += (rule(e)[:, 16:] & act[16:][None]).sum((1, 2)); n += int(act[16:].sum())
        return crossing(list(grid), list(fails/n))
    pc_req[c] = {m: round(pc_crossing(lambda e, m=m: e[m]), 2) for m in REAL}
    pc_kbest[c] = round(pc_req[c]['candidate_persistent']
                        - round(pc_crossing(lambda e: e['candidate_persistent'] & e['baseline_lmmse_kbest']), 2), 2)
    k = int(np.flatnonzero(grid == pc_point[c])[0])
    failed = sum(int((e['candidate_persistent'][k, 16:] & act[16:]).sum()) for act, e in eps)
    stuck = sum(int((np.logical_and.reduce([e[r][k, 16:] for r in REAL]) & act[16:]).sum()) for act, e in eps)
    pc_stuck[c] = 100*stuck/failed
claim("which changed the transmit power since the newest entry for fewer than one in ten TBs with an entry, mostly by "
      "1\\,dB", max(pc_changed.values()) < 10 and np.mean(np.isclose(np.abs(steps), 1.)) > .5,
      'power changes since the newest memory entry')
claim("the errors now stem from many more UEs, and the proposed NRX has fewer errors than the Large NRX with "
      "$N_{\\mathrm{it}}=8$ in more than twice as many sequences as the reverse",
      min(pc_ues.values()) > 1.5*max(counts) and fewer > 2*more, 'UEs and sequences with power control')
claim("although these differences rest on only a few UEs", max(counts) <= 20 and max(counts) < min(pc_ues.values()),
      'few UEs without power control')
print('power control: power changed', {c: round(v, 2) for c, v in pc_changed.items()}, 'UEs', pc_ues,
      'sequences fewer/more', fewer, more)
pc_own = [pc_req[c]['candidate_persistent'] for c in core]
pc_ahead = [round(pc_req[c]['frozen_cold_k8']-pc_req[c]['candidate_persistent'], 2) for c in core]
pc_memory = [round(pc_req[c]['candidate_reset']-pc_req[c]['candidate_persistent'], 2) for c in core]
claim("The proposed NRX then even outperforms the Large NRX with $N_{\\mathrm{it}}=8$ in all three conditions",
      min(pc_ahead) > 0, 'ahead of N_it=8 with power control')
claim("With closed-loop power control, the proposed NRX even outperforms the released eight-iteration NRX", min(pc_ahead) > 0,
      'ahead of N_it=8 with power control, summary', ('abstract', 'introduction'))
claim("and the memory again reduces the required SNR", min(pc_memory) > 0, 'memory gain with power control')
claim("Decoding failed TBs again with LMMSE + K-Best still barely helps, as most errors still occur on TBs that no "
      "receiver without perfect CSI decodes", max(pc_kbest.values()) < .1 and min(pc_stuck.values()) > 80,
      'fallback with power control')
print('receivers with power control', pc_req, 'ahead of N_it=8', pc_ahead, 'memory', pc_memory,
      'K-Best fallback', pc_kbest, {c: round(v, 1) for c, v in pc_stuck.items()})

def logistic(rows, inputs):
    x = np.array([[r[i] for i in inputs] for r in rows]); y = np.array([r['failed'] for r in rows], float)
    mu, sd = x.mean(0), x.std(0); sd[sd == 0] = 1
    z = np.c_[np.ones(len(x)), (x-mu)/sd]
    def objective(w):
        s = z@w
        loss = np.sum(np.logaddexp(0, s)-y*s)+.5*np.sum(w[1:]**2)
        grad = z.T@(1/(1+np.exp(-s))-y)+np.r_[0, w[1:]]
        return loss, grad
    w = minimize(objective, np.zeros(z.shape[1]), jac=True, method='L-BFGS-B',
                 options={'gtol': 1e-12, 'ftol': 1e-15, 'maxiter': 10000}).x
    return lambda rows2: np.c_[np.ones(len(rows2)), (np.array([[r[i] for i in inputs] for r in rows2])-mu)/sd]@w

def threshold_for(scores, rows, share):
    ok = np.array([not r['failed'] for r in rows])
    return np.quantile(scores[ok], 1-share)

failed_test = np.array([r['failed'] for r in pc_test]); first_test = failed_test & np.array([r['prev'] is not True for r in pc_test])
rule_flags = np.array([r['prev'] is True for r in pc_test])
rule_budget = np.mean([r['prev'] is True for r in pc_fit if not r['failed']])
claim("requesting a retransmission whenever the UE's previous TB failed catches about three quarters of the errors in "
      "the test sequences", 70 <= 100*rule_flags[failed_test].mean() <= 80, 'CRC-history rule with power control')
CHECKS = [['previous_failed_flag', 'no_previous', 'noise', 'current_strength', 'has_memory', 'recent_strength', 'entries',
           'newest_age', 'estimate_strength', 'estimate_error']]
CHECKS.append(CHECKS[0]+['llr_bit_error'])
CHECKS.append(CHECKS[1]+['unsatisfied_checks', 'failed_code_blocks', 'posterior_bit_error'])
models = [logistic(pc_fit, inputs) for inputs in CHECKS]
fit_scores, test_scores = [m(pc_fit) for m in models], [m(pc_test) for m in models]
ok_fit = np.array([not r['failed'] for r in pc_fit])
lo, hi = 0., rule_budget
for _ in range(60):
    mid = (lo+hi)/2
    union = np.any([s > threshold_for(s, pc_fit, mid) for s in fit_scores], 0)
    lo, hi = (lo, mid) if union[ok_fit].mean() > rule_budget else (mid, hi)
limits = [threshold_for(s, pc_fit, lo) for s in fit_scores]
first_check = np.full(len(pc_test), -1)
for i in (2, 1, 0):
    first_check[test_scores[i] > limits[i]] = i
caught = first_check >= 0
ctrl, rule, early = (100*caught[failed_test].mean(), 100*rule_flags[failed_test].mean(),
                     100*(first_check[failed_test] == 0).mean())
no_more_alarms = caught[~failed_test].mean() <= rule_flags[~failed_test].mean()
claim(f"the controller catches {ctrl:.0f}\\% of the errors instead of {rule:.0f}\\%, most of them already after channel "
      "estimation, without more unneeded retransmissions", early > ctrl/2 and no_more_alarms, 'early-retransmission controller')
claim("It also catches almost half of the errors that do not follow an error", 40 <= 100*caught[first_test].mean() < 50,
      'first errors caught')
claim("lets the network request most retransmissions before decoding finishes, clearly more than the CRC history alone",
      ctrl > 50 and ctrl-rule > 5, 'controller summary', ('abstract', 'introduction'))
claim("clearly more than the CRC history alone and without more unneeded retransmissions", no_more_alarms,
      'no more unneeded retransmissions, abstract', ('abstract',))
single = {}
for label, inputs in (('any', ['noise', 'current_strength']),
                      ('memory', ['noise', 'current_strength', 'has_memory', 'recent_strength', 'entries', 'newest_age',
                                  'estimate_strength', 'estimate_error']),
                      ('llr', ['noise', 'current_strength', 'has_memory', 'recent_strength', 'entries', 'newest_age',
                               'estimate_strength', 'estimate_error', 'llr_bit_error'])):
    model = logistic(pc_fit, inputs)
    flags = model(pc_test) > threshold_for(model(pc_fit), pc_fit, .01)
    single[label] = 100*flags[failed_test].mean()
claim("the noise level and LS channel power catch about a quarter of the errors after channel estimation, the memory and "
      "the error covariance add little, and the LLRs about double the share",
      20 <= single['any'] <= 30 and single['memory']-single['any'] < 3 and 1.7 <= single['llr']/single['memory'] <= 2.3,
      'evidence by source')
print('power control: operating points', pc_point, 'persistence', round(pc_persistent, 2),
      'rule', round(100*rule_flags[failed_test].mean(), 2), round(100*rule_flags[~failed_test].mean(), 2),
      'controller', round(100*caught[failed_test].mean(), 2), round(100*caught[~failed_test].mean(), 2),
      'first', round(100*caught[first_test].mean(), 2), 'single', {k: round(v, 2) for k, v in single.items()})
# ---- Retrained two-iteration baseline (experiments/20261113_retrained_k2_baseline): own loaders and crossings.
RT = 'experiments/20261113_retrained_k2_baseline'
def rt_rates(directories, cond):
    """Pooled TBLER per SNR over TBs 17-64; the first directory wins on a shared SNR point, which must agree."""
    table = {}
    for directory in directories:
        files = sorted(glob.glob(f'{RT}/results/test/{directory}/{cond}/retrained_k2/episode_*.npz'))
        assert len(files) == 32, (directory, cond, len(files))
        stored = SW if directory != 'power_control' else 'experiments/20261112_early_retransmission/results/receivers_pc'
        counts = {}
        for path in files:
            with np.load(path) as data, np.load(f'{stored}/{directory}/{cond}/frozen_cold_k2/{os.path.basename(path)}'
                                                if directory != 'power_control' else
                                                f'{stored}/{cond}/frozen_cold_k2/{os.path.basename(path)}') as ref:
                assert np.array_equal(data['fingerprint'], ref['fingerprint']) and np.array_equal(data['snr_db'], ref['snr_db'])
                mask = data['active'][16:]
                for k, snr in enumerate(data['snr_db']):
                    entry = counts.setdefault(float(snr), [0, 0])
                    entry[0] += int((data['errors'][k, 16:] & mask).sum()); entry[1] += int(mask.sum())
        for snr, value in counts.items():
            if snr in table:
                assert table[snr] == value, ('shared SNR point disagrees', cond, snr)
            table.setdefault(snr, value)
    snrs = sorted(table)
    return crossing(snrs, [table[s][0]/table[s][1] for s in snrs])
rt = {c: rt_rates(('sweep', 'sweep_high'), c) for c in core}
rt_pc = {c: rt_rates(('power_control',), c) for c in core}
r2 = lambda v: round(v, 2)
narrow = [r2(r2(required[c]['frozen_cold_k2'])-r2(rt[c])) for c in core]
lead = [r2(r2(rt[c])-r2(required[c]['candidate_persistent'])) for c in core]
no_memory_lead = [r2(r2(rt[c])-r2(required[c]['candidate_reset'])) for c in core]
pc_lead = [r2(r2(rt_pc[c])-pc_req[c]['candidate_persistent']) for c in core]
claim("Retraining it with $N_{\\mathrm{it}}=2$ recovers only about half a dB of this gap, so it remains at least 0.6\\,dB "
      "behind the proposed NRX", .4 <= min(narrow) and max(narrow) <= .7 and min(lead) >= .6, 'retrained baseline')
claim("The proposed NRX with no memory already outperforms this retrained baseline, and the memory adds the rest",
      min(no_memory_lead) > 0, 'no memory vs retrained baseline')
claim("it remains clearly ahead of the Large NRX retrained with $N_{\\mathrm{it}}=2$", min(pc_lead) > .5,
      'retrained baseline with power control')
claim("performs close to the eight-iteration NRX and at least 0.6\\,dB better than the same NRX retrained with two "
      "iterations on the same data", max(k8.values()) < .3 and min(lead) >= .6, 'fair comparison, abstract', ('abstract',))
claim("it performs close to the eight-iteration NRX for urban microcell (UMi) channels, channel changes, and scheduling "
      "gaps, and it clearly outperforms the same NRX retrained with two iterations on the same data",
      max(k8.values()) < .3 and min(lead) > .5, 'fair comparison, introduction', ('introduction',))
# Budget: the baseline's selected checkpoint lies within the 22,000 updates the paper states, and training ran them all.
best_rt = json.load(open(f'{RT}/results/best.json'))
history = sum(1 for _ in open('models/retrained_k2_baseline_20261113/history.jsonl'))
checked += 1
if not (history == 22000 and best_rt['update'] <= 22000 and
        'retrain the Large NRX with $N_{\\mathrm{it}}=2$ for the same $2.2\\cdot 10^{4}$ steps' in section):
    bad += 1; print('MISSING retrained baseline budget', history, best_rt['update'])
# The memory adds the rest: the no-memory lead and the memory gain sum to the remaining gap in every condition.
checked += 1
if not all(abs(no_memory_lead[i]+gap(c, 'candidate_reset')-lead[i]) < .011 for i, c in enumerate(core)):
    bad += 1; print('CLAIM FAILS the memory adds the rest', no_memory_lead, lead)
for statement, holds in {
        'retraining improves the two-iteration NRX in every condition': min(narrow) > 0,
        'the proposed NRX beats the retrained baseline in every condition, with and without power control':
            min(lead) > 0 and min(pc_lead) > 0,
        'the retrained baseline stays behind the Large NRX with N_it=8': all(rt[c] > required[c]['frozen_cold_k8'] for c in core),
}.items():
    checked += 1
    if not holds:
        bad += 1; print('CLAIM FAILS', statement)
# Cost of the iteration cut even after retraining (abstract and introduction): the retrained N_it=2 baseline vs N_it=8.
retrained_cut = [r2(r2(rt[c])-r2(required[c]['frozen_cold_k8'])) for c in core]
for phrase, where in (("which costs up to 1\\,dB even after retraining for two iterations", 'abstract'),
                      ("of 10\\% by up to 1\\,dB, even when the NRX is retrained for two iterations", 'introduction')):
    claim(phrase, .9 <= max(retrained_cut) <= 1. and min(retrained_cut) > .5, f'retrained iteration cut, {where}', (where,))

# ---- Effective SNR (SNR + channel power of the UE in dB), pooled over all sequences, 1-dB bins as for Fig. 3.
NI = 'experiments/20261114_no_idd_ablation/results/test'
def effective_curve(cond, method, base=SW):
    table = {}
    for directory, skip in (('sweep', 0), ('sweep_high', 1)):
        files = sorted(glob.glob(f'{SW}/{directory}/{cond}/candidate_persistent/episode_*.npz'))
        assert len(files) == 32
        for path in files:
            other = f'{base}/{directory}/{cond}/{method}/{os.path.basename(path)}'
            if not os.path.isfile(other):
                return None
            with np.load(path) as ref, np.load(other) as data:
                assert np.array_equal(ref['snr_db'], data['snr_db'])
                power = 10*np.log10(ref['channel_power'])
                for k in range(skip, len(data['snr_db'])):
                    for tb in range(16, 64):
                        for ue in range(2):
                            if ref['active'][tb, ue]:
                                bin_ = math.floor(data['snr_db'][k] + power[tb, ue])
                                if -10 <= bin_ < 30:
                                    entry = table.setdefault(bin_, [0, 0])
                                    entry[0] += int(data['errors'][k, tb, ue]); entry[1] += 1
    bins = sorted(table)
    return [b + .5 for b in bins], [table[b][0]/table[b][1] for b in bins]
nrx = {'frozen_cold_k2': SW, 'frozen_cold_k8': SW, 'retrained_k2': f'{RT}/results/test', 'candidate_reset': SW,
       'candidate_persistent': SW}
nrx.update({'candidate_persistent_no_idd': NI, 'candidate_reset_no_idd': NI})
eff = {c: {m: effective_curve(c, m, b) for m, b in nrx.items()} for c in core}
eff_req = {c: {m: crossing(*eff[c][m]) for m in nrx} for c in core}
drop = {c: {m: crossing(*eff[c][m], .05)-crossing(*eff[c][m], .8) for m in nrx} for c in core}
claim("against which the TBLER of every NRX falls from 80\\% to 5\\% within 4\\,dB",
      max(v for c in core for v in drop[c].values()) <= 4, 'effective SNR: steep curves')
eff_lead = [eff_req[c]['retrained_k2']-eff_req[c]['candidate_persistent'] for c in core]
eff_k8 = [eff_req[c]['frozen_cold_k8']-eff_req[c]['candidate_persistent'] for c in core]
eff_memory = [eff_req[c]['candidate_reset']-eff_req[c]['candidate_persistent'] for c in core]
claim("In effective SNR, the proposed NRX requires at least 0.7\\,dB less than the retrained Large NRX and matches or "
      "outperforms the Large NRX with $N_{\\mathrm{it}}=8$", min(eff_lead) >= .7 and min(eff_k8) > -.1,
      'effective SNR: retrained baseline and N_it=8')
claim("and the required effective SNR by more than half a dB", min(eff_memory) > .5, 'effective SNR: memory gain')
print('effective SNR required', {c: {m: round(v, 2) for m, v in eff_req[c].items()} for c in core})
print('effective SNR: lead over retrained', [round(v, 2) for v in eff_lead], 'vs k8', [round(v, 2) for v in eff_k8],
      'memory', [round(v, 2) for v in eff_memory], 'widest 80-5% drop', round(max(v for c in core for v in drop[c].values()), 2))

# ---- Channel change: one UE whose new channel is very weak fails in every TB after the change, for every receiver.
everyone = ['baseline_lslin_lmmse', 'baseline_lmmse_kbest', 'baseline_perf_csi_kbest', 'frozen_cold_k2', 'frozen_cold_k8',
            'candidate_reset', 'candidate_persistent']
floor_ok, culprit = True, None
for m in everyone:
    files = sorted(glob.glob(f'{SW}/sweep_high/controlled_topology_change/{m}/episode_*.npz'))
    fails = np.array([[(np.load(f)['errors'][k, 16:] & np.load(f)['active'][16:]).sum() for f in files]
                      for k in range(len(np.load(files[0])['snr_db']))])
    blocks = sum(int(np.load(f)['active'][16:].sum()) for f in files)
    floor_ok &= fails[-1].sum()/blocks >= .009
with np.load(f'{SW}/sweep_high/controlled_topology_change/candidate_persistent/episode_00024.npz') as data:
    after = data['active'][32:, 1]
    all_power = np.concatenate([10*np.log10(np.load(f)['channel_power'][16:][np.load(f)['active'][16:]])
                                for f in glob.glob(f'{SW}/sweep_high/controlled_topology_change/candidate_persistent/*.npz')])
    weak = np.median(10*np.log10(data['channel_power'][32:, 1])) - np.median(all_power)
for m in everyone:
    with np.load(f'{SW}/sweep_high/controlled_topology_change/{m}/episode_00024.npz') as data:
        floor_ok &= bool(data['errors'][:, 32:, 1][:, after].mean(1).min() >= .9)
claim("For channel change, TBLER stays at about 1\\% or more for all receivers because the new channel of one UE is about "
      "18\\,dB weaker than the median, so almost all of its TBs after the change fail", floor_ok and -18.5 <= weak <= -17.5,
      'channel-change floor')
print('channel-change floor: weak UE', round(float(weak), 1), 'dB below the median TB')

# ---- Complexity: real multiplications per slot (experiments/20261114_no_idd_ablation/results/complexity.json),
# with the CGNN iteration cross-checked against its weights times the grid positions of both UEs.
cx = json.load(open('experiments/20261114_no_idd_ablation/results/complexity.json'))
grid_positions = cx['grid']['subcarriers']*cx['grid']['ofdm_symbols']*cx['grid']['users']
cgnn = cx['cgnn_iteration']['flops']/2
checked += 1
if abs(cgnn/(cx['weights']['cgnn_iteration']*grid_positions)-1) > .02:
    bad += 1; print('CLAIM FAILS CGNN multiplications vs weights x positions', cgnn, cx['weights']['cgnn_iteration']*grid_positions)
filt = cx['channel_filter_counted']['real_mult']
idd = cx['idd_detector']['real_mult']+cx['idd_combining']['real_mult']
claim("one CGNN iteration requires about $2.2\\cdot 10^{9}$ real multiplications per slot, whereas the channel filter with "
      "its error covariance requires $4\\cdot 10^{7}$ and IDD $2.2\\cdot 10^{8}$, together about an eighth of an iteration",
      round(cgnn/1e9, 1) == 2.2 and round(filt/1e7) == 4 and round(idd/1e8, 1) == 2.2 and .11 <= (filt+idd)/cgnn <= .14,
      'multiplications per slot')
print('multiplications per slot: CGNN iteration', f'{cgnn:.3g}', 'filter', f'{filt:.3g}', 'IDD', f'{idd:.3g}',
      'share', round((filt+idd)/cgnn, 3))
# ---- Proposed NRX without IDD (experiments/20261114_no_idd_ablation): same weights, LLRs L straight to the
# 20-iteration LDPC decoder of all other receivers. Same sequences as the sweep (fingerprints and SNR grids).
def ni_required(cond, method):
    table = {}
    for directory in ('sweep', 'sweep_high'):
        files = sorted(glob.glob(f'{NI}/{directory}/{cond}/{method}/episode_*.npz'))
        assert len(files) == 32, (cond, method, directory, len(files))
        for path in files:
            with np.load(path) as data, np.load(f'{SW}/{directory}/{cond}/candidate_persistent/{os.path.basename(path)}') as ref:
                assert np.array_equal(data['fingerprint'], ref['fingerprint']) and np.array_equal(data['snr_db'], ref['snr_db'])
                assert np.array_equal(data['active'], ref['active']) and int(data['bp_iterations']) == 20
                mask = data['active'][16:]
                for k, snr in enumerate(data['snr_db']):
                    entry = table.setdefault(float(snr), {}).setdefault(directory, [0, 0])
                    entry[0] += int((data['errors'][k, 16:] & mask).sum()); entry[1] += int(mask.sum())
    snrs = sorted(table)
    return crossing(snrs, [table[x].get('sweep', table[x].get('sweep_high'))[0]/table[x].get('sweep', table[x].get('sweep_high'))[1]
                           for x in snrs])
ni = {c: {m: ni_required(c, f'{m}_no_idd') for m in ('candidate_persistent', 'candidate_reset')} for c in core}
ni_memory = [r2(r2(ni[c]['candidate_reset'])-r2(ni[c]['candidate_persistent'])) for c in core]
ni_lead = [r2(r2(rt[c])-r2(ni[c]['candidate_persistent'])) for c in core]
ni_no_memory_lead = [r2(r2(rt[c])-r2(ni[c]['candidate_reset'])) for c in core]
ni_eff_memory = [eff_req[c]['candidate_reset_no_idd']-eff_req[c]['candidate_persistent_no_idd'] for c in core]
claim("The memory reduces the required SNR in all three conditions, by 0.3 to 0.4\\,dB, and the required effective SNR by "
      "more than half a dB, with and without IDD",
      all(round(v, 1) in (.3, .4) for v in list(memory_gain.values())+ni_memory) and min(eff_memory+ni_eff_memory) > .5,
      'memory gain with and without IDD')
claim("We also evaluate both proposed versions without IDD, i.e., with their LLRs $L$ passed to the LDPC decoder. With IDD, "
      "they use 18 BP iterations, all others 20", True, 'no-IDD setup (20 BP iterations asserted per file)')
claim("Without IDD, which leaves the LDPC decoder unchanged, the proposed NRX still outperforms the retrained baseline in all "
      "three conditions, but its no-memory version does not", min(ni_lead) > 0 and max(ni_no_memory_lead) < 0,
      'no IDD vs retrained baseline')
claim("Without IDD, the additions cost less than 2\\% of an iteration", filt/cgnn < .02, 'filter share of an iteration')
claim("its gain remains without IDD, which leaves the channel decoder unchanged", min(ni_memory) > 0 and min(ni_eff_memory) > 0,
      'no-IDD memory gain, abstract', ('abstract',))
claim("Without IDD, which leaves the low-density parity-check (LDPC) decoder unchanged, the memory gain remains, and the NRX "
      "still outperforms the retrained one", min(ni_memory) > 0 and min(ni_lead) > 0, 'no IDD, introduction', ('introduction',))
claim("and a memory gain that remains without IDD", min(ni_memory) > 0, 'no IDD, conclusion')
print('without IDD', {c: {m: round(v, 2) for m, v in ni[c].items()} for c in core}, 'memory', ni_memory,
      'effective memory', [round(v, 2) for v in ni_eff_memory], 'lead over retrained', ni_lead,
      'no-memory lead', ni_no_memory_lead, 'filter share', round(filt/cgnn, 4))
print('retrained baseline', {c: round(v, 2) for c, v in rt.items()}, 'power control', {c: round(v, 2) for c, v in rt_pc.items()},
      'narrow', narrow, 'lead', lead, 'no memory lead', no_memory_lead, 'pc lead', pc_lead)
print('RESULT:', 'ALL MATCH' if not bad else f'{bad} MISMATCHES')
