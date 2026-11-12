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
checked = 0

def expect(string, label):
    global bad, checked
    checked += 1
    if string not in section:
        bad += 1; print('MISSING', label, string)

core = ['natural_umi', 'controlled_topology_change', 'schedule_gap_reentry']
required = {c: {m: crossing(*curve(c, m)) for m in ['baseline_lslin_lmmse', 'baseline_lmmse_kbest', 'baseline_perf_csi_kbest', 'frozen_cold_k2', 'frozen_cold_k8', 'candidate_reset', 'candidate_persistent']} for c in core}
def gap(c, m):
    return abs(round(required[c][m], 2) - round(required[c]['candidate_persistent'], 2))
for c in core:
    expect(f"{gap(c, 'frozen_cold_k8'):.2f}\\,dB", 'gap to Large NRX')
    expect(f"{gap(c, 'candidate_reset'):.2f}\\,dB", 'memory gain')
    expect(f"{gap(c, 'frozen_cold_k2'):.2f}\\,dB", 'gain over N_it=2')
# Related Work quotes the loss of cutting the Large NRX from N_it=8 to 2, smallest and largest over the conditions.
cut = {c: abs(round(required[c]['frozen_cold_k2'], 2) - round(required[c]['frozen_cold_k8'], 2)) for c in core}
phrase = (f"ranges from {min(cut.values()):.2f}~dB for channel change to "
          f"{max(cut.values()):.2f}~dB for scheduling gaps")
checked += 1
if (min(cut, key=cut.get), max(cut, key=cut.get)) != ('controlled_topology_change', 'schedule_gap_reentry') \
        or phrase not in re.sub(r'\s+', ' ', tex):
    bad += 1; print('MISSING iteration-cut loss in Related Work', cut)
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
    'Large NRX ahead of the proposed NRX in some fixed-SNR cases':
        any(rates[k + ('frozen_cold_k8',)] < rates[k + ('candidate_persistent',)] for k in cases),
    'proposed behind for UMi and gaps, ahead for channel change': required['natural_umi']['candidate_persistent'] > required['natural_umi']['frozen_cold_k8'] and required['schedule_gap_reentry']['candidate_persistent'] > required['schedule_gap_reentry']['frozen_cold_k8'] and required['controlled_topology_change']['candidate_persistent'] < required['controlled_topology_change']['frozen_cold_k8'],
}
for claim, holds in claims.items():
    checked += 1
    if not holds:
        bad += 1; print('CLAIM FAILS', claim)
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
for stratum, label in enumerate(('0-50', '50-175', '175-400')):
    gain = speed_curve('candidate_reset', stratum) - speed_curve('candidate_persistent', stratum)
    expect(f'{gain:.2f}\\,dB', f'memory gain at {label} Hz')
    print('memory gain', label, 'Hz', round(gain, 3))
slow_gap = speed_curve('candidate_reset', 0) - speed_curve('baseline_perf_csi_kbest', 0)
checked += 1
if slow_gap > .1:
    bad += 1; print('CLAIM FAILS no memory matches Perf.-CSI on slow channels', slow_gap)
print('slow channels: no memory minus Perf.-CSI', round(slow_gap, 3))

with np.load(f'{SW}/latency.npz') as data:
    med = {k[len('latency_ms_'):]: float(np.median(data[k])) for k in data.files if k.startswith('latency_ms_')}
for m, v in med.items():
    if f'{v:.2f}' not in text:
        bad += 1; print('LATENCY MISSING', m, v)
print('latency medians', {k: round(v, 2) for k, v in med.items()})
saving = 100 * (1 - med['candidate_persistent'] / med['frozen_cold_k8'])
added = med['candidate_persistent'] - med['frozen_cold_k2']
print('saving % ', round(saving, 1), 'added ms', round(added, 2))
expect(f'{added:.2f}\\,ms more than the Large NRX', 'added latency over N_it=2')
expect(f'{saving:.0f}\\% less than', 'latency saving over N_it=8')
for phrase in ('real-time budget of 1',):
    if phrase not in text.replace('- ', '-') and phrase not in section:
        bad += 1; print('PHRASE MISSING', phrase)

# Abstract and introduction: the summary numbers quoted there, each as the exact phrase in each place it appears.
parts = {'abstract': (r'\begin{abstract}', r'\end{abstract}'),
         'introduction': (r'\section{Introduction}', r'\section{Related Work}')}
parts = {name: re.sub(r'\s+', ' ', tex[tex.index(start):tex.index(end)]) for name, (start, end) in parts.items()}
def span(values):
    return f"{min(values):.2f}--{max(values):.2f}\\,dB"
k8 = {c: gap(c, 'frozen_cold_k8') for c in core}
fast = speed_curve('candidate_reset', 2) - speed_curve('candidate_persistent', 2)
for where, string in (
        (('abstract', 'introduction'), span(cut.values())),
        (('abstract', 'introduction'), span([gap(c, 'candidate_reset') for c in core])),
        (('abstract',), f"within {max(k8['natural_umi'], k8['schedule_gap_reentry']):.2f}\\,dB of the eight-iteration "
                        f"NRX in two of three conditions and {k8['controlled_topology_change']:.2f}\\,dB better in the third"),
        (('abstract',), f"rising to {fast:.2f}\\,dB at Doppler shifts of 175--400"),
        (('introduction',), f"{k8['natural_umi']:.2f}\\,dB and {k8['schedule_gap_reentry']:.2f}\\,dB behind the "
                            f"eight-iteration NRX"),
        (('introduction',), f"{k8['controlled_topology_change']:.2f}\\,dB ahead after a channel change"),
        (('introduction',), f"{span([gap(c, 'frozen_cold_k2') for c in core])} ahead of the same NRX with two iterations"),
        (('introduction',), f"{saving:.0f}\\% lower than that of the eight-iteration NRX"),
):
    for name in where:
        checked += 1
        if string not in parts[name]:
            bad += 1; print('MISSING in', name, string)
for claim, holds in {
        'memory lowers the required SNR in all three conditions':
            all(required[c]['candidate_persistent'] < required[c]['candidate_reset'] for c in core),
        'proposed NRX ahead of the Large NRX with N_it=2 in all three conditions':
            all(required[c]['candidate_persistent'] < required[c]['frozen_cold_k2'] for c in core),
}.items():
    checked += 1
    if not holds:
        bad += 1; print('CLAIM FAILS', claim)
print('RESULT:', 'ALL MATCH' if not bad else f'{bad} MISMATCHES')
