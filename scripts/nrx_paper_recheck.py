"""Second pass: recompute the evaluation values from raw per-episode files with separate code and compare with the text."""
import glob, json, math, os, re, subprocess
import numpy as np
SW = 'experiments/20261104_tbler_sweep/results'
text = re.sub(r'\s+', ' ', subprocess.run(['pdftotext', 'paper/build/main.pdf', '-'], capture_output=True, text=True).stdout)
tex = open('paper/main.tex', encoding='utf8').read()
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
    expect(f"{required[c]['candidate_persistent']:.2f}\\,dB", 'proposed required SNR')
    expect(f"{gap(c, 'frozen_cold_k8'):.2f}\\,dB", 'gap to Large NRX')
    expect(f"{gap(c, 'candidate_reset'):.2f}\\,dB", 'memory gain')
for c in ('natural_umi', 'controlled_topology_change', 'schedule_gap_reentry'):
    expect(f"{gap(c, 'frozen_cold_k2'):.2f}\\,dB", 'gain over N_it=2')
expect(f"{min(gap(c, 'baseline_lslin_lmmse') for c in core):.2f}\\,dB", 'minimum gap to LS + LMMSE')
expect(f"{min(gap(c, 'baseline_perf_csi_kbest') for c in core):.2f}\\,dB to {max(gap(c, 'baseline_perf_csi_kbest') for c in core):.2f}\\,dB", 'range of gaps to Perf.-CSI')
expect(f"{gap('natural_umi', 'baseline_lmmse_kbest'):.2f}\\,dB", 'match to LMMSE + K-Best')
assert gap('controlled_topology_change', 'baseline_lmmse_kbest') <= gap('natural_umi', 'baseline_lmmse_kbest')
expect(f"{gap('schedule_gap_reentry', 'baseline_lmmse_kbest'):.2f}\\,dB", 'lead over LMMSE + K-Best')

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
            if name in core:
                t = pooled.setdefault((ds, m), [0, 0]); t[0] += e; t[1] += n
for key, (e, n) in pooled.items():
    expect(f'{100 * e / n:.2f}\\%', f'pooled {key}')
for m in methods:
    expect(f"{rates['first', 'ray_traced_munich', m]:.2f}\\%", f'Munich {m}')
for ds in SE:
    for m in methods[1:]:
        expect(f"{abs(100 * pooled[ds, m][0] / pooled[ds, m][1] - 100 * pooled[ds, 'candidate_persistent'][0] / pooled[ds, 'candidate_persistent'][1]):.2f}", f'difference {ds} {m}')
further = ['close_los', 'near_far']
claims = {
    'memory lowers TBLER on further channels': all(rates[ds, n, 'candidate_persistent'] < rates[ds, n, 'candidate_reset'] for ds in SE for n in further if not (ds == 'second' and n == 'close_los')) and rates['first', 'ray_traced_munich', 'candidate_persistent'] < rates['first', 'ray_traced_munich', 'candidate_reset'],
    'close LOS unchanged on second dataset': rates['second', 'close_los', 'candidate_persistent'] == rates['second', 'close_los', 'candidate_reset'],
    'Large NRX ahead on first dataset for UMi, topology change, close LOS, near-far': all(rates['first', n, 'frozen_cold_k8'] < rates['first', n, 'candidate_persistent'] for n in ('natural_umi', 'controlled_topology_change', 'close_los', 'near_far')),
    'proposed behind for UMi and gaps, ahead for channel change': required['natural_umi']['candidate_persistent'] > required['natural_umi']['frozen_cold_k8'] and required['schedule_gap_reentry']['candidate_persistent'] > required['schedule_gap_reentry']['frozen_cold_k8'] and required['controlled_topology_change']['candidate_persistent'] < required['controlled_topology_change']['frozen_cold_k8'],
}
for claim, holds in claims.items():
    checked += 1
    if not holds:
        bad += 1; print('CLAIM FAILS', claim)
if re.search(r'Table|tab:', section):
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

for stratum, label in enumerate(('0-50', '50-175', '175-400')):
    gain = speed_curve('candidate_reset', stratum) - speed_curve('candidate_persistent', stratum)
    expect(f'{gain:.2f}\\,dB', f'memory gain at {label} Hz')
    print('memory gain', label, 'Hz', round(gain, 3))
gap = speed_curve('candidate_reset', 0) - speed_curve('baseline_perf_csi_kbest', 0)
checked += 1
if gap > .1:
    bad += 1; print('CLAIM FAILS no memory matches Perf.-CSI on slow channels', gap)
print('slow channels: no memory minus Perf.-CSI', round(gap, 3))

with np.load(f'{SW}/latency.npz') as data:
    med = {k[len('latency_ms_'):]: float(np.median(data[k])) for k in data.files if k.startswith('latency_ms_')}
for m, v in med.items():
    if f'{v:.2f}' not in text:
        bad += 1; print('LATENCY MISSING', m, v)
print('latency medians', {k: round(v, 2) for k, v in med.items()})
print('saving % ', round(100 * (1 - med['candidate_persistent'] / med['frozen_cold_k8']), 1), 'added ms', round(med['candidate_persistent'] - med['frozen_cold_k2'], 2))
for phrase in ('1,653', '25,860', 'real-time budget of 1'):
    if phrase not in text.replace('- ', '-') and phrase.replace(',', '{,}') not in section and phrase not in section:
        bad += 1; print('PHRASE MISSING', phrase)
print('RESULT:', 'ALL MATCH' if not bad else f'{bad} MISMATCHES')
