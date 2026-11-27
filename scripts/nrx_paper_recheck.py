"""Second pass: recompute the paper's evaluation values from the raw per-sequence files with code that is separate
from scripts/nrx_paper_numbers.py, and compare them with the text of paper/*.tex.

Every check is either a number that must appear in the text as printed (expect) or a statement that must appear
verbatim and that the recomputed data must support (claim). Exit code 1 if anything is missing or does not hold.

    python scripts/nrx_paper_recheck.py
"""
import glob
import json
import math
import os
import re

import numpy as np

SW = 'experiments/20261104_tbler_sweep/results'                       # released Large NRX, classical receivers
NEW = 'experiments/20261118_test_evaluation/results'                  # proposed NRX, retrained Large NRX
PC = 'experiments/20261112_early_retransmission/results/receivers_pc'  # stored receivers with power control
NET = 'experiments/20261118_network_decisions/results'
TRT = 'experiments/20261114_tensorrt_latency/results'
OURS = ('pilot_memory', 'pilot_memory_emptied', 'neural_rx_k2_trained')
EXTRA = ('pilot_memory_current_tb', 'filter_kbest', 'wiener_kbest')            # same folder as OURS
STORED = ('baseline_lslin_lmmse', 'baseline_lmmse_kbest', 'baseline_perf_csi_kbest', 'frozen_cold_k2', 'frozen_cold_k3',
          'frozen_cold_k8')
CORE = ['natural_umi', 'controlled_topology_change', 'schedule_gap_reentry']
PAPER = os.environ.get('NRX_PAPER', 'paper')                 # paper_ieee for the IEEE version
tex = re.sub(r'\\input\{([^}]+)\}', lambda m: open(f'{PAPER}/{m.group(1)}.tex', encoding='utf8').read(),
             open(f'{PAPER}/main.tex', encoding='utf8').read())
tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)
text = re.sub(r'\s+', ' ', tex)
bad = checked = 0
# The IEEE version (paper_ieee) has no Network Decisions section and words some statements differently; its own wording is
# checked in the block "IEEE version" below, from the same data, and the workshop wording of these labels is skipped there.
IEEE = PAPER != 'paper'
NETWORK = '\\section{Network Decisions}' in tex
IEEE_REWORDED = {'conclusion: phase jumps', 'abstract lead', 'abstract eight iterations', 'abstract three iterations',
                 'modulations: setup', 'modulations: released weights per MCS', 'modulations: memory',
                 'abstract: QPSK and eight iterations', 'modulations: introduction and conclusion'}
section = None                                  # 'network' while checking statements of the Network Decisions section


def skipped(label):
    return (IEEE and label in IEEE_REWORDED) or (section == 'network' and not NETWORK)


def expect(string, label):
    global bad, checked
    if skipped(label):
        return
    checked += 1
    if string not in text:
        bad += 1; print('MISSING', label, '|', string)


def claim(phrase, holds, label):
    """A statement: it must appear verbatim, and the recomputed data must support it."""
    global bad, checked
    if skipped(label):
        return
    checked += 2
    if phrase not in text:
        bad += 1; print('MISSING', label, '|', phrase)
    if not holds:
        bad += 1; print('CLAIM FAILS', label)


def directories(setting, method):
    if setting == 'phase':                       # power control with a phase jump at every power step
        assert method in OURS
        return [f'{NEW}/power_control_phase']
    if setting == 'pc':
        return [f'{NEW}/power_control' if method in OURS+EXTRA else PC]
    return [f'{NEW if method in OURS+EXTRA else SW}/{d}' for d in ('sweep', 'sweep_high')]


def errors(setting, cond, method, combine=None):
    """Failures and blocks per SNR point over TBs 17-64 of active UEs; `combine` lists receivers that must all fail."""
    table = {}
    for position, directory in enumerate(directories(setting, method)):
        files = sorted(glob.glob(f'{directory}/{cond}/{method}/episode_*.npz'))
        assert len(files) == 32, (setting, cond, method, len(files))
        for path in files:
            with np.load(path) as data:
                mask = data['active'][16:].astype(bool)
                failed = data['errors'][:, 16:]
                for other in combine or ():
                    with np.load(path.replace(f'{directory}/{cond}/{method}', f'{directories(setting, other)[position]}/{cond}/{other}')) as second:
                        assert np.array_equal(second['fingerprint'], data['fingerprint'])
                        failed = failed & second['errors'][:, 16:]
                for k, snr in enumerate(data['snr_db']):
                    entry = table.setdefault(float(snr), {}).setdefault(position, [0, 0])
                    entry[0] += int((failed[k] & mask).sum()); entry[1] += int(mask.sum())
    snrs = sorted(table)
    return snrs, [table[x][min(table[x])][0]/table[x][min(table[x])][1] for x in snrs]


def crossing(snrs, rates, target=.1):
    # monotone envelope by pooled adjacent violators, then log-TBLER interpolation
    levels = []
    for r in rates:
        levels.append([r, 1])
        while len(levels) > 1 and levels[-2][0] < levels[-1][0]:
            v2, n2 = levels.pop(); v1, n1 = levels.pop()
            levels.append([(v1*n1+v2*n2)/(n1+n2), n1+n2])
    fit = [v for v, n in levels for _ in range(n)]
    for i in range(len(fit)-1):
        if fit[i] >= target >= fit[i+1]:
            if fit[i] == fit[i+1]:
                return (snrs[i]+snrs[i+1])/2
            f = ((math.log(target)-math.log(fit[i]))/(math.log(fit[i+1])-math.log(fit[i])) if fit[i+1] > 0
                 else (fit[i]-target)/(fit[i]-fit[i+1]))
            return snrs[i]+f*(snrs[i+1]-snrs[i])
    return None


def span(values):
    values = sorted(round(v, 2) for v in values)
    return f'{values[0]:.2f} to {values[-1]:.2f}'


# ---------------------------------------------------------------------------------------------- required SNR
required = {s: {c: {m: round(crossing(*errors(s, c, m)), 2) for m in (*OURS, *STORED)} for c in CORE} for s in ('no', 'pc')}
def gap(s, a, b):
    return [round(required[s][c][a]-required[s][c][b], 2) for c in CORE]


lead, three, behind = gap('no', 'neural_rx_k2_trained', 'pilot_memory'), gap('no', 'frozen_cold_k3', 'pilot_memory'), gap('no', 'pilot_memory', 'frozen_cold_k8')
memory, no_memory = gap('no', 'pilot_memory_emptied', 'pilot_memory'), gap('no', 'neural_rx_k2_trained', 'pilot_memory_emptied')
cut, retraining = gap('no', 'frozen_cold_k2', 'frozen_cold_k8'), gap('no', 'frozen_cold_k2', 'neural_rx_k2_trained')
# The workshop text states the results qualitatively, as Cammerer et al. and Wiesmayr et al. do (user, 2026-11-27);
# Table 1 to 5 carry the values. Each qualitative word is tied to a threshold here.
claim('significantly raises the required SNR in all three conditions',
      min(cut) > 1, 'iteration cut')
claim('recovers only part of this loss',
      all(.4 <= v <= .7 and v < c for v, c in zip(retraining, cut)), 'retraining gain')
claim('the pilot memory and the channel filter, clearly outperforms it', min(lead) > .3, 'lead over retrained')
claim('It also outperforms the Large NRX with $N_{\\mathrm{it}}=3$ in all three conditions and performs close to the Large NRX '
      'with $N_{\\mathrm{it}}=8$', min(three) > 0 and max(behind) <= .25, 'three and eight iterations')
claim('The proposed NRX outperforms LS + LMMSE in all three conditions', min(gap('no', 'baseline_lslin_lmmse', 'pilot_memory')) > 0, 'LS + LMMSE')
kbest = gap('no', 'pilot_memory', 'baseline_lmmse_kbest')
claim('remains ahead of it for UMi and channel change, whereas the proposed NRX is ahead for scheduling gaps',
      kbest[0] > .1 and kbest[1] > .1 and kbest[2] < -.1, 'LMMSE + K-Best')
claim('Perf.-CSI + K-Best remains ahead of all other receivers',
      all(required[s][c]['baseline_perf_csi_kbest'] < required[s][c][m] for s in required for c in CORE
          for m in required[s][c] if m != 'baseline_perf_csi_kbest'), 'perfect CSI')
claim('which significantly reduces the required SNR in all three conditions. Without its memory, the proposed NRX performs at most '
      'slightly worse than the retrained Large NRX', min(memory) > .5 and 0 < max(-v for v in no_memory) <= .25, 'memory gain')
pc_lead, pc_three, pc_behind, pc_memory = (gap('pc', 'neural_rx_k2_trained', 'pilot_memory'), gap('pc', 'frozen_cold_k3', 'pilot_memory'),
                                           gap('pc', 'pilot_memory', 'frozen_cold_k8'), gap('pc', 'pilot_memory_emptied', 'pilot_memory'))
claim('The proposed NRX clearly outperforms the retrained Large NRX and the Large NRX with $N_{\\mathrm{it}}=3$, performs close to the '
      'Large NRX with $N_{\\mathrm{it}}=8$', min(pc_lead) > .3 and min(pc_three) > .3 and max(pc_behind) <= .25, 'power control')
claim('outperforms LMMSE + K-Best in all three conditions', min(gap('pc', 'baseline_lmmse_kbest', 'pilot_memory')) > 0,
      'power control: LMMSE + K-Best')
claim('The memory again yields a significant gain.', min(pc_memory) > .5, 'power control: memory')
phase_required = {c: {m: round(crossing(*errors('phase', c, m)), 2) for m in OURS} for c in CORE}
phase_memory = [round(phase_required[c]['pilot_memory_emptied']-phase_required[c]['pilot_memory'], 2) for c in CORE]
phase_lead = [round(phase_required[c]['neural_rx_k2_trained']-phase_required[c]['pilot_memory'], 2) for c in CORE]
claim('with a uniformly random phase jump at every power step (Table~\\ref{tab:'
      + ('impairments}' if IEEE else 'modulations}, bottom') + '), the memory keeps almost all of its gain', all(a >= b-.05 for a, b in zip(phase_memory, pc_memory)), 'phase jumps: memory')
claim('it requires neither power consistency nor phase continuity across the stored slots',
      min(pc_memory) > .5 and min(phase_memory) > .5 and min(phase_lead) > .4, 'related work: power and phase')
claim('and keeps its memory gain with a phase jump at every power step',
      all(abs(a-b) <= .05 for a, b in zip(phase_memory, pc_memory)), 'conclusion: phase jumps')
both = lead+pc_lead
claim('requires more than 0.5\\,dB less SNR than a stateless two-iteration NRX retrained on the same data', min(both) > .5, 'abstract lead')
if not IEEE:
    claim('and stays within 0.2\\,dB of the eight-iteration NRX, with and without power control', max(behind+pc_behind) <= .2,
          'abstract eight iterations')
    claim('outperforms the three-iteration NRX, and stays within', min(three+pc_three) > 0, 'abstract three iterations')
claim('clearly outperforms a stateless two-iteration NRX retrained on the same data and the released NRXs with up to three iterations',
      min(gap(s, m, 'pilot_memory')[i] for s in required for m in ('frozen_cold_k2', 'frozen_cold_k3', 'neural_rx_k2_trained')
          for i in range(3)) > 0, 'introduction: stateless NRXs')
claim('With two iterations, it recovers most of the accuracy that the cut from eight iterations costs',
      all(required[s][c]['frozen_cold_k2']-required[s][c]['pilot_memory'] > .5*(required[s][c]['frozen_cold_k2']-required[s][c]['frozen_cold_k8'])
          for s in required for c in CORE), 'introduction: most of the accuracy')
claim('the cut to two iterations clearly raises the signal-to-noise ratio (SNR) required for a transport block error rate (TBLER) of 10\\%, '
      'even when the NRX is retrained for two iterations',
      min(gap(s, 'neural_rx_k2_trained', 'frozen_cold_k8')[i] for s in ('no', 'pc') for i in range(3)) > .5, 'introduction')

# ------------------------------------------- ablation with the current TB only; the filter with a classical receiver
for s in required:
    for c in CORE:
        for m in EXTRA:
            required[s][c][m] = round(crossing(*errors(s, c, m)), 2)
current_only = gap('no', 'pilot_memory_current_tb', 'pilot_memory_emptied')
claim('the proposed NRX requires even slightly more than with its memory emptied', 0 <= min(current_only) and max(current_only) <= .2,
      'current TB only')
claim('The gain stems from the earlier slots', min(current_only+gap('pc', 'pilot_memory_current_tb', 'pilot_memory_emptied')) >= 0,
      'no gain without earlier slots')
filter_gain = gap('no', 'baseline_lmmse_kbest', 'filter_kbest')
over_wiener, wiener_gain = gap('no', 'wiener_kbest', 'filter_kbest'), gap('no', 'baseline_lmmse_kbest', 'wiener_kbest')
claim('they reduce its required SNR in all three conditions (Table~\\ref{tab:required})', min(filter_gain) > 0, 'filter with K-Best')
claim('performs close to the channel filter for UMi and scheduling gaps but falls clearly behind it for channel change, even behind LMMSE + K-Best',
      max(over_wiener[0], over_wiener[2]) < .1 and over_wiener[1] > .5 and wiener_gain[1] < -.3, 'classical multi-slot estimator')
pc_gain, pc_over = gap('pc', 'baseline_lmmse_kbest', 'filter_kbest'), gap('pc', 'wiener_kbest', 'filter_kbest')
# Related Work and introduction: what classical multi-slot estimation cannot do, with the measured comparison (the IEEE
# version words it in its own block; the co-author took it out of the workshop version on 2026-11-27)
pc_wiener_gain = gap('pc', 'baseline_lmmse_kbest', 'wiener_kbest')

# ------------------------------------------------------------------------- effective SNR and Doppler groups (UMi)
def effective(cond, method, group=None, edges=np.arange(-10., 30.5, 1.)):
    fails, blocks = np.zeros(len(edges)-1), np.zeros(len(edges)-1)
    for first, directory in enumerate(directories('no', method)):
        for path in sorted(glob.glob(f'{directory}/{cond}/{method}/episode_*.npz')):
            index = int(path[-9:-4])
            if group is not None and index % 3 != group:
                continue
            with np.load(path) as data, np.load(f'{SW}/{"sweep_high" if first else "sweep"}/{cond}/candidate_persistent/episode_{index:05d}.npz') as ref:
                assert np.array_equal(data['fingerprint'], ref['fingerprint'])
                mask = ref['active'][16:].astype(bool)
                power = 10*np.log10(ref['channel_power'][16:])
                for k in range(first, len(data['snr_db'])):     # the shared point is counted once
                    bins = np.floor(data['snr_db'][k]+power-edges[0]).astype(int)
                    ok = mask & (bins >= 0) & (bins < len(fails))
                    np.add.at(fails, bins[ok], data['errors'][k, 16:][ok]); np.add.at(blocks, bins[ok], 1)
    rates = fails/np.maximum(blocks, 1)
    # weighted pooled adjacent violators
    levels = []
    for r, n in zip(rates, np.maximum(blocks, 1)):
        levels.append([r, n, 1])
        while len(levels) > 1 and levels[-2][0] < levels[-1][0]:
            v2, n2, c2 = levels.pop(); v1, n1, c1 = levels.pop()
            levels.append([(v1*n1+v2*n2)/(n1+n2), n1+n2, c1+c2])
    fit = [v for v, _, c in levels for _ in range(c)]
    centers = edges[:-1]+.5
    for i in range(len(fit)-1):
        if fit[i] >= .1 >= fit[i+1] and fit[i] != fit[i+1]:
            f = (math.log(.1)-math.log(fit[i]))/(math.log(fit[i+1])-math.log(fit[i])) if fit[i+1] > 0 else (fit[i]-.1)/(fit[i]-fit[i+1])
            return centers[i]+f
    return None


doppler = {g: {m: effective('natural_umi', m, g, np.arange(-6., 14.)) for m in OURS} for g in range(3)}
gains = [doppler[g]['pilot_memory_emptied']-doppler[g]['pilot_memory'] for g in range(3)]
expect('as Fig.~\\ref{fig:doppler} shows for UMi in the effective SNR', 'Doppler gain')
claim('The gain of the memory grows with the Doppler shift', gains[0] < gains[1] < gains[2], 'Doppler order')

# ------------------------------------------------------------------------------------- TBs and training length
counts = []
for cond in CORE:
    files = sorted(glob.glob(f'{NEW}/sweep/{cond}/pilot_memory/episode_*.npz'))
    assert len(files) == 32
    counts.append(sum(int(np.load(path)['active'][16:].sum()) for path in files))
expect(f'over {min(counts):,} to {max(counts):,} TBs per condition and SNR point'.replace(',', '{,}'), 'TBs per SNR point')

# --------------------------------------------------------------------------------------------------- latency
aw = np.load(f'{TRT}/fused_a100_samples.npz')
ms = {k: float(np.median(aw[f'events_{k}'])) for k in ('large_k2', 'large_k3', 'large_k8', 'proposed', 'fit_only',
                                                        'proposed_and_fit', 'proposed_overlapped')}
claim('median of 2{,}000 runs on an NVIDIA A100 GPU', 'A100' in json.load(open(f'{TRT}/fused_a100.json'))['gpu']
      and all(len(aw[f'events_{k}']) == 2000 for k in ms), 'GPU and number of runs')
claim('with TensorRT~10.3 in FP16', json.load(open(f'{TRT}/fused_a100.json'))['tensorrt'].startswith('10.3.'), 'TensorRT version')
claim('Each further CGNN iteration adds about 0.3\\,ms, so $N_{\\mathrm{it}}=2$ meets the 1\\,ms budget, which $N_{\\mathrm{it}}=3$ exceeds.',
      .25 <= (ms['large_k8']-ms['large_k2'])/6 < .35 and ms['large_k2'] < 1 < ms['large_k3'], 'iteration time')
claim('The proposed NRX takes only slightly longer than the Large NRX with $N_{\\mathrm{it}}=2$',
      0 < ms['proposed']-ms['large_k2'] < .05 and ms['proposed'] < 1, 'time of the proposed NRX')
claim('the GPU work per slot stays below 1\\,ms, whether this computation runs after the slot or in parallel with it',
      max(ms['proposed'], ms['proposed_and_fit'], ms['proposed_overlapped']) < 1, 'budget including the fit')
expect('only depend on earlier slots and are computed before the slot arrives', 'time before the slot')
for phrase in ('meets the 1\\,ms real-time budget, which the three-iteration NRX exceeds',       # abstract
               'show that it remains real-time capable, unlike the released three-iteration NRX',  # contributions
               'it remains real-time capable, unlike the three-iteration NRX'):                    # conclusion
    claim(phrase, ms['proposed'] < 1 < ms['large_k3'], 'real time')

# -------------------------------------------------------------------------------------------- network decisions
section = 'network'
realistic = [m for m in (*OURS, *STORED) if m != 'baseline_perf_csi_kbest']
fallback, unfixable, repeated = {}, {}, {}
for s in ('no', 'pc'):
    for c in CORE:
        snrs, alone = errors(s, c, 'pilot_memory')
        fallback[s, c] = round(round(crossing(snrs, alone), 2)-round(crossing(*errors(s, c, 'pilot_memory', ['baseline_lmmse_kbest'])), 2), 2)
        point = int(np.argmin(np.abs(np.asarray(snrs)-crossing(snrs, alone))))
        every = errors(s, c, 'pilot_memory', [m for m in realistic if m != 'pilot_memory'])[1][point]
        unfixable[s, c] = every/alone[point]
        flags = []
        position, directory = max((p, d) for p, d in enumerate(directories(s, 'pilot_memory'))
                                  if snrs[point] in np.load(sorted(glob.glob(f'{d}/{c}/pilot_memory/episode_*.npz'))[0])['snr_db'])
        for path in sorted(glob.glob(f'{directory}/{c}/pilot_memory/episode_*.npz')):
            with np.load(path) as data:
                k = int(np.flatnonzero(data['snr_db'] == snrs[point])[0])
                active, ids, failed, last = data['active'].astype(bool), data['ue_ids'], data['errors'][k], {}
                for tb in range(active.shape[0]):
                    for u in np.flatnonzero(active[tb]):
                        if tb >= 16 and failed[tb, u]:
                            flags.append(last.get(int(ids[tb, u]), False))
                        last[int(ids[tb, u])] = bool(failed[tb, u])
        repeated[s, c] = float(np.mean(flags))


def percent(values):
    values = sorted(round(100*v) for v in values)
    return f'{values[0]}\\% to {values[-1]}\\%'


# fallbacks after a CRC failure, recomputed from the stored per-TB results at the operating points; pooled over the
# three conditions as in Table 5 (the blocks per SNR point are the same for all receivers of a condition)
rescued, pooled_rescued, pooled_repeated = [], {}, {}
for s in ('no', 'pc'):
    failed_total, after_failure = 0., 0.
    for c in CORE:
        snrs, alone = errors(s, c, 'pilot_memory')
        point = int(np.argmin(np.abs(np.asarray(snrs)-crossing(snrs, alone))))
        for backup in ('baseline_lslin_lmmse', 'pilot_memory_emptied', 'neural_rx_k2_trained', 'frozen_cold_k2'):
            rescued.append(1-errors(s, c, 'pilot_memory', [backup])[1][point]/alone[point])
        blocks = sum(int(np.load(path)['active'][16:].sum()) for path in
                     sorted(glob.glob(f'{directories(s, "pilot_memory")[0]}/{c}/pilot_memory/episode_*.npz')))
        failed_total += alone[point]*blocks
        after_failure += repeated[s, c]*alone[point]*blocks
        for backup in ('baseline_lslin_lmmse', 'neural_rx_k2_trained', 'baseline_lmmse_kbest', 'frozen_cold_k8'):
            pooled_rescued[s, backup] = pooled_rescued.get((s, backup), 0.) + (alone[point]-errors(s, c, 'pilot_memory', [backup])[1][point])*blocks
    for backup in ('baseline_lslin_lmmse', 'neural_rx_k2_trained', 'baseline_lmmse_kbest', 'frozen_cold_k8'):
        pooled_rescued[s, backup] /= failed_total
    pooled_repeated[s] = after_failure/failed_total
claim('a fallback that a real-time BS could run, LS + LMMSE or a stateless NRX with $N_{\\mathrm{it}}=2$, rescues almost none of the failed TBs',
      max(rescued) < .01, 'real-time fallbacks')
claim('and LMMSE + K-Best and the Large NRX with $N_{\\mathrm{it}}=8$ rescue only a few',
      max(pooled_rescued[s, b] for s in ('no', 'pc') for b in ('baseline_lmmse_kbest', 'frozen_cold_k8')) < .1, 'slow fallbacks')
claim('Most failed TBs follow an error of the same UE', min(pooled_repeated.values()) > .5, 'errors after errors')
# when the channel filter falls back from its memory (experiments/20261122_robustness/fallback_k.py)
probe = json.load(open('experiments/20261122_robustness/results/fallback_k_summary.json'))
change, jump = probe['power_control'], probe['power_control_phase']
expect('The receiver itself decides when to fall back from its memory', 'filter fallback')
claim('it uses none at almost every channel change or phase jump',
      change['change']['no_stored_tb_share'] > .9 and jump['phase_jump']['no_stored_tb_share'] > .9
      and change['error_mismatches'] == 0 and jump['error_mismatches'] == 0 and change['sequences'] == jump['sequences'] == 16,
      'filter fallback at changes and jumps')
claim('and almost never otherwise', change['other']['no_stored_tb_share'] < .01 and jump['other']['no_stored_tb_share'] < .01,
      'filter fallback elsewhere')
claim('a decision the classical multi-slot estimator cannot make', wiener_gain[1] < 0 and over_wiener[1] == max(over_wiener),
      'classical estimator after a change')
controller = json.load(open(f'{NET}/controller.json'))
step = controller['step_by_step']['CRC history and receiver evidence, budget of the rule']
single = controller['single_check']
claim('already catches most errors in the test sequences', controller['rule']['caught'] > .5, 'rule')
claim('the controller catches almost all of them', step['caught'] > .95, 'controller')
claim('and flags as many correct TBs in the test sequences as this rule',
      f'{100*step["false_alarm"]:.1f}' == f'{100*controller["rule"]["false_alarm"]:.1f}', 'false alarms')
compare = json.load(open(f'{NET}/compare_receivers.json'))['receivers']
ours, stateless = compare['proposed'], compare['stateless']
claim('Without the memory state, it still catches almost all errors of the proposed NRX, and clearly more errors of the stateless NRX than the rule',
      ours['caught_share'] > .95 and stateless['caught_share']-stateless['rule_caught_share'] > .05, 'controller on the stateless NRX')
claim('As that NRX fails more often at the same SNR', stateless['failures_per_100']/ours['failures_per_100'] > 1.5, 'twice as many failures')
claim('many more errors remain for retransmission after the CRC fails than with the proposed NRX',
      stateless['late_per_100']/ours['late_per_100'] > 3, 'late retransmissions')
claim('leaving far fewer for after the CRC fails than with a stateless NRX',
      stateless['late_per_100']/ours['late_per_100'] > 3, 'six times fewer late')
claim(f'request {round(100*step["caught"])}\\% of the needed retransmissions before decoding finishes, against '
      f'{round(100*controller["rule"]["caught"])}\\% for the CRC history alone at the same false-alarm rate',
      f'{100*step["false_alarm"]:.1f}' == f'{100*controller["rule"]["false_alarm"]:.1f}', 'abstract controller')

# --------------------------------------------------------------------------------------------- robustness (Modal)
section = None
ROB = 'experiments/20261122_robustness/results'
robust = json.load(open(f'{ROB}/summary.json'))
for test in ('cfo', 'tdd_dddsu'):
    assert all(not v['incomplete'] and len(v['required_snr_db']) >= 3 for v in robust[test].values()), test
cfo, tdd = robust['cfo'].values(), robust['tdd_dddsu'].values()
loss = lambda v, r: v['required_snr_db'][r]-v['reference_snr_db'][r]
expect('Except in the impairment tests, each UE keeps its phase continuous across slots, as for DMRS bundling,', 'assumptions: robustness tests')
claim('With a frequency offset of $\\pm 0.1$\\,ppm per UE, the most 3GPP allows',
      'CFO_PPM = 0.1' in open('experiments/20261122_robustness/banks.py').read(), 'CFO: offset')
claim('the proposed NRX loses even less than the retrained Large NRX',
      all(loss(v, 'pilot_memory') < .1 and .13 <= loss(v, 'neural_rx_k2_trained') <= .18 for v in cfo), 'CFO: losses')
claim('as the channel filter tracks the rotation',
      all(v['pilot_memory_memory_gain_db'] > v['reference_memory_gain_db'] for v in cfo), 'CFO: memory gain grows')
expect('With the TDD pattern DDDSU, i.e., one uplink slot every 2.5\\,ms,', 'TDD: pattern')
claim('the memory keeps most of its gain',
      all(v['pilot_memory_memory_gain_db'] > .7*v['reference_memory_gain_db'] for v in tdd), 'TDD: memory share')
claim('and the proposed NRX still clearly outperforms the retrained Large NRX',
      all(v['pilot_memory_lead_over_retrained_db'] >= .4 for v in tdd), 'TDD: lead over retrained')
section = 'network'                                          # simulated HARQ
harq = json.load(open(f'{ROB}/harq_summary.json'))
conditions = [harq[c] for c in ('natural_umi', 'controlled_topology_change', 'schedule_gap_reentry')]
claim('Simulating the retransmissions with Chase combining',
      harq['all']['decision_mismatches'] == 0 and all(c['compared'] > 100 for c in conditions), 'HARQ: decisions reproduced')
claim('early requests let the BS decode a failed TB one slot earlier in median',
      harq['all']['median_slots_saved'] == 1 and all(c['median_slots_saved'] == 1 for c in conditions), 'HARQ: one slot')
claim('at the cost of a few unneeded transmissions', harq['all']['early_extra_transmissions_per_100_tbs'] <= 3, 'HARQ: extra transmissions')
section = None
expect('Future work includes measured data.', 'future work')


# --------------------------------------------------------------------------------------------- QPSK and 64-QAM
MOD = 'experiments/20261121_modulations/results'
mods = {label: json.load(open(f'{MOD}/{label}/summary.json')) for label in ('nrx_large_qpsk', 'nrx_large_64qam')}


def snr(label, setting, condition, receiver):
    value = mods[label][setting][condition]
    assert not value['incomplete'], (label, setting, condition)
    return value['required_snr_db'][receiver]


cells = [(label, s, c) for label in mods for s in ('sweep', 'power_control') for c in CORE]
gains = [snr(*k, 'pilot_memory_emptied')-snr(*k, 'pilot_memory') for k in cells]
expect('For QPSK (MCS 9) and 64-QAM (MCS 19), with all NRXs trained in the same way from the released weights for each MCS',
       'modulations: setup')
claim('trained in the same way from the released weights for each MCS',
      all(json.load(open(f'{MOD}/{label}/stage1/recipe.json'))['initializer_checkpoint'] == f'models/pretrained/{label}_weights'
          for label in mods), 'modulations: released weights per MCS')
claim('the memory again yields a significant gain (Table~\\ref{tab:modulations}, top)', min(gains) > .5, 'modulations: memory')
claim('the proposed NRX outperforms the retrained Large NRX and the Large NRX with $N_{\\mathrm{it}}=3$ in all cases, with and without power control',
      all(snr(*k, 'pilot_memory') < min(snr(*k, 'neural_rx_k2_trained'), snr(*k, 'frozen_cold_k3')) for k in cells),
      'modulations: retrained and three iterations')
qpsk = [k for k in cells if k[0] == 'nrx_large_qpsk']
claim('With QPSK, it also outperforms the Large NRX with $N_{\\mathrm{it}}=8$ in all cases',
      all(snr(*k, 'pilot_memory') < snr(*k, 'frozen_cold_k8') for k in qpsk), 'modulations: QPSK and eight iterations')
claim('With QPSK, and on the tapped delay line and urban macrocell channels of two further datasets, it also outperforms the eight-iteration NRX',
      all(snr(*k, 'pilot_memory') < snr(*k, 'frozen_cold_k8') for k in qpsk), 'abstract: QPSK and eight iterations')
behind = {k: snr(*k, 'pilot_memory')-snr(*k, 'frozen_cold_k8') for k in cells if k[0] == 'nrx_large_64qam'}
claim('with 64-QAM, it performs close to it with power control',
      all(v <= .15 for k, v in behind.items() if k[1] == 'power_control'), 'modulations: 64-QAM with power control')
claim('but falls behind it after scheduling gaps without power control',
      round(behind['nrx_large_64qam', 'sweep', 'schedule_gap_reentry'], 1) == .7
      and max(behind, key=behind.get) == ('nrx_large_64qam', 'sweep', 'schedule_gap_reentry'), 'modulations: 64-QAM after gaps')
for phrase in ('MU-MIMO with QPSK, 16-QAM, and 64-QAM over sequences', 'an evaluation with QPSK, 16-QAM, and 64-QAM in which'):
    expect(phrase, 'modulations: introduction and conclusion')

# --------------------------------------------------------------------------------------------- IEEE version
if IEEE:
    mod_lead = [snr(*k, 'neural_rx_k2_trained')-snr(*k, 'pilot_memory') for k in cells]
    mod_three = [snr(*k, 'frozen_cold_k3')-snr(*k, 'pilot_memory') for k in cells]
    all_lead = lead+pc_lead+mod_lead
    claim('requires more than 0.5\\,dB less SNR than a stateless two-iteration NRX retrained on the same data and outperforms the '
          'three-iteration NRX in all cases, with and without power control', min(all_lead) > .5 and min(three+pc_three+mod_three) > 0,
          'IEEE abstract: lead and three iterations, three modulations')
    claim('with QPSK, it also outperforms the eight-iteration NRX',
          all(snr(*k, 'pilot_memory') < snr(*k, 'frozen_cold_k8') for k in qpsk), 'IEEE abstract: QPSK')
    claim('It keeps most of its memory gain with phase jumps, frequency offsets, and TDD uplink slots',
          min(a/b for a, b in zip(phase_memory, pc_memory)) > .9
          and all(v['pilot_memory_memory_gain_db'] > .5*v['reference_memory_gain_db'] for t in ('cfo', 'tdd_dddsu')
                  for v in robust[t].values()), 'IEEE abstract: robustness')
    claim('For QPSK (MCS 9) and 64-QAM (MCS 19), we train and evaluate all NRXs in the same way, starting from the released '
          'weights for each MCS',
          all(json.load(open(f'{MOD}/{label}/stage1/recipe.json'))['initializer_checkpoint'] == f'models/pretrained/{label}_weights'
              for label in mods), 'IEEE: modulations, setup')
    claim('The memory again yields a significant gain, and the proposed NRX outperforms the retrained Large NRX', min(gains) > .5,
          'IEEE: modulations, memory')
    claim('With two CGNN iterations and QPSK, 16-QAM, or 64-QAM, it outperforms stateless NRXs with up to three iterations',
          min(lead+pc_lead+mod_lead) > 0 and min(three+pc_three+mod_three) > 0, 'IEEE: conclusion')
    expect('MU-MIMO with QPSK, 16-QAM, and 64-QAM over sequences', 'IEEE: introduction')
    expect('\\includegraphics[width=\\columnwidth]{figures/modulations_summary.pdf}', 'IEEE: modulation figure')

# --------------------------------------------------------------------------------------------- further datasets
FURTHER = {'first': 'experiments/20261118_further_datasets/results/v2/errors.npz',
           'second': 'experiments/20261118_further_datasets/results/seed86/errors.npz'}
if all(glob.glob(path) for path in FURTHER.values()) and 'further datasets at a fixed SNR' in text:
    names = ['historical_tdl_b100', 'historical_tdl_c300', 'natural_umi', 'natural_uma', 'controlled_topology_change',
             'schedule_gap_reentry', 'close_los', 'near_far', 'ray_traced_munich']
    rates, blocks = {}, {}
    for ds, path in FURTHER.items():
        data = np.load(path)
        for c, name in enumerate(names[:data['active'].shape[0]]):
            mask = data['active'][c, :, 16:]
            blocks[ds, name] = int(mask.sum())
            for m in ('candidate_persistent', 'candidate_reset', 'frozen_cold_k8'):
                rates[ds, name, m] = (data[f'errors_{m}'][c, 0, :, 16:] & mask).sum()/mask.sum()
    cases = sorted({k[:2] for k in rates})
    lower = [k for k in cases if rates[k+('candidate_persistent',)] < rates[k+('candidate_reset',)]]
    same = [k for k in cases if rates[k+('candidate_persistent',)] == rates[k+('candidate_reset',)]]
    behind = [k for k in cases if rates[k+('frozen_cold_k8',)] < rates[k+('candidate_persistent',)]]
    ahead = [k for k in cases if rates[k+('candidate_persistent',)] < rates[k+('frozen_cold_k8',)]]
    claim('The memory lowers TBLER in all cases but one', len(lower) == len(cases)-1, 'further datasets: memory')
    claim('outperforms the Large NRX with $N_{\\mathrm{it}}=8$ for both TDL channels and for the UMa channels on both datasets',
          set(ahead) == {(ds, name) for ds in FURTHER for name in ('historical_tdl_b100', 'historical_tdl_c300', 'natural_uma')},
          'further datasets: which cases')
    claim('which it does not without its memory',
          len(ahead) == 6 and all(rates[k+('frozen_cold_k8',)] < rates[k+('candidate_reset',)] for k in ahead),
          'further datasets: behind without the memory')
    if not IEEE:
        claim('on the tapped delay line and urban macrocell channels of two further datasets, it also outperforms the eight-iteration NRX',
              {name for _, name in ahead} == {'historical_tdl_b100', 'historical_tdl_c300', 'natural_uma'}, 'abstract: further datasets')
    claim('remains ahead in most other cases',
          len(behind) > (len(cases)-len(ahead))/2, 'further datasets: eight iterations ahead')
    relative = sorted(behind, key=lambda k: rates[k+('candidate_persistent',)]/rates[k+('frozen_cold_k8',)], reverse=True)
    claim('most clearly for closely spaced LOS and near-far UEs', all(k[1] in ('close_los', 'near_far') for k in relative[:3]),
          'further datasets: largest deficits')
else:
    print('further datasets: not checked (results or paragraph missing)')

# ------------------------------------------------------------------------------------------------------ tables
# Every cell of the tables that scripts/make_nrx_paper_tables.py writes, against the values recomputed above.
def table(name):
    """Rows of <paper>/tables/<name>.tex as (label, cells); None if the paper does not use the table."""
    if f'\\input{{tables/{name}}}' not in tex:
        return None
    rows = []
    for line in open(f'{PAPER}/tables/{name}.tex', encoding='utf8'):
        if '&' in line and '\\multicolumn' not in line:
            cells = [c.strip() for c in line.strip().removesuffix('\\\\').split('&')]
            rows.append((cells[0], cells[1:]))
    return rows


def cell(value, digits=2):
    text = f'{value:.{digits}f}'
    return '$-$'+text[1:] if text.startswith('-') else text


def compare_rows(name, expected, digits=2):
    """expected: (label, values) in table order; values None for '--'."""
    global bad, checked
    rows = table(name)
    if rows is None:
        return
    value = re.compile(r'(\$-\$)?\d+\.\d+|--')
    body = [r for r in rows if r[1] and all(value.fullmatch(c) for c in r[1])]
    checked += 1
    if [label for label, _ in body] != [label for label, _ in expected]:
        bad += 1; print('TABLE ROWS', name, [label for label, _ in body], [label for label, _ in expected]); return
    for (label, cells), (_, values) in zip(body, expected):
        wanted = ['--' if v is None else cell(v, digits) for v in values]
        checked += 1
        if cells != wanted:
            bad += 1; print('TABLE CELLS', name, label, cells, wanted)


LABEL = {'baseline_lslin_lmmse': 'LS + LMMSE', 'baseline_lmmse_kbest': 'LMMSE + K-Best', 'baseline_perf_csi_kbest': 'Perf.-CSI + K-Best',
         'frozen_cold_k2': 'Large NRX, $N_{\\mathrm{it}}=2$', 'frozen_cold_k3': 'Large NRX, $N_{\\mathrm{it}}=3$',
         'frozen_cold_k8': 'Large NRX, $N_{\\mathrm{it}}=8$', 'neural_rx_k2_trained': 'Large NRX retrained, $N_{\\mathrm{it}}=2$',
         'pilot_memory': 'Proposed NRX', 'pilot_memory_emptied': '\\quad no memory', 'pilot_memory_current_tb': '\\quad current TB only',
         'filter_kbest': 'K-Best, memory and filter', 'wiener_kbest': 'K-Best, multi-slot LMMSE'}
order = ('baseline_lslin_lmmse', 'baseline_lmmse_kbest', 'baseline_perf_csi_kbest', 'frozen_cold_k2', 'frozen_cold_k3', 'frozen_cold_k8',
         'neural_rx_k2_trained', 'pilot_memory', 'pilot_memory_emptied', 'pilot_memory_current_tb', 'filter_kbest', 'wiener_kbest')
compare_rows('required_snr', [(LABEL[m], [required[s][c][m] for s in ('no', 'pc') for c in CORE]) for m in order])
top = [(LABEL[m], [snr(label, s, c, m) for label in ('nrx_large_qpsk', 'nrx_large_64qam') for s in ('sweep', 'power_control') for c in CORE])
       for m in ('frozen_cold_k3', 'frozen_cold_k8', 'neural_rx_k2_trained', 'pilot_memory', 'pilot_memory_emptied')]
bottom = [(LABEL[m], [required['pc'][c][m] for c in CORE] + [phase_required[c][m] for c in CORE]
           + [robust['cfo'][c]['required_snr_db'][m] for c in CORE] + [robust['tdd_dddsu'][c]['required_snr_db'][m] for c in CORE])
          for m in ('neural_rx_k2_trained', 'pilot_memory', 'pilot_memory_emptied')]
compare_rows('modulations', top+bottom)
compare_rows('impairments', bottom)
compare_rows('latency', [(cell(ms['large_k2']), [ms[k] for k in ('large_k3', 'large_k8', 'proposed', 'fit_only', 'proposed_and_fit',
                                                                  'proposed_overlapped')])])
if FURTHER and 'rates' in globals():
    NAMES = {'historical_tdl_b100': 'TDL-B100', 'historical_tdl_c300': 'TDL-C300', 'natural_umi': 'UMi', 'natural_uma': 'UMa',
             'controlled_topology_change': 'Channel change', 'schedule_gap_reentry': 'Scheduling gaps', 'close_los': 'Close LOS UEs',
             'near_far': 'Near-far UEs', 'ray_traced_munich': 'Munich, ray-traced'}
    compare_rows('further_datasets', [(NAMES[n], [100*rates[(ds, n, m)] if (ds, n, m) in rates else None for ds in FURTHER
                                                  for m in ('frozen_cold_k8', 'candidate_persistent', 'candidate_reset')]) for n in names], 1)
    snr_points = np.load(FURTHER['first'])['snr_db'][:, 0]
    if table('further_datasets') is not None:
        expect('at ' + ', '.join(f'{v:.0f}' for v in snr_points[:-1]) + f', and {snr_points[-1]:.0f}\\,dB from the top', 'further datasets: SNR points')
if NETWORK:
    stateless_label = 'Stateless NRX, $N_{\\mathrm{it}}=2$'
    compare_rows('network', [
        ('\\quad at a channel change', [None, 100*change['change']['no_stored_tb_share']]),
        ('\\quad after a phase jump', [None, 100*jump['phase_jump']['no_stored_tb_share']]),
        ('\\quad otherwise, at most', [None, 100*max(change['other']['no_stored_tb_share'], jump['other']['no_stored_tb_share'])]),
        *[('\\quad '+(stateless_label if b == 'neural_rx_k2_trained' else LABEL[b]), [100*pooled_rescued[s, b] for s in ('no', 'pc')])
          for b in ('baseline_lslin_lmmse', 'neural_rx_k2_trained', 'baseline_lmmse_kbest', 'frozen_cold_k8')],
        ('Failed TBs after an error of the UE [\\%]', [100*pooled_repeated[s] for s in ('no', 'pc')]),
        ('\\quad proposed NRX', [100*controller['rule']['caught'], 100*step['caught']]),
        ('\\quad proposed NRX, no memory state', [100*ours['rule_caught_share'], 100*ours['caught_share']]),
        ('\\quad stateless NRX, no memory state', [100*stateless['rule_caught_share'], 100*stateless['caught_share']]),
        ('Correct TBs flagged [\\%], proposed', [100*controller['rule']['false_alarm'], 100*step['false_alarm']]),
        ('\\quad proposed NRX, no memory state', [ours['rule_late_per_100'], ours['late_per_100']]),
        ('\\quad stateless NRX, no memory state', [stateless['rule_late_per_100'], stateless['late_per_100']]),
        ('Median time to decoding [ms]', [harq['all']['regular']['median_ms_to_decoding'], harq['all']['early']['median_ms_to_decoding']]),
        ('Unneeded transmissions per 100 TBs', [0., harq['all']['early_extra_transmissions_per_100_tbs']])], 1)

print(f'{checked} checks, {bad} problems')
raise SystemExit(1 if bad else 0)
