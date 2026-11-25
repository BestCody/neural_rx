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
IEEE_REWORDED = {'phase jumps: memory', 'conclusion: phase jumps', 'abstract lead', 'abstract eight iterations',
                 'abstract three iterations', 'related work: classical estimator',
                 'related work: filter ahead after a channel change', 'time of the proposed NRX', 'budget including the fit',
                 'time before the slot', 'real time', 'modulations: setup', 'modulations: released weights per MCS',
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
claim('costs more than 1\\,dB in all three conditions', min(gap('no', 'frozen_cold_k2', 'frozen_cold_k8')) > 1, 'iteration cut')
claim('recovers about half a dB', all(.4 <= v <= .7 for v in gap('no', 'frozen_cold_k2', 'neural_rx_k2_trained')), 'retraining gain')
expect(f'The proposed NRX requires {span(lead)}\\,dB less than this retrained Large NRX', 'lead over retrained')
expect(f'by {span(three)}\\,dB, and stays within {max(behind):.2f}\\,dB of the Large NRX with $N_{{\\mathrm{{it}}}}=8$', 'three and eight iterations')
claim('The proposed NRX outperforms LS + LMMSE in all three conditions', min(gap('no', 'baseline_lslin_lmmse', 'pilot_memory')) > 0, 'LS + LMMSE')
kbest = gap('no', 'pilot_memory', 'baseline_lmmse_kbest')
claim('remains ahead of it for UMi and channel change, whereas the proposed NRX is ahead for scheduling gaps',
      kbest[0] > .1 and kbest[1] > .1 and kbest[2] < -.1, 'LMMSE + K-Best')
claim('Perf.-CSI + K-Best remains ahead of all other receivers',
      all(required[s][c]['baseline_perf_csi_kbest'] < required[s][c][m] for s in required for c in CORE
          for m in required[s][c] if m != 'baseline_perf_csi_kbest'), 'perfect CSI')
expect(f'by {span(memory)}\\,dB. Without its memory, the proposed NRX requires up to {max(-v for v in no_memory):.2f}\\,dB more than the retrained Large NRX',
       'memory gain')
pc_lead, pc_three, pc_behind, pc_memory = (gap('pc', 'neural_rx_k2_trained', 'pilot_memory'), gap('pc', 'frozen_cold_k3', 'pilot_memory'),
                                           gap('pc', 'pilot_memory', 'frozen_cold_k8'), gap('pc', 'pilot_memory_emptied', 'pilot_memory'))
expect(f'requires {span(pc_lead)}\\,dB less than the retrained Large NRX and {span(pc_three)}\\,dB less than the Large NRX with $N_{{\\mathrm{{it}}}}=3$. '
       f'It stays within {max(pc_behind):.2f}\\,dB of the Large NRX with $N_{{\\mathrm{{it}}}}=8$', 'power control')
claim('is on par with it for UMi', abs(pc_behind[0]) <= .05, 'power control: UMi')
claim('outperforms LMMSE + K-Best in all three conditions', min(gap('pc', 'baseline_lmmse_kbest', 'pilot_memory')) > 0,
      'power control: LMMSE + K-Best')
expect(f'The memory contributes {span(pc_memory)}\\,dB', 'power control: memory')
phase_required = {c: {m: round(crossing(*errors('phase', c, m)), 2) for m in OURS} for c in CORE}
phase_memory = [round(phase_required[c]['pilot_memory_emptied']-phase_required[c]['pilot_memory'], 2) for c in CORE]
phase_lead = [round(phase_required[c]['neural_rx_k2_trained']-phase_required[c]['pilot_memory'], 2) for c in CORE]
jumps = json.load(open(f'{NEW}/phase_jumps.json'))
shares = sorted(100*jumps[c]['phase_jump_share'] for c in CORE)
expect(f'With a uniformly random phase jump at every power step, which occurs before {shares[0]:.1f}\\% to '
       f'{shares[-1]:.1f}\\% of the scored TBs, the memory still contributes {span(phase_memory)}\\,dB', 'phase jumps: memory')
claim("on each UE's first three TBs after a jump, the proposed NRX fails less often than the retrained Large NRX in all conditions",
      all(jumps[c]['tbler_near_jumps']['pilot_memory'] < jumps[c]['tbler_near_jumps']['neural_rx_k2_trained'] for c in CORE),
      'phase jumps: TBs after a jump')
claim('it requires neither power consistency nor phase continuity across the stored slots',
      min(pc_memory) > .5 and min(phase_memory) > .5 and min(phase_lead) > .4, 'related work: power and phase')
claim('and keeps its memory gain with a phase jump at every power step',
      all(abs(a-b) <= .05 for a, b in zip(phase_memory, pc_memory)), 'conclusion: phase jumps')
both = lead+pc_lead
claim(f'requires {min(both):.2f} to {max(both):.2f}\\,dB less SNR than a stateless two-iteration NRX retrained on the same data', True, 'abstract lead')
claim(f'stays within {max(behind+pc_behind):.2f}\\,dB of the released eight-iteration NRX', True, 'abstract eight iterations')
claim('outperforms the released three-iteration NRX in all three test conditions', min(three+pc_three) > 0, 'abstract three iterations')
claim('outperforms the released NRXs with up to three iterations',
      min(gap(s, m, 'pilot_memory')[i] for s in required for m in ('frozen_cold_k2', 'frozen_cold_k3') for i in range(3)) > 0,
      'introduction: stateless NRXs')
claim('With two iterations, it recovers most of the accuracy that the cut from eight iterations costs',
      all(required[s][c]['frozen_cold_k2']-required[s][c]['pilot_memory'] > .5*(required[s][c]['frozen_cold_k2']-required[s][c]['frozen_cold_k8'])
          for s in required for c in CORE), 'introduction: most of the accuracy')
claim('by up to 1\\,dB, even when the NRX is retrained for two iterations',
      .9 <= max(gap('no', 'neural_rx_k2_trained', 'frozen_cold_k8')) <= 1.05, 'introduction')

# ------------------------------------------- ablation with the current TB only; the filter with a classical receiver
for s in required:
    for c in CORE:
        for m in EXTRA:
            required[s][c][m] = round(crossing(*errors(s, c, m)), 2)
current_only = gap('no', 'pilot_memory_current_tb', 'pilot_memory_emptied')
expect(f'the proposed NRX requires {span(current_only)}\\,dB more than with its memory emptied', 'current TB only')
claim('The gain stems from the earlier slots', min(current_only+gap('pc', 'pilot_memory_current_tb', 'pilot_memory_emptied')) >= 0,
      'no gain without earlier slots')
filter_gain = gap('no', 'baseline_lmmse_kbest', 'filter_kbest')
expect(f'they reduce its required SNR by {span(filter_gain)}\\,dB.', 'filter with K-Best')
over_wiener, wiener_gain = gap('no', 'wiener_kbest', 'filter_kbest'), gap('no', 'baseline_lmmse_kbest', 'wiener_kbest')
expect(f'This classical estimator trails the channel filter by {span([over_wiener[0], over_wiener[2]])}\\,dB for UMi and scheduling gaps '
       f'and by {over_wiener[1]:.2f}\\,dB for channel change, where it even falls {-wiener_gain[1]:.2f}\\,dB behind LMMSE + K-Best',
       'classical multi-slot estimator')
pc_gain, pc_over = gap('pc', 'baseline_lmmse_kbest', 'filter_kbest'), gap('pc', 'wiener_kbest', 'filter_kbest')
# Related Work and introduction: what classical multi-slot estimation cannot do, with the measured comparison
pc_wiener_gain = gap('pc', 'baseline_lmmse_kbest', 'wiener_kbest')
expect(f'a classical multi-slot estimator given each condition\'s statistics falls {-wiener_gain[1]:.2f}\\,dB behind its single-slot '
       f'version after a channel change, while our filter stays ahead of both', 'related work: classical estimator')
claim('while our filter stays ahead of both',
      all(gap(s, b, 'filter_kbest')[1] > 0 for s in ('no', 'pc') for b in ('wiener_kbest', 'baseline_lmmse_kbest')),
      'related work: filter ahead after a channel change')
claim('Their weights, however, rest on channel statistics known in advance, which do not reveal when the channel of a user changes',
      wiener_gain[1] < 0, 'introduction: classical estimators after a channel change')

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


eff = {c: {m: effective(c, m) for m in (*OURS, 'frozen_cold_k8')} for c in CORE}
expect(f'the proposed NRX requires {span([eff[c]["neural_rx_k2_trained"]-eff[c]["pilot_memory"] for c in CORE])}\\,dB less than the retrained Large NRX '
       f'and stays within {max(round(eff[c]["pilot_memory"]-eff[c]["frozen_cold_k8"], 2) for c in CORE):.2f}\\,dB of the Large NRX with $N_{{\\mathrm{{it}}}}=8$',
       'effective SNR')
doppler = {g: {m: effective('natural_umi', m, g, np.arange(-6., 14.)) for m in OURS} for g in range(3)}
gains = [doppler[g]['pilot_memory_emptied']-doppler[g]['pilot_memory'] for g in range(3)]
expect(f'it rises from {gains[0]:.2f}\\,dB at Doppler shifts below 50\\,Hz to {gains[2]:.2f}\\,dB above 175\\,Hz', 'Doppler gain')
claim('The gain of the memory grows with the Doppler shift', gains[0] < gains[1] < gains[2], 'Doppler order')

# ------------------------------------------------------------------------------------- TBs and training length
counts = []
for cond in CORE:
    files = sorted(glob.glob(f'{NEW}/sweep/{cond}/pilot_memory/episode_*.npz'))
    assert len(files) == 32
    counts.append(sum(int(np.load(path)['active'][16:].sum()) for path in files))
expect(f'over {min(counts):,} to {max(counts):,} TBs per condition and SNR point'.replace(',', '{,}'), 'TBs per SNR point')
stage1 = json.load(open('experiments/20261117_neural_rx_training/results/best.json'))
later = [v for update, v in stage1['development'].items() if int(update) > stage1['update']]
claim('whose validation TBLER had stopped improving', len(later) >= 3 and min(later) >= stage1['core_tbler'], 'stage 1 trained out')

# --------------------------------------------------------------------------------------------------- latency
aw = np.load(f'{TRT}/fused_a100_samples.npz')
ms = {k: float(np.median(aw[f'events_{k}'])) for k in ('large_k2', 'large_k3', 'large_k8', 'proposed', 'fit_only',
                                                        'proposed_and_fit', 'proposed_overlapped')}
claim('median of 2{,}000 runs on an NVIDIA A100 GPU', 'A100' in json.load(open(f'{TRT}/fused_a100.json'))['gpu']
      and all(len(aw[f'events_{k}']) == 2000 for k in ms), 'GPU and number of runs')
claim('with TensorRT~10.3 in FP16', json.load(open(f'{TRT}/fused_a100.json'))['tensorrt'].startswith('10.3.'), 'TensorRT version')
expect(f'takes {ms["large_k2"]:.2f}\\,ms, and each further CGNN iteration adds {(ms["large_k8"]-ms["large_k2"])/6:.2f}\\,ms', 'iteration time')
claim(f'$N_{{\\mathrm{{it}}}}=2$ meets the 1\\,ms budget, which $N_{{\\mathrm{{it}}}}=3$ exceeds ({ms["large_k3"]:.2f}\\,ms)',
      ms['large_k2'] < 1 < ms['large_k3'], 'budget of the Large NRX')
claim(f'The proposed NRX takes {ms["proposed"]:.2f}\\,ms, i.e., {ms["proposed"]-ms["large_k2"]:.2f}\\,ms more than the Large NRX with '
      '$N_{\\mathrm{it}}=2$', ms['proposed'] < 1, 'time of the proposed NRX')
claim(f'the GPU work per slot is {ms["proposed_and_fit"]:.2f}\\,ms, and if it runs at the same time as the slot, the slot takes '
      f'{ms["proposed_overlapped"]:.2f}\\,ms. The proposed NRX thus meets the 1\\,ms budget in either case',
      max(ms['proposed'], ms['proposed_and_fit'], ms['proposed_overlapped']) < 1, 'budget including the fit')
expect(f'are computed before the slot arrives, which takes {ms["fit_only"]:.2f}\\,ms', 'time before the slot')
for phrase in ('It meets the 1\\,ms real-time budget, which the three-iteration NRX exceeds',
               'show that it remains real-time capable, whereas the three-iteration NRX exceeds the 1\\,ms budget',
               'by which it remains real-time capable, unlike the three-iteration NRX'):
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


expect(f'by {span([fallback["no", c] for c in CORE])}\\,dB, at the cost of a second receiver', 'fallback without power control')
expect(f'{percent([unfixable["no", c] for c in CORE])} of the failed TBs fail for every receiver without perfect CSI, and '
       f'{percent([repeated["no", c] for c in CORE])} follow an error', 'unfixable without power control')
expect(f'With power control, LMMSE + K-Best gains at most {max(fallback["pc", c] for c in CORE):.2f}\\,dB, '
       f'and only {percent([repeated["pc", c] for c in CORE])} of the errors follow an error of the same UE', 'fallback with power control')
# fallbacks a real-time BS could run, recomputed from the stored per-TB results at the operating points
rescued = []
for s in ('no', 'pc'):
    for c in CORE:
        snrs, alone = errors(s, c, 'pilot_memory')
        point = int(np.argmin(np.abs(np.asarray(snrs)-crossing(snrs, alone))))
        for backup in ('baseline_lslin_lmmse', 'pilot_memory_emptied', 'neural_rx_k2_trained', 'frozen_cold_k2'):
            rescued.append(1-errors(s, c, 'pilot_memory', [backup])[1][point]/alone[point])
claim('a fallback that a real-time BS could run, LS + LMMSE or a stateless NRX with $N_{\\mathrm{it}}=2$, rescues less than 1\\% of the failed TBs',
      max(rescued) < .01, 'real-time fallbacks')
# when the channel filter falls back from its memory (experiments/20261122_robustness/fallback_k.py)
probe = json.load(open('experiments/20261122_robustness/results/fallback_k_summary.json'))
change, jump = probe['power_control'], probe['power_control_phase']
expect('The receiver itself decides when to fall back from its memory', 'filter fallback')
claim("it uses none at a channel change or a phase jump for more than 90\\% of the UEs' DMRS symbols",
      change['change']['no_stored_tb_share'] > .9 and jump['phase_jump']['no_stored_tb_share'] > .9
      and change['error_mismatches'] == 0 and jump['error_mismatches'] == 0 and change['sequences'] == jump['sequences'] == 16,
      'filter fallback at changes and jumps')
claim('and otherwise for less than 1\\%', change['other']['no_stored_tb_share'] < .01 and jump['other']['no_stored_tb_share'] < .01,
      'filter fallback elsewhere')
claim('a decision the classical multi-slot estimator cannot make', wiener_gain[1] < 0 and over_wiener[1] == max(over_wiener),
      'classical estimator after a change')
controller = json.load(open(f'{NET}/controller.json'))
step = controller['step_by_step']['CRC history and receiver evidence, budget of the rule']
single = controller['single_check']
expect(f'catches {round(100*controller["rule"]["caught"])}\\% of the errors in the test sequences', 'rule')
expect(f'the controller catches {round(100*step["caught"])}\\% of the errors instead', 'controller')
claim('about four fifths of them already at the first check', .75 <= step['caught_at']['check 1']/step['caught'] <= .85, 'check 1 share')
claim(f'flags {100*step["false_alarm"]:.1f}\\% of the correct TBs in the test sequences, as does this rule',
      f'{100*step["false_alarm"]:.1f}' == f'{100*controller["rule"]["false_alarm"]:.1f}', 'false alarms')
expect(f'It also catches {round(100*step["caught_first_failures"])}\\% of the errors that do not follow an error', 'first failures')
claim('the memory state and the refined estimate add little',
      single['check 1, + filtered estimate']['caught']-single['check 1, any receiver: noise and current channel strength']['caught'] < .03, 'adds little')
shares = [single[name]['caught'] for name in ('check 1, any receiver: noise and current channel strength',
          'check 1, + filtered estimate', 'check 2, + LLRs after detection', 'check 3, + parity checks after 8 BP iterations')]
claim("Among the receiver's evidence, the LLRs matter most",
      shares[2]-shares[1] == max(b-a for a, b in zip(shares, shares[1:])), 'LLRs matter most')
compare = json.load(open(f'{NET}/compare_receivers.json'))['receivers']
ours, stateless = compare['proposed'], compare['stateless']
expect(f'Without the memory state, the controller still catches {round(100*ours["caught_share"])}\\% of the errors of the proposed NRX, '
       f'and {round(100*stateless["caught_share"])}\\% of those of the stateless NRX retrained with $N_{{\\text{{it}}}}=2$, against '
       f'{round(100*stateless["rule_caught_share"])}\\% for the rule', 'controller on the stateless NRX')
claim('As that NRX fails about twice as often at the same SNR points',
      1.8 <= stateless['failures_per_100']/ours['failures_per_100'] <= 2.2, 'twice as many failures')
expect(f'{stateless["late_per_100"]:.1f} errors per 100 TBs remain for retransmission after the CRC fails, against '
       f'{ours["late_per_100"]:.1f} with the proposed NRX', 'late retransmissions')
claim('leaving six times fewer for after the CRC fails than a stateless NRX',
      round(stateless['late_per_100']/ours['late_per_100']) == 6, 'six times fewer late')
claim(f'request {round(100*step["caught"])}\\% of the needed retransmissions before decoding finishes, compared with {round(100*controller["rule"]["caught"])}\\% for the CRC history alone',
      True, 'abstract controller')

# --------------------------------------------------------------------------------------------- robustness (Modal)
section = None
ROB = 'experiments/20261122_robustness/results'
robust = json.load(open(f'{ROB}/summary.json'))
for test in ('cfo', 'tdd_dddsu'):
    assert all(not v['incomplete'] and len(v['required_snr_db']) >= 3 for v in robust[test].values()), test
cfo, tdd = robust['cfo'].values(), robust['tdd_dddsu'].values()
loss = lambda v, r: v['required_snr_db'][r]-v['reference_snr_db'][r]
expect('Except in the last tests with power control, we assume, as for DMRS bundling,', 'assumptions: robustness tests')
claim('With a frequency offset of $\\pm 0.1$\\,ppm per UE, the most 3GPP allows',
      'CFO_PPM = 0.1' in open('experiments/20261122_robustness/banks.py').read(), 'CFO: offset')
claim('the proposed NRX loses less than 0.1\\,dB and the retrained Large NRX about 0.15\\,dB',
      all(loss(v, 'pilot_memory') < .1 and .13 <= loss(v, 'neural_rx_k2_trained') <= .18 for v in cfo), 'CFO: losses')
claim('as the channel filter tracks the rotation',
      all(v['pilot_memory_memory_gain_db'] > v['reference_memory_gain_db'] for v in cfo), 'CFO: memory gain grows')
expect('With the TDD pattern DDDSU, i.e., one uplink slot every 2.5\\,ms,', 'TDD: pattern')
claim('the memory keeps more than 70\\% of its gain',
      all(v['pilot_memory_memory_gain_db'] > .7*v['reference_memory_gain_db'] for v in tdd), 'TDD: memory share')
claim('or at least 0.6\\,dB if the eight-slot limit counts uplink slots',
      all(v['pilot_memory_uplink_memory_gain_db'] >= .6 for v in tdd), 'TDD: uplink counting')
claim('and the proposed NRX still leads the retrained Large NRX by at least 0.4\\,dB',
      all(v['pilot_memory_lead_over_retrained_db'] >= .4 for v in tdd), 'TDD: lead over retrained')
section = 'network'                                          # simulated HARQ
harq = json.load(open(f'{ROB}/harq_summary.json'))
conditions = [harq[c] for c in ('natural_umi', 'controlled_topology_change', 'schedule_gap_reentry')]
claim('Simulating the retransmissions with Chase combining',
      harq['all']['decision_mismatches'] == 0 and all(c['compared'] > 100 for c in conditions), 'HARQ: decisions reproduced')
claim('early requests let the BS decode a failed TB one slot earlier in median',
      harq['all']['median_slots_saved'] == 1 and all(c['median_slots_saved'] == 1 for c in conditions), 'HARQ: one slot')
claim('for decoding times from 0.25 to 1\\,ms',
      all(v == 1 for v in harq['all']['request_slots_saved_per_caught_error'].values()), 'HARQ: decoding times')
claim('at the cost of about two unneeded transmissions per 100 TBs',
      1.5 <= harq['all']['early_extra_transmissions_per_100_tbs'] <= 2.5, 'HARQ: extra transmissions')
section = None
expect('Future work includes measured data and multiple training seeds.', 'future work')


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
claim('the memory contributes 0.7 to 1.1\\,dB', f'{min(gains):.1f}' == '0.7' and f'{max(gains):.1f}' == '1.1', 'modulations: memory')
claim('the proposed NRX outperforms the retrained Large NRX and the Large NRX with $N_{\\mathrm{it}}=3$ in all cases, with and without power control',
      all(snr(*k, 'pilot_memory') < min(snr(*k, 'neural_rx_k2_trained'), snr(*k, 'frozen_cold_k3')) for k in cells),
      'modulations: retrained and three iterations')
qpsk = [k for k in cells if k[0] == 'nrx_large_qpsk']
claim('With QPSK, it also outperforms the Large NRX with $N_{\\mathrm{it}}=8$ in all cases',
      all(snr(*k, 'pilot_memory') < snr(*k, 'frozen_cold_k8') for k in qpsk), 'modulations: QPSK and eight iterations')
claim('and, with QPSK, in all conditions', all(snr(*k, 'pilot_memory') < snr(*k, 'frozen_cold_k8') for k in qpsk),
      'abstract: QPSK and eight iterations')
behind = {k: snr(*k, 'pilot_memory')-snr(*k, 'frozen_cold_k8') for k in cells if k[0] == 'nrx_large_64qam'}
claim('with 64-QAM, it stays within 0.15\\,dB of it with power control',
      all(v <= .15 for k, v in behind.items() if k[1] == 'power_control'), 'modulations: 64-QAM with power control')
claim('but trails it by about 0.7\\,dB after scheduling gaps without power control',
      round(behind['nrx_large_64qam', 'sweep', 'schedule_gap_reentry'], 1) == .7
      and max(behind, key=behind.get) == ('nrx_large_64qam', 'sweep', 'schedule_gap_reentry'), 'modulations: 64-QAM after gaps')
for phrase in ('MU-MIMO with QPSK, 16-QAM, and 64-QAM over sequences', 'an evaluation with QPSK, 16-QAM, and 64-QAM in which'):
    expect(phrase, 'modulations: introduction and conclusion')

# --------------------------------------------------------------------------------------------- IEEE version
if IEEE:
    lslin = {label: [snr(label, st, c, 'baseline_lslin_lmmse')-snr(label, st, c, 'pilot_memory')
                     for st in ('sweep', 'power_control') for c in CORE] for label in mods}
    mod_lead = [snr(*k, 'neural_rx_k2_trained')-snr(*k, 'pilot_memory') for k in cells]
    mod_three = [snr(*k, 'frozen_cold_k3')-snr(*k, 'pilot_memory') for k in cells]
    all_lead = lead+pc_lead+mod_lead
    claim('requires 0.5 to 0.9\\,dB less SNR than a stateless two-iteration NRX retrained on the same data',
          f'{min(all_lead):.1f}' == '0.5' and f'{max(all_lead):.1f}' == '0.9', 'IEEE abstract: lead, three modulations')
    claim('outperforms the released three-iteration NRX in all cases, with and without power control',
          min(three+pc_three+mod_three) > 0, 'IEEE abstract: three iterations')
    claim('with QPSK, it also outperforms the released eight-iteration NRX in all cases',
          all(snr(*k, 'pilot_memory') < snr(*k, 'frozen_cold_k8') for k in qpsk), 'IEEE abstract: QPSK')
    claim('It keeps most of its memory gain with phase jumps, frequency offsets, and TDD uplink slots',
          min(a/b for a, b in zip(phase_memory, pc_memory)) > .9
          and all(v['pilot_memory_memory_gain_db'] > .5*v['reference_memory_gain_db'] for t in ('cfo', 'tdd_dddsu')
                  for v in robust[t].values()), 'IEEE abstract: robustness')
    for phrase in ('it meets the 1\\,ms real-time budget, which the three-iteration NRX exceeds',
                   'remains real-time capable, whereas the three-iteration NRX exceeds the 1\\,ms budget',
                   'while it remains real-time capable, unlike the three-iteration NRX'):
        claim(phrase, ms['proposed'] < 1 < ms['large_k3'], 'IEEE: real time')
    expect(f'With a uniformly random phase jump at every power step, the memory still contributes {span(phase_memory)}\\,dB',
           'IEEE: phase jumps')
    expect(f"after a channel change, it stays ahead of a classical multi-slot estimator given each condition's statistics, "
           f"which even falls {-wiener_gain[1]:.2f}\\,dB behind its single-slot version", 'IEEE: related work, classical estimator')
    claim('after a channel change, it stays ahead of a classical multi-slot estimator',
          all(gap(s_, b, 'filter_kbest')[1] > 0 for s_ in ('no', 'pc') for b in ('wiener_kbest', 'baseline_lmmse_kbest')),
          'IEEE: related work, filter ahead')
    expect(f'are computed before the slot arrives, in {ms["fit_only"]:.2f}\\,ms for both UEs', 'IEEE: time before the slot')
    claim(f'The proposed NRX takes {ms["proposed"]:.2f}\\,ms, and the GPU work per slot including this computation is '
          f'{ms["proposed_and_fit"]:.2f}\\,ms, or {ms["proposed_overlapped"]:.2f}\\,ms if it runs at the same time as the slot, so '
          f'it meets the 1\\,ms budget either way', max(ms['proposed'], ms['proposed_and_fit'], ms['proposed_overlapped']) < 1,
          'IEEE: budget')
    claim("On the TB of a channel change or a phase jump, the channel filter uses no stored TB for more than 90\\% of the UEs' "
          "DMRS symbols, and otherwise for less than 1\\%",
          change['change']['no_stored_tb_share'] > .9 and jump['phase_jump']['no_stored_tb_share'] > .9
          and change['other']['no_stored_tb_share'] < .01 and jump['other']['no_stored_tb_share'] < .01
          and change['error_mismatches'] == jump['error_mismatches'] == 0, 'IEEE: filter fallback')
    claim('For QPSK (MCS 9) and 64-QAM (MCS 19), we train and evaluate all NRXs in the same way, starting from the released '
          'weights for each MCS',
          all(json.load(open(f'{MOD}/{label}/stage1/recipe.json'))['initializer_checkpoint'] == f'models/pretrained/{label}_weights'
              for label in mods), 'IEEE: modulations, setup')
    claim('With QPSK and 64-QAM, the memory contributes 0.7 to 1.1\\,dB', f'{min(gains):.1f}' == '0.7' and f'{max(gains):.1f}' == '1.1',
          'IEEE: modulations, memory')
    claim('LS + LMMSE needs at least 0.8\\,dB more with 64-QAM and 2.3\\,dB more with QPSK',
          min(lslin['nrx_large_64qam']) >= .8 and min(lslin['nrx_large_qpsk']) >= 2.3, 'IEEE: modulations, LS + LMMSE')
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
    claim(f'the memory lowers TBLER in {len(lower)} of the {len(cases)} cases', True, 'further datasets: memory')
    claim(f'outperforms the Large NRX with $N_{{\\mathrm{{it}}}}=8$ in {len(ahead)} ' + ('cases' if IEEE else f'of the {len(cases)} cases'), True,
          'further datasets: ahead of eight iterations')
    claim('for both TDL channels and the UMa channels on both datasets' if IEEE else 'namely for both TDL channels and for the UMa channels on both datasets',
          set(ahead) == {(ds, name) for ds in FURTHER for name in ('historical_tdl_b100', 'historical_tdl_c300', 'natural_uma')},
          'further datasets: which cases')
    claim('which it does not without its memory' if IEEE else 'Without its memory, it is behind in all six',
          len(ahead) == 6 and all(rates[k+('frozen_cold_k8',)] < rates[k+('candidate_reset',)] for k in ahead),
          'further datasets: behind without the memory')
    claim(f'remains ahead in {len(behind)} cases', True, 'further datasets: eight iterations ahead')
    relative = sorted(behind, key=lambda k: rates[k+('candidate_persistent',)]/rates[k+('frozen_cold_k8',)], reverse=True)
    claim('most clearly for closely spaced LOS and near-far UEs', all(k[1] in ('close_los', 'near_far') for k in relative[:3]),
          'further datasets: largest deficits')
else:
    print('further datasets: not checked (results or paragraph missing)')

print(f'{checked} checks, {bad} problems')
raise SystemExit(1 if bad else 0)
