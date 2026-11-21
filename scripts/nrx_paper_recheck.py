"""Second pass: recompute the paper's evaluation values from the raw per-sequence files with code that is separate
from scripts/nrx_paper_numbers.py, and compare them with the text of paper/*.tex.

Every check is either a number that must appear in the text as printed (expect) or a statement that must appear
verbatim and that the recomputed data must support (claim). Exit code 1 if anything is missing or does not hold.

    python scripts/nrx_paper_recheck.py
"""
import glob
import json
import math
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
tex = re.sub(r'\\input\{([^}]+)\}', lambda m: open(f'paper/{m.group(1)}.tex', encoding='utf8').read(),
             open('paper/main.tex', encoding='utf8').read())
tex = re.sub(r'(?m)(?<!\\)%.*$', '', tex)
text = re.sub(r'\s+', ' ', tex)
bad = checked = 0


def expect(string, label):
    global bad, checked
    checked += 1
    if string not in text:
        bad += 1; print('MISSING', label, '|', string)


def claim(phrase, holds, label):
    """A statement: it must appear verbatim, and the recomputed data must support it."""
    global bad, checked
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
expect(f'With a uniformly distributed phase jump at every power step, which occurs before {shares[0]:.1f}\\% to '
       f'{shares[-1]:.1f}\\% of the scored TBs, the memory still contributes {span(phase_memory)}\\,dB', 'phase jumps: memory')
claim("on each UE's first three TBs after a jump, the proposed NRX fails less often than the retrained Large NRX in all three conditions",
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
expect(f'the channel filter gains at most {max(pc_gain):.2f}\\,dB', 'power control: filter with K-Best')
claim('leads the classical estimator except for scheduling gaps', pc_over[0] > 0 and pc_over[1] > 0 and pc_over[2] < 0,
      'power control: classical estimator')

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
expect(f'takes {ms["large_k2"]:.2f}\\,ms, each further CGNN iteration adds {(ms["large_k8"]-ms["large_k2"])/6:.2f}\\,ms', 'iteration time')
claim('takes three times as long as $N_{\\mathrm{it}}=2$', 2.8 < ms['large_k8']/ms['large_k2'] < 3.3, 'eight vs two')
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
expect(f'the fallback reduces the required SNR by at most {max(fallback["pc", c] for c in CORE):.2f}\\,dB, and '
       f'{percent([unfixable["pc", c] for c in CORE])} of the failed TBs', 'fallback with power control')
expect(f'Only {percent([repeated["pc", c] for c in CORE])} of the errors then follow an error of the same UE', 'CRC history with power control')
controller = json.load(open(f'{NET}/controller.json'))
step = controller['step_by_step']['CRC history and receiver evidence, budget of the rule']
single = controller['single_check']
expect(f'catches {round(100*controller["rule"]["caught"])}\\% of the errors in the test sequences', 'rule')
expect(f'the controller catches {round(100*step["caught"])}\\% of the errors instead', 'controller')
claim('about four fifths of them already at the first check', .75 <= step['caught_at']['check 1']/step['caught'] <= .85, 'check 1 share')
claim(f'flags {100*step["false_alarm"]:.1f}\\% of the correct TBs in the test sequences, as does this rule',
      f'{100*step["false_alarm"]:.1f}' == f'{100*controller["rule"]["false_alarm"]:.1f}', 'false alarms')
expect(f'It also catches {round(100*step["caught_first_failures"])}\\% of the errors that do not follow an error', 'first failures')
claim('catch about a fifth of the errors after channel estimation',
      .17 <= single['check 1, any receiver: noise and current channel strength']['caught'] <= .23, 'fifth')
claim('the memory state and the refined estimate add little',
      single['check 1, + filtered estimate']['caught']-single['check 1, any receiver: noise and current channel strength']['caught'] < .03, 'adds little')
expect(f'the LLRs raise the share to {round(100*single["check 2, + LLRs after detection"]["caught"])}\\%, and the parity checks to '
       f'{round(100*single["check 3, + parity checks after 8 BP iterations"]["caught"])}\\%', 'LLRs and parity checks')
compare = json.load(open(f'{NET}/compare_receivers.json'))['receivers']
ours, stateless = compare['proposed'], compare['stateless']
expect(f'Without the memory state, the controller still catches {round(100*ours["caught_share"])}\\% of the errors of the proposed NRX, '
       f'and {round(100*stateless["caught_share"])}\\% of those of the stateless NRX retrained with $N_{{\\text{{it}}}}=2$, against '
       f'{round(100*stateless["rule_caught_share"])}\\% for the rule', 'controller on the stateless NRX')
claim('As that NRX fails about twice as often at the same SNR points',
      1.8 <= stateless['failures_per_100']/ours['failures_per_100'] <= 2.2, 'twice as many failures')
expect(f'{stateless["late_per_100"]:.1f} errors per 100 TBs remain for retransmission after the CRC fails, against '
       f'{ours["late_per_100"]:.1f} with the proposed NRX', 'late retransmissions')
claim('leaving six times fewer for after the CRC fails than with a stateless NRX',
      round(stateless['late_per_100']/ours['late_per_100']) == 6, 'six times fewer late')
claim(f'request {round(100*step["caught"])}\\% of the needed retransmissions before decoding finishes, compared with {round(100*controller["rule"]["caught"])}\\% for the CRC history alone',
      True, 'abstract controller')

# --------------------------------------------------------------------------------------------- further datasets
FURTHER = {'first': 'experiments/20261118_further_datasets/results/v2/errors.npz',
           'second': 'experiments/20261118_further_datasets/results/seed86/errors.npz'}
if all(glob.glob(path) for path in FURTHER.values()) and 'further independent datasets' in text:
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
    tdl = ('first', 'historical_tdl_b100')
    expect(f"from {100*rates[tdl+('candidate_reset',)]:.1f}\\% to {100*rates[tdl+('candidate_persistent',)]:.1f}\\% for a TDL channel",
           'further datasets: TDL example')
    claim('for closely spaced LOS UEs on one dataset, TBLER is unchanged', [k[1] for k in same] == ['close_los']
          and len(lower)+len(same) == len(cases), 'further datasets: unchanged')
    claim(f'outperforms the Large NRX with $N_{{\\mathrm{{it}}}}=8$ in {len(ahead)} of the {len(cases)} cases', True,
          'further datasets: ahead of eight iterations')
    claim('namely for both TDL channels and for the UMa channels on both datasets',
          set(ahead) == {(ds, name) for ds in FURTHER for name in ('historical_tdl_b100', 'historical_tdl_c300', 'natural_uma')},
          'further datasets: which cases')
    example = ('first', 'historical_tdl_c300')
    expect(f"with a TBLER of {100*rates[example+('candidate_persistent',)]:.1f}\\% instead of {100*rates[example+('frozen_cold_k8',)]:.1f}\\%",
           'further datasets: example ahead')
    claim('Without its memory, it is behind in all six',
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
