from __future__ import annotations

from artifact_provenance import TRAINING_SEED_NAMESPACE

import hashlib
import json

import numpy as np
import sionna as sn
import tensorflow as tf

from geometry_mumimo_data import GeometryAwareMUMIMOGenerator
from temporal_training_data import TemporalTrainingDataGenerator
from temporal_generalization_protocol import ChannelProfileSpec
from .training_render import TrainingRenderer
from .training_options import DEFAULT_RENDER_BATCH_TBS


def validate_history_settings(recipe):
    lengths = recipe["episode_tbs"]
    if not lengths or any(type(p) is not int or p < 3 for p in lengths):
        raise ValueError("Episodes need at least three TBs")
    for key in ("cold_tbs", "paired_suffix_tbs"):
        if type(recipe[key]) is not int or recipe[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if recipe["paired_suffix_tbs"] >= min(lengths):
        raise ValueError("Paired episodes require history before their common suffix")
    gaps = recipe["schedule_gap_tbs"]
    if not gaps or any(type(g) is not int or g < 1 for g in gaps):
        raise ValueError("Scheduling gaps must have positive lengths")
    for key in ("cold_fraction", "paired_history_fraction", "schedule_reorder_probability",
                "schedule_probability", "channel_change_probability", "power_imbalance_probability",
                "weak_link_noise_probability"):
        if not np.isfinite(recipe[key]) or not 0 <= recipe[key] <= 1:
            raise ValueError(f"Invalid probability: {key}")
    if not set(recipe["channel_families"]).issubset({"umi", "uma", "tdl"}):
        raise ValueError("Unknown training channel family")
    for key in ("schedule_switch_probability_range", "schedule_gap_probability_range"):
        values = recipe[key]
        if len(values) != 2 or not np.isfinite(values).all() or not 0 <= values[0] <= values[1] <= 1:
            raise ValueError(f"Invalid scheduling probability range: {key}")
    if not recipe["tdl_profiles"] or not set(recipe["tdl_profiles"]).issubset({"A", "B", "C", "D", "E"}):
        raise ValueError("Invalid TDL profiles")
    ranges = [recipe["delay_spread_s"], recipe["power_gap_db"], *recipe["doppler_strata_hz"]]
    if not recipe["doppler_strata_hz"] or any(len(r) != 2 or not np.isfinite(r).all()
            or not 0 <= r[0] <= r[1] for r in ranges):
        raise ValueError("Invalid channel sampling ranges")
    noise = recipe["noise_schedule"]
    bounds, core = noise["range_db"], noise["core_range_db"]
    if len(bounds) != 2 or len(core) != 2 or not np.isfinite([*bounds, *core]).all():
        raise ValueError("Noise ranges must have two finite bounds")
    if not bounds[0] <= core[0] < core[1] <= bounds[1]:
        raise ValueError("Noise core range must lie within the overall range")
    if any(not 0 <= noise[key] <= 1 for key in ("core_probability", "change_probability")):
        raise ValueError("Noise probabilities must lie in [0, 1]")
    if not np.isfinite(noise["max_change_db"]) or noise["max_change_db"] <= 0:
        raise ValueError("Noise changes need a positive finite bound")


def domain_seed(seed, update, example, domain):
    raw = json.dumps([TRAINING_SEED_NAMESPACE, int(seed), int(update), int(example), domain]).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") % (2**31 - 1)


def update_width(recipe, seed, update):
    rng = np.random.RandomState(domain_seed(seed, update, 0, "width"))
    return int(rng.choice(recipe["prbs"], p=recipe["prb_probabilities"]))


def noise_schedule(settings, length, rng, change_rng):
    bounds = settings["core_range_db"] if rng.rand() < settings["core_probability"] else settings["range_db"]
    initial = float(rng.uniform(*bounds))
    values = np.full((1, length), initial, np.float32)
    if length > 1 and change_rng.rand() < settings["change_probability"]:
        low, high = settings["range_db"]
        span = settings["max_change_db"]
        after = float(change_rng.uniform(max(low, initial-span), min(high, initial+span)))
        start = int(change_rng.randint(1, length))
        values[:, start:] = after
    return values


def sample_schedule(recipe, length, pool_size, users, rng, varying):
    assigned = rng.permutation(pool_size)[:users]
    ids = np.tile(assigned, (1, length, 1)).astype(np.int32)
    active = np.ones((1, length, users), np.float32)
    switch = float(rng.uniform(*recipe["schedule_switch_probability_range"])) if varying else 0.
    gap = float(rng.uniform(*recipe["schedule_gap_probability_range"])) if varying else 0.
    unavailable_until = np.zeros(pool_size, np.int32)
    events = []
    for t in range(1, length):
        if varying:
            for identity in assigned:
                if t >= unavailable_until[identity] and rng.rand() < gap:
                    duration = int(rng.choice(recipe["schedule_gap_tbs"]))
                    unavailable_until[identity] = t+duration
                    events.append(dict(ue_id=int(identity), start_tb=t, duration_tbs=duration))
            if rng.rand() < switch:
                assigned = rng.permutation(pool_size)[:users]
            if rng.rand() < recipe["schedule_reorder_probability"]:
                assigned = assigned[rng.permutation(users)]
        ids[0, t] = assigned
        active[0, t] = t >= unavailable_until[assigned]
    return ids, active, dict(varying=varying, switch_probability=switch,
                            gap_probability=gap, gaps=events)


class TrainingEpisodes:
    def __init__(self, contexts, recipe, seed, render_batch_tbs=DEFAULT_RENDER_BATCH_TBS):
        self.contexts = contexts
        self.recipe = recipe
        self.seed = int(seed)
        self.generators = {}
        self.renderers = {}
        self.render_batch_tbs = render_batch_tbs

    def _generator(self, width, family, speed):
        key = (width, family, speed)
        if key not in self.generators:
            p, e2e = self.contexts[width]
            low, high = self.recipe["doppler_strata_hz"][speed]
            common = dict(ue_pool_size=self.recipe["memory_capacity"],
                          advance_slot_pilots=True, random_slot_offset=True)
            if family in ("umi", "uma"):
                gen = GeometryAwareMUMIMOGenerator(
                    p, e2e, geometry_scenario=family, min_doppler_hz=low,
                    max_doppler_hz=high, **common)
            else:
                profiles = tuple(ChannelProfileSpec(
                    f"TDL-{name}", "tdl", name, *self.recipe["delay_spread_s"], low, high)
                    for name in self.recipe["tdl_profiles"])
                gen = TemporalTrainingDataGenerator(
                    p, e2e, channel_profiles=profiles, event_channel_profiles=profiles,
                    randomize_profile_assignment=True, **common)
            self.generators[key] = gen
        return self.generators[key]

    def sample(self, update, example, width):
        return self.render(self.plan(update, example, width))

    def plan(self, update, example, width):
        seed = lambda domain: domain_seed(self.seed, update, example, domain)
        rng = np.random.RandomState(seed("draw"))
        recipe = self.recipe
        cold = bool(rng.rand() < recipe["cold_fraction"])
        paired = bool(rng.rand() < recipe["paired_history_fraction"] and not cold)
        family = str(rng.choice(recipe["channel_families"], p=recipe["channel_probabilities"]))
        speed = int(rng.randint(len(recipe["doppler_strata_hz"])))
        scheduling = bool(rng.rand() < recipe["schedule_probability"] and not cold)
        changing = bool(rng.rand() < recipe["channel_change_probability"] and not cold)
        length = int(recipe["cold_tbs"] if cold else
                     rng.choice(recipe["episode_tbs"], p=recipe["episode_probabilities"]))
        gen = self._generator(width, family, speed)
        pool_size, users = recipe["memory_capacity"], gen.num_scheduled_tx
        streams = {name: seed(name) for name in ("channel", "payload", "noise", "schedule")}
        sn.config.seed = streams["channel"]
        gen._activate_seed_batch(streams)
        ids, active, schedule = sample_schedule(recipe, length, pool_size, users,
                                               np.random.RandomState(streams["schedule"]), scheduling)
        original_ids = ids[0, 0].copy()
        change_tb = change_id = None
        regime = None
        if changing:
            regime = np.zeros((1, length, pool_size), bool)
            change_tb = int(rng.randint(1, length))
            change_id = int(rng.choice(pool_size if scheduling else original_ids))
            regime[:, change_tb:, change_id] = True
        pool = gen.sample_pool_channel(1, length, None if regime is None else tf.constant(regime))
        levels = np.zeros(pool_size, np.float32)
        imbalanced = bool(rng.rand() < recipe["power_imbalance_probability"])
        if imbalanced:
            span = float(rng.uniform(*recipe["power_gap_db"]))
            levels = rng.uniform(-span/2, span/2, pool_size).astype(np.float32)
            levels[original_ids] = rng.permutation(np.linspace(-span/2, span/2, users))
            pool *= tf.cast(tf.reshape(10.**(levels/20), [1, 1, 1, 1, pool_size, 1, 1, 1]), pool.dtype)
        channel = gen.gather_scheduled_channel(pool, tf.constant(ids))
        slots = (int(rng.randint(int(gen.p.pilots.shape[0]))) + np.arange(length)) % int(gen.p.pilots.shape[0])
        ebnos = noise_schedule(recipe["noise_schedule"], length, rng,
                              np.random.RandomState(seed("noise_profile")))
        weak_reference = bool(imbalanced and rng.rand() < recipe["weak_link_noise_probability"])
        if weak_reference:
            ebnos = ebnos-float(levels.min())
        metadata = {
            "update": update, "example": example, "width": width, "family": family,
            "speed_stratum": speed, "schedule": schedule, "channel_change": changing,
            "cold": cold, "paired": paired,
            "num_tbs": length, "paired_suffix_tbs": recipe["paired_suffix_tbs"] if paired else 0,
            "change_tb": change_tb, "change_ue_id": change_id,
            "power_db_by_id": levels.tolist(),
            "noise_reference": "weakest_pool_ue" if weak_reference else "common",
            "nominal_weak_ebno_db_by_tb": (ebnos[0]+float(levels.min())).tolist(),
            "ebno_db": float(ebnos[0, 0]), "ebno_db_by_tb": ebnos[0].tolist(),
            "noise_profile_seed": seed("noise_profile"), "seeds": streams,
            "paired_payload_seed": seed("paired_payload"), "paired_noise_seed": seed("paired_noise"),
        }
        return dict(channel=channel, active=active, ue_ids=ids, slot_number=slots,
                    ebno_db=ebnos, metadata=metadata)

    def render(self, plan):
        metadata = plan["metadata"]
        width = int(metadata["width"])
        if width not in self.renderers:
            p, e2e = self.contexts[width]
            self.renderers[width] = TrainingRenderer(TemporalTrainingDataGenerator(p, e2e), self.render_batch_tbs)
        render = self.renderers[width]
        channel = tf.convert_to_tensor(plan["channel"])
        active, ids, slots, ebnos = (plan[k] for k in ("active", "ue_ids", "slot_number", "ebno_db"))
        target = int(metadata["num_tbs"]) - int(metadata["paired_suffix_tbs"])
        a = render(channel, active, ids, slots, ebnos,
                   metadata["seeds"]["payload"], metadata["seeds"]["noise"])
        if metadata["paired"]:
            b = render(channel[:, :target], active[:, :target], ids[:, :target],
                       slots[:target], ebnos[:, :target],
                       metadata["paired_payload_seed"], metadata["paired_noise_seed"])
            for name in ("bits", "y", "ls", "active", "ue_ids", "no", "channel_targets", "coded"):
                b[name] = tf.concat([b[name], a[name][:, target:]], 1)
                a[name] = tf.concat([a[name], b[name]], 0)
        a.update(coded=tf.stop_gradient(a["coded"]), metadata=metadata)
        return a
