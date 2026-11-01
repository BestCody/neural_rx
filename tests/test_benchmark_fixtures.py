import numpy as np

from temporal_benchmark_runtime import BenchmarkEpisode


class ZeroAdapter:

    def __init__(self, manifest):
        self.manifest = manifest
        self.state = 0

    def start_episode(self):
        self.state = 0

    def start_episode_batch(self, batch_size):
        self.state = 0

    def reset_memory(self):
        self.state = 0

    def state_is_empty(self):
        return self.state == 0

    def decode_tb(self, inputs):
        users = len(inputs["ue_ids"])
        self.state += 1
        return {
            "hard_bits": np.zeros((users, 4), bool),
            "crc_status": np.ones(users, bool),
            "logits": np.zeros((users, 4), np.float32),
        }

    def decode_tb_batch(self, inputs):
        self.state += 1
        users = len(inputs[0]["ue_ids"])
        batch = len(inputs)
        return {
            "hard_bits": np.zeros((batch, users, 4), bool),
            "crc_status": np.ones((batch, users), bool),
            "logits": np.zeros((batch, users, 4), np.float32),
        }

    def checksums_unchanged(self):
        return True


class SyntheticScenarioBank:

    def generate(self, job, episode_index):
        tbs = int(job["num_tbs"])
        users = 2
        base = int(job["seed_streams"]["channel"]) + int(episode_index)
        active = np.ones((tbs, users), bool)
        ue_ids = np.tile(np.arange(users, dtype=np.int32), (tbs, 1))
        change_event = np.zeros((tbs, users), bool)
        if job["provider"] == "shared_geometry_change":
            change_event[job["provider_settings"]["change_tb"] - 1] = True
        if job["provider"] == "shared_geometry_schedule":
            active[4, 1] = False
            ue_ids[8:, 0] = 2
        bits = np.zeros((tbs, users, 4), bool)
        receiver = []
        for snr_index, snr in enumerate(job["snr_db"]):
            receiver_tbs = []
            for tb in range(tbs):
                point = float(base + 10 * snr_index + tb)
                inputs = {
                    "active": active[tb],
                    "ls_channel": np.full((users, 2), point, np.float32),
                    "noise_variance": np.asarray(10.0 ** (-snr / 10.0)),
                    "received_grid": np.full((2, 4), point, np.float32),
                    "slot_number": np.asarray(tb % 20, np.int32),
                    "ue_ids": ue_ids[tb],
                }
                receiver_tbs.append(inputs)
            receiver.append(tuple(receiver_tbs))
        return BenchmarkEpisode(
            bits=bits,
            active=active,
            ue_ids=ue_ids,
            channel_change_event=change_event,
            receiver_inputs=tuple(receiver),
        )


def synthetic_candidate_manifest(protocol):
    from temporal_benchmark_frozen import make_manifest
    from temporal_generalization_protocol import seal_payload
    manifest = make_manifest(protocol, "frozen_cold_k2")[0]
    manifest["role"] = "candidate"
    manifest["inference"]["persistent_memory"] = True
    return seal_payload(manifest, field="manifest_sha256")
