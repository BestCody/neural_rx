from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from artifact_io import atomic_json, file_sha256
from rt_motion import Motion
from temporal_generalization_protocol import seal_payload


UPSTREAM = {
    "repository": "https://github.com/zhouyou-gu/ocudu-gpu-channel",
    "commit": "f95dabd0d13db3b24e9a4b04c919ad3620468f11",
    "motion_source": "apps/sionna_bridge/run_bridge.py:Motion",
    "license": "licenses/ocudu-gpu-channel.txt",
    "adaptation": "Motion class without docstrings; scene and routes are supplied by the configuration",
}


def motion(node):
    return Motion(start=tuple(node["start_m"]), velocity=tuple(node.get("velocity_mps", (0., 0., 0.))),
                  x_bounds=tuple(node["route_x_m"]) if "route_x_m" in node else None,
                  waypoints=tuple(map(tuple, node.get("route_m", ()))),
                  route_mode=node.get("route_mode", "pingpong"), speed_mps=float(node.get("speed_mps", 0.)))


def trajectory_start_times(node, count, duration):
    points = np.asarray(node["route_m"], np.float64)
    speed = float(node["speed_mps"])
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("Each UE requires at least two finite three-dimensional route points")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.isfinite(speed) or speed <= 0 or np.any(lengths <= 0):
        raise ValueError("Route segments and UE speed must be positive")
    turn_times = np.cumsum(lengths) / speed
    times = np.arange(count) * (turn_times[-1] / count)
    next_turn = turn_times[np.searchsorted(turn_times, times, side="left")]
    indices = np.flatnonzero(times + duration < next_turn)
    if not len(indices):
        raise ValueError("No trajectory sample supports the declared local-motion interval")
    return indices, times[indices]


def path_arrays(paths, config):
    a, tau = paths.cir(num_time_steps=1, normalize_delays=config["normalize_delays"],
                       reverse_direction=True, out_type="numpy")
    a = np.asarray(a[..., 0], np.complex64)
    tau = np.asarray(tau, np.float32)
    doppler = np.asarray(paths.doppler.numpy(), np.float32)
    if tau.ndim == 3:
        doppler = doppler.transpose(1, 0, 2)
        tau = tau[:, None, :, None, :]
        doppler = doppler[:, None, :, None, :]
    else:
        doppler = doppler.transpose(2, 3, 0, 1, 4)
    if not all(np.isfinite(value).all() for value in (a, tau, doppler)):
        raise ValueError("Ray tracer produced non-finite paths")
    if not np.all(np.sum(np.abs(a)**2, axis=(0, 1, 3, 4)) > 0):
        return None
    tau, doppler = (np.broadcast_to(value, a.shape) for value in (tau, doppler))
    count = config["num_paths"]
    if a.shape[-1] > count:
        indices = np.argsort(-np.abs(a), axis=-1, kind="stable")[..., :count]
        a, tau, doppler = (np.take_along_axis(value, indices, axis=-1)
                           for value in (a, tau, doppler))
    elif a.shape[-1] < count:
        padding = [(0, 0)] * (a.ndim - 1) + [(0, count - a.shape[-1])]
        a, tau, doppler = (np.pad(value, padding, constant_values=fill)
                           for value, fill in ((a, 0), (tau, -1), (doppler, 0)))
    return a, tau, doppler


def export(config, output):
    import sionna.rt as rt

    if importlib.metadata.version("sionna-rt") != "2.0.1":
        raise ValueError("Use the separate Sionna RT 2.0.1 environment")
    if config["split"] not in ("calibration", "development", "confirmation"):
        raise ValueError("Declare a calibration, development, or confirmation channel split")
    if config["episodes"] < 1 or config["max_duration_s"] <= 0:
        raise ValueError("Positive episode count and local-motion duration are required")
    for key in ("episodes", "trajectory_samples", "max_attempts_per_episode", "num_paths"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(config["normalize_delays"], bool):
        raise ValueError("normalize_delays must be a boolean")
    if config["power_normalization"] not in ("per_ue_episode", "none"):
        raise ValueError("Unknown channel power normalization")
    nodes = [config["base_station"], *config["users"]]
    if not config["users"] or len({n["name"] for n in nodes}) != len(nodes):
        raise ValueError("Provide users and unique node names")
    for node in nodes:
        if "route_x_m" in node and not node["route_x_m"][0] < node["route_x_m"][1]:
            raise ValueError("Route bounds must increase")
        if node.get("route_mode", "pingpong") not in ("pingpong", "loop"):
            raise ValueError("Unsupported route mode")
    duration = float(config["max_duration_s"])
    starts = [trajectory_start_times(node, config["trajectory_samples"], duration)
              for node in config["users"]]
    scene_path = Path(config["scene"])
    if scene_path.suffix != ".xml":
        scene_path = Path(getattr(rt.scene, config["scene"]))
    scene_path = scene_path.resolve()
    scene = rt.load_scene(str(scene_path), merge_shapes=True)
    scene.frequency = config["carrier_frequency_hz"]
    scene.rx_array = rt.PlanarArray(**config["tx_array"])
    scene.tx_array = rt.PlanarArray(**config["rx_array"])
    for index, node in enumerate(nodes):
        kind = rt.Transmitter if index == 0 else rt.Receiver
        scene.add(kind(name=node["name"], position=node["start_m"],
                       orientation=node.get("orientation_rad", [0., 0., 0.])))
    motions = [motion(node) for node in nodes]
    solver = rt.PathSolver()
    rng = np.random.RandomState(config["seed"])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    rejected = 0
    for index in range(config["episodes"]):
        for attempt in range(config["max_attempts_per_episode"]):
            selected = [int(rng.randint(len(indices))) for indices, _ in starts]
            trajectory_indices = [int(indices[pick]) for (indices, _), pick in zip(starts, selected)]
            start_times = [0.] + [float(times[pick]) for (_, times), pick in zip(starts, selected)]
            positions, velocities = [], []
            for node, route, start in zip(nodes, motions, start_times):
                position = np.asarray(route.position_at(start))
                velocity = np.asarray(route.velocity_at(start))
                for delta in (duration / 2, duration):
                    if not np.allclose(route.position_at(start + delta), position + delta * velocity,
                                       rtol=0., atol=1e-6):
                        raise ValueError("Episode crosses a route turn")
                scene.get(node["name"]).position = position.tolist()
                scene.get(node["name"]).velocity = velocity.tolist()
                positions.append(position.tolist())
                velocities.append(velocity.tolist())
            seed = int(rng.randint(0, 2**31 - 1))
            paths = solver(scene, seed=seed, **config["solver"])
            arrays = path_arrays(paths, config)
            if arrays is not None:
                break
            rejected += 1
            print(json.dumps({"rejected_episode": index, "attempt": attempt + 1,
                              "trajectory_indices": trajectory_indices, "seed": seed,
                              "reason": "no_received_path"}), flush=True)
        else:
            raise RuntimeError("Could not sample a connected episode within max_attempts_per_episode")
        a, tau, doppler = arrays
        name = f"episode_{index:05d}.npz"
        np.savez_compressed(output / name, a=a, tau=tau, doppler_hz=doppler)
        records.append(dict(file=name, sha256=file_sha256(output/name), seed=seed,
                            trajectory_indices=trajectory_indices, start_s_per_node=start_times,
                            positions_m=positions, velocities_mps=velocities, attempts=attempt + 1,
                            coefficient_shape=list(a.shape)))
        print(json.dumps({"exported_episode": index + 1, "path_slots": a.shape[-1]}), flush=True)
    assets = {str(p.relative_to(scene_path.parent)): file_sha256(p)
              for p in sorted(scene_path.parent.rglob("*")) if p.is_file()}
    manifest = seal_payload({
        "schema": "sionna_rt_paths_v1", "config": config, "upstream": UPSTREAM,
        "time_model": "fixed_geometry_path_doppler_at_ofdm_symbol_times",
        "delay_reference": ("first_arrival_per_link_fixed_for_episode" if config["normalize_delays"]
                            else "absolute_propagation_delays_no_per_tb_phase_reset"),
        "tracing_direction": "base_station_to_ues_reversed_to_uplink",
        "path_selection": "strongest_per_antenna_pair_with_matching_delay_and_doppler",
        "sampling": {"positions": "independent_uniform_distance_indices_per_ue",
                     "eligible_positions_per_ue": [len(indices) for indices, _ in starts],
                     "rejected_no_path_episodes": rejected},
        "scene_files": assets, "episodes": records,
        "packages": {name: importlib.metadata.version(name) for name in ("sionna-rt", "mitsuba", "drjit", "numpy")},
        "sources": {name: file_sha256(Path(__file__).parent/name)
                    for name in ("export_rt_channels.py", "rt_motion.py")},
    }, field="bank_sha256")
    atomic_json(output/"manifest.json", manifest)
    print(json.dumps({"channel_bank": str(output), "bank_sha256": manifest["bank_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Export offline ray-traced paths for the shared temporal benchmark.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--split", choices=("calibration", "development", "confirmation"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    config = json.loads(args.config.read_text())
    for key in ("split", "seed", "episodes"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    export(config, args.output_dir.resolve())


if __name__ == "__main__":
    main()
