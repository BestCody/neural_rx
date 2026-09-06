<!-- SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited. -->

# Real-Time Inference of 5G NR Multi-user MIMO Neural Receivers

The code in this repository allows to design, train, and evaluate [neural
receivers](https://developer.nvidia.com/blog/towards-environment-specific-base-stations-ai-ml-driven-neural-5g-nr-multi-user-mimo-receiver/)
using the [NVIDIA® Sionna™ link-level simulation
library](https://nvlabs.github.io/sionna/) and TensorFlow. Further, trained
receivers can be prepared for real-time inference via [NVIDIA®
TensorRT™](https://developer.nvidia.com/tensorrt).

[<p align="center"><img src="misc/nrx_overview.png" height=400></p>](notebooks/jumpstart_tutorial.ipynb)

The following features are currently supported:

- 5G NR compliant Multi-user MIMO PUSCH receiver
- Training pipeline using 3GPP compliant channel models
- TensorRT / ONNX model export for real-time inference
- Support for varying number of PRBs, users, and different MCS schemes per user
- End-to-end learning of custom constellations for [pilotless communications](https://arxiv.org/pdf/2009.05261) [3]
- Site-specific training using ray-tracing based channel simulations from [SionnaRT](https://nvlabs.github.io/sionna/api/rt.html) as done in [2]

We recommend starting with the [Jumpstart NRX Tutorial notebook](notebooks/jumpstart_tutorial.ipynb) for a detailed introduction and overview of the project.

## Temporal receiver

This fork adds persistent per-UE channel memory for a latency-constrained
Neural RX. The retained v18 receiver combines:

- Causal PD-L8 channel memory from current and previous DMRS observations.
- A two-iteration CGNN distilled from the frozen eight-iteration receiver.
- Coded Joint-List Refinement between detection and LDPC decoding.

Memory follows physical UE identity across scheduling order changes. It stores
pilot-derived channel evidence, not previous payload bits or hidden detection
state. Causal DMRS disagreement and UE re-entry reset stale history.

The current design, limitations, and retained results are documented in
[Temporal receiver v18](docs/temporal_receiver_v18.md). Earlier conclusions are
summarized in [Experiment history](docs/experiment_history.md).
The exact streamed PD-L8 optimization is part of the active V18 adapter.
The corrected training contract is specified in
[V18 training correction](docs/temporal_receiver_v18_training_correction.md).
The current weak-user correction is specified in
[V18 training V3](docs/temporal_receiver_v18_training_correction_v3.md).

### Current source

- [PD-L8 memory state](scripts/pilot_domain_memory.py)
- [Pilot-domain estimator](scripts/pilot_domain_channel.py)
- [Active streamed estimator](scripts/pilot_domain_channel_v19.py)
- [Active streamed memory](scripts/pilot_domain_memory_v19.py)
- [Shared receiver helpers](scripts/temporal_receiver_core.py)
- [CJLR detector and decoder](scripts/cjlr_receiver.py)
- [Optimized CJLR kernel](scripts/cjlr_optimized_receiver.py)
- [Frozen v18 runtime](scripts/cjlr_final_runtime.py)
- [Training and scenario data](scripts/temporal_training_data.py)
- [V3 power training data](scripts/temporal_power_training_data.py)
- [V18 configuration](config/temporal_receiver_v18_segment_safe.json)
- [Current V3 training data](config/temporal_benchmark_training_data_v3.json)
- [Current V3 training recipe](config/temporal_benchmark_training_recipe_v3.json)
- [Reproducible V2 data](config/temporal_benchmark_training_data_v2.json)
- [Reproducible V2 recipe](config/temporal_benchmark_training_recipe_v2.json)
- [Event-relative targets](scripts/temporal_training_curriculum.py)
- [Aligned trainer](scripts/orchestrate_v18_aligned_training.py)

### Systematic evaluation

Every new receiver must use
[Temporal benchmark v1](docs/temporal_benchmark_v1.md). The benchmark owns the
data, seeds, paired methods, reset control, metrics, latency timing, and
acceptance rules. Architecture code supplies only a decoder adapter.

The primary accuracy comparison is:

- Candidate K=2 with persistent memory.
- The same checkpoint with memory reset every TB.
- Training-matched Cold K=8.

Frozen Cold K=3 is measured only in the fixed latency microbenchmark.

Screening uses four core conditions, 16 episodes, one SNR, and 16 TBs.
Development uses the same conditions with 64 episodes, three SNRs, and 64
TBs. Confirmation uses three independently trained checkpoints and starts at
64 episodes; it expands to 128 or 256 only if the predeclared confidence target
is not met. Historical Cold K=8 and perfect-CSI K-Best are cached once for final
reporting. Accuracy batches independent episodes while memory remains strictly
sequential within each episode; latency uses a separate batch-one test.

Generated checkpoints and experiment outputs belong under `outputs/`; they
are intentionally excluded from Git.

The V3 screen is dependency ordered. It trains and seals matched Cold K=8
first, then warm-starts and distills K=2 from that exact checkpoint:

```bash
python scripts/orchestrate_v18_aligned_training.py \
  --output-dir outputs/v18_aligned_screen --gpu 0
```

Add `--smoke-only` to run the two five-update Stage 0 checks without starting
the 500-update screen.

The basic neural receiver architecture is introduced and described in [a Neural Receiver for 5G NR Multi-user MIMO](https://arxiv.org/pdf/2312.02601) [1].
The real-time experiments and the site-specific training is described in [Design of a Standard-Compliant Real-Time
Neural Receiver for 5G NR](https://arxiv.org/abs/2409.02912) [2].

Demos of this receiver architecture have been shown at [Mobile World Congress 2023](https://www.youtube.com/watch?v=BQyxBYzdg5k) and [Mobile World Congress 2024](https://www.keysight.com/us/en/assets/3124-1306/demos/6G-AI-Neural-Receiver-Design.mp4).

For further details regarding solutions for deployment in an actual Radio Access Network (RAN), we recommend registering for the [NVIDIA 6G Developer Program](https://developer.nvidia.com/6g-program).

[<p align="center"><img src="misc/nrx_bler_results.png" height=400></p>](notebooks/jumpstart_tutorial.ipynb)

## Summary

We introduce a neural network (NN)-based multi-user multiple-input
multiple-output (MU-MIMO) receiver with full 5G New Radio (5G NR) physical
uplink shared channel (PUSCH) compatibility based on graph and convolutional
neural network (CGNN) components. The proposed architecture can be easily
re-parametrized to an arbitrary number of sub-carriers and supports a varying
number of users without the need for any additional re-training. The receiver
operates on an entire 5G NR slot, i.e., it processes the entire received
orthogonal frequency division multiplexing (OFDM) time-frequency resource grid
by jointly performing channel estimation, equalization, and demapping. We show
the importance of a carefully designed training process such that the trained
receiver does not overfit to a specific channel realization and remains
universal for a wide range of different unseen channel conditions. A particular
focus of the architecture design is put on the real-time inference capability
such that the receiver can be executed within 1 ms latency on an NVIDIA A100
GPU.

[<p align="center"><img src="misc/nrx_latency.png" height=450></p>](notebooks/real_time_nrx.ipynb)

## Setup

Running this code requires [Sionna 0.18](https://nvlabs.github.io/sionna/).
To run the notebooks on your machine, you also need [Jupyter](https://jupyter.org).
We recommend Ubuntu 22.04, Python 3.10, and TensorFlow 2.15.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For [TensorRT](https://developer.nvidia.com/tensorrt), we recommend version 9.6 and newer.
For [ONNX](https://onnx.ai/) exporting, this repository uses `onnx==1.16.2`.

## Structure of this repository

This repository is structured in the following way:
- [config](config/) contains the system configurations for different experiments
- [notebooks](notebooks/) contains tutorials and code examples
- [scripts](scripts/) contains the scripts to train, evaluate and debug the NRX
- [utils](utils/) contains the NRX definition and all Python utilities
- [weights](weights/) contains weights of pre-trained neural receivers for different configuration files
- [results](results/) contains pre-computed BLER performance results

The following folders are generated locally:
- `logs` contains log files of the training
- `onnx_models` contains exported ONNX neural receiver models
- `data` contains a ray tracing-based dataset of channel realizations for site-specific evaluation
- `outputs` contains temporal checkpoints and experiment results

We recommend starting with the [Jumpstart NRX Tutorial notebook](notebooks/jumpstart_tutorial.ipynb) for a detailed introduction and overview of the project.

## References

[1] S. Cammerer, F. Aït Aoudia, J. Hoydis, A. Oeldemann, A. Roessler, T. Mayer, and A. Keller, "[A Neural Receiver for 5G NR Multi-user MIMO](https://arxiv.org/pdf/2312.02601)", IEEE Workshops (GC Wkshps), Dec. 2023.

[2] R. Wiesmayr, S. Cammerer, F. Aït Aoudia, J. Hoydis, J. Zakrzewski, and A. Keller, "[Design of a Standard-Compliant Real-Time Neural Receiver for 5G NR](https://arxiv.org/abs/2409.02912)", arxiv preprint, 2024.

[3] F. Aït Aoudia and J. Hoydis, "[End-to-end Learning for OFDM: From Neural Receivers to Pilotless Communication](https://arxiv.org/pdf/2009.05261)", IEEE Trans on Wireless Commun., 2021.

## License

Copyright &copy; 2024, NVIDIA Corporation. All rights reserved.

This work is made available under the [NVIDIA License](LICENSE.txt).

# Citation

```
@software{neural_rx,
    title = {Real-time 5G NR Multi-user MIMO Receivers},
    author = {Sebastian Cammerer, Reinhard Wiesmayr, Fayçal Aït Aoudia, Jakob Hoydis, Tommi Koivisto, Jakub Zakrzewski, Ruqing Xu, Pawel Morkisz, Chris Dick, and Alexander Keller},
    note = {https://github.com/NVlabs/neural-rx},
    year = 2024
}
```

# Acknowledgement

This work has received financial support from the European Union under
Grant Agreement 101096379 (CENTRIC). Views and opinions expressed are
however those of the author(s) only and do not necessarily reflect those of the
European Union or the European Commission (granting authority). Neither the
European Union nor the granting authority can be held responsible for them.
