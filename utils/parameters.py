# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import configparser
import tensorflow as tf
from pathlib import Path
from sionna.nr import PUSCHConfig, PUSCHDMRSConfig, TBConfig, CarrierConfig, PUSCHTransmitter, PUSCHPilotPattern
from sionna.channel.tr38901 import PanelArray, UMi
from sionna.mimo import StreamManagement
from sionna.channel import OFDMChannel
from .channel_models import DoubleTDLChannel

class Parameters:
    def __init__(self, config_name, training=False,
                 verbose=False, num_tx_eval=None,
                 n_size_bwp_override=None):

        assert isinstance(verbose, bool), "verbose must be bool."
        assert isinstance(training, bool), "training must be bool."
        assert isinstance(config_name, str), "config_name must be str."

        fn = Path(__file__).resolve().parent.parent / "config" / Path(config_name).name
        if fn.exists():
            config = configparser.RawConfigParser()
            config.read(str(fn))
        else:
            raise FileNotFoundError("Unknown config file.")

        self.config_str = ""
        for section in config.sections():
            s = f"\n---- {section} ----- "
            self.config_str += s + "<br />"
            if verbose:
                print(s)
            for option in config.options(section):
                setattr(self, f"{option}", eval(config.get(section,option)))
                s = f"{option}: {eval(config.get(section,option))}"
                self.config_str += s + "<br />"
                if verbose:
                    print(s)

        if not training:
            self.channel_type = self.channel_type_eval
            self.n_size_bwp = self.n_size_bwp_eval
            self.max_ut_velocity = self.max_ut_velocity_eval
            self.min_ut_velocity = self.min_ut_velocity_eval
            self.channel_norm = self.channel_norm_eval

        if n_size_bwp_override is not None:
            if int(n_size_bwp_override) <= 0:
                raise ValueError("n_size_bwp_override must be positive")
            self.n_size_bwp = int(n_size_bwp_override)

        carrier_config = CarrierConfig(
                n_cell_id=self.n_cell_id,
                cyclic_prefix=self.cyclic_prefix,
                subcarrier_spacing=int(self.subcarrier_spacing/1e3),
                n_size_grid=self.n_size_bwp,
                n_start_grid=self.n_start_grid,
                slot_number=self.slot_number,
                frame_number=self.frame_number)

        pusch_dmrs_config=PUSCHDMRSConfig(
                config_type=self.dmrs_config_type,
                type_a_position=self.dmrs_type_a_position,
                additional_position=self.dmrs_additional_position,
                length=self.dmrs_length,
                dmrs_port_set=self.dmrs_port_sets[0],
                n_scid=self.n_scid,
            num_cdm_groups_without_data=self.num_cdm_groups_without_data)

        mcs_list = self.mcs_index
        self.pusch_configs = []
        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs.append([])
            mcs_index = mcs_list[mcs_list_idx]
            tb_config = TBConfig(
                    mcs_index=mcs_index,
                    mcs_table=self.mcs_table,
                    channel_type="PUSCH")

            pc = PUSCHConfig(
                    carrier_config=carrier_config,
                    pusch_dmrs_config=pusch_dmrs_config,
                    tb_config=tb_config,
                    num_antenna_ports = self.num_antenna_ports,
                    precoding = self.precoding,
                    symbol_allocation = self.symbol_allocation,
                    tpmi = self.tpmi,
                    mapping_type=self.dmrs_mapping_type,)

            for idx,_ in enumerate(self.dmrs_port_sets):
                p = pc.clone()
                p.dmrs.dmrs_port_set = self.dmrs_port_sets[idx]
                p.n_id = self.n_ids[idx]
                p.dmrs.n_id = self.dmrs_nid[idx]
                p.n_rnti = self.n_rntis[idx]
                self.pusch_configs[mcs_list_idx].append(p)


        if not training:
            if num_tx_eval is None:
                num_tx_eval = len(self.dmrs_port_sets)
            self.max_num_tx = num_tx_eval
            self.min_num_tx = num_tx_eval

        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs[mcs_list_idx] = self.pusch_configs[mcs_list_idx][:self.max_num_tx]
        print(f"Evaluating the first {self.max_num_tx} port sets.")


        self.pilots = []
        for slot_num in range(carrier_config.num_slots_per_frame):
            for pcs in self.pusch_configs:
                for pc in pcs:
                    pc.carrier.slot_number = slot_num
            pilot_pattern = PUSCHPilotPattern(self.pusch_configs[0])
            self.pilots.append(pilot_pattern.pilots)
        self.pilots = tf.stack(self.pilots, axis=0)
        for pcs in self.pusch_configs:
            for pc in pcs:
                pc.carrier.slot_number = self.slot_number

        self.transmitters = []
        for mcs_list_idx in range(len(mcs_list)):
            self.transmitters.append(
                PUSCHTransmitter(
                            self.pusch_configs[mcs_list_idx],
                            return_bits=False,
                            output_domain="freq",
                            verbose=self.verbose))

            self.transmitters[mcs_list_idx]._mapper.constellation.center = True

        self.sm = StreamManagement(np.ones([1, self.max_num_tx], int), 1)

        if self.channel_type == "UMi":
            bs_array = PanelArray(num_rows_per_panel = 1,
                                  num_cols_per_panel = self.num_rx_antennas//2,
                                  polarization = "dual",
                                  polarization_type  = 'cross',
                                  antenna_pattern = '38.901',
                                  carrier_frequency = self.carrier_frequency)

            ut_array = PanelArray(num_rows_per_panel = 1,
                                  num_cols_per_panel = pc.num_antenna_ports,
                                  polarization = 'single',
                                  polarization_type = 'V',
                                  antenna_pattern = 'omni',
                                  carrier_frequency = self.carrier_frequency)

            self.channel_model = UMi(
                            carrier_frequency=self.carrier_frequency,
                            o2i_model = 'low',
                            bs_array = bs_array,
                            ut_array = ut_array,
                            direction = 'uplink',
                            enable_pathloss = False,
                            enable_shadow_fading = False)

            self.channel = OFDMChannel(
                    channel_model=self.channel_model,
                    resource_grid=self.transmitters[0]._resource_grid,
                    add_awgn=True,
                    normalize_channel=self.channel_norm,
                    return_channel=True)

        elif self.channel_type == "DoubleTDLlow":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,
                                    num_tx_ant=pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        else:
            raise ValueError("Unknown Channel type.")
