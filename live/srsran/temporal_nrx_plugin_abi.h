#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

#define TEMPORAL_NRX_PLUGIN_ABI_VERSION 1u

typedef struct temporal_nrx_complex32 {
  float re;
  float im;
} temporal_nrx_complex32;

typedef struct temporal_nrx_request_v1 {
  uint32_t abi_version;
  uint16_t crnti;
  uint16_t reserved0;
  uint64_t slot_index;
  uint32_t nof_rx_ports;
  uint32_t nof_symbols;
  uint32_t nof_subcarriers;
  uint32_t nof_bits_per_symbol;
  uint32_t dmrs_symbol_mask;
  uint32_t reserved1;
  const temporal_nrx_complex32* rx_grid;
  const temporal_nrx_complex32* channel_estimate;
} temporal_nrx_request_v1;

typedef struct temporal_nrx_response_v1 {
  uint32_t abi_version;
  uint32_t reserved0;
  float* llr_grid;
  uint64_t llr_grid_capacity;
  uint64_t llr_grid_written;
  float llr_range_limit;
  char* error_message;
  uint32_t error_message_capacity;
} temporal_nrx_response_v1;

int temporal_nrx_infer_v1(const temporal_nrx_request_v1* request,
                          temporal_nrx_response_v1* response);

#ifdef __cplusplus
}
#endif
