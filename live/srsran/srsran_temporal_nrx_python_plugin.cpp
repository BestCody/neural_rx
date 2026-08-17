#include "temporal_nrx_plugin_abi.h"

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace {

std::mutex     python_runtime_mutex;
std::once_flag python_module_once;
std::string    init_error;
PyObject*      infer_callable = nullptr;

void set_error(temporal_nrx_response_v1* response, const std::string& message)
{
  if (response == nullptr || response->error_message == nullptr || response->error_message_capacity == 0) {
    return;
  }
  std::snprintf(response->error_message,
                static_cast<size_t>(response->error_message_capacity),
                "%s",
                message.c_str());
}

std::string python_error_string()
{
  if (!PyErr_Occurred()) {
    return "unknown Python error";
  }
  PyObject *type = nullptr, *value = nullptr, *traceback = nullptr;
  PyErr_Fetch(&type, &value, &traceback);
  PyErr_NormalizeException(&type, &value, &traceback);
  PyObject* text = value != nullptr ? PyObject_Str(value) : nullptr;
  std::string out = "Python exception";
  if (text != nullptr) {
    const char* utf8 = PyUnicode_AsUTF8(text);
    if (utf8 != nullptr) {
      out = utf8;
    }
  }
  Py_XDECREF(text);
  Py_XDECREF(type);
  Py_XDECREF(value);
  Py_XDECREF(traceback);
  PyErr_Clear();
  return out;
}

// Ensure CPython exists, but never execute Python/NumPy C-API operations here.
// When this plugin is the component that initializes Python (the normal gNB
// case), release the bootstrap GIL immediately so arbitrary PUSCH worker
// threads can subsequently enter through PyGILState_Ensure().
void ensure_python_runtime()
{
  if (Py_IsInitialized()) {
    return;
  }
  std::lock_guard<std::mutex> lock(python_runtime_mutex);
  if (!Py_IsInitialized()) {
    Py_Initialize();
    if (Py_IsInitialized()) {
      PyEval_SaveThread();
    }
  }
}

// Called exactly once while the caller holds the GIL.
void initialize_python_module()
{
  if (_import_array() < 0) {
    init_error = "NumPy C API initialization failed: " + python_error_string();
    return;
  }

  const char* module_name = std::getenv("TEMPORAL_NRX_PY_MODULE");
  if (module_name == nullptr || *module_name == '\0') {
    module_name = "srsran_embedded_entry";
  }
  PyObject* module = PyImport_ImportModule(module_name);
  if (module == nullptr) {
    init_error = std::string("cannot import ") + module_name + ": " + python_error_string();
    return;
  }
  infer_callable = PyObject_GetAttrString(module, "infer_pusch");
  Py_DECREF(module);
  if (infer_callable == nullptr || !PyCallable_Check(infer_callable)) {
    Py_XDECREF(infer_callable);
    infer_callable = nullptr;
    init_error = std::string(module_name) + ".infer_pusch is not callable";
  }
}

PyObject* complex_array_view(const temporal_nrx_complex32* data,
                             npy_intp d0,
                             npy_intp d1,
                             npy_intp d2)
{
  static_assert(sizeof(temporal_nrx_complex32) == 2 * sizeof(float), "complex ABI must be two float32 values");
  npy_intp dims[3] = {d0, d1, d2};
  return PyArray_SimpleNewFromData(3, dims, NPY_COMPLEX64, const_cast<temporal_nrx_complex32*>(data));
}

float llr_range_limit_from_env()
{
  const char* text = std::getenv("TEMPORAL_NRX_LLR_RANGE");
  if (text == nullptr || *text == '\0') {
    return 20.0F;
  }
  char* end = nullptr;
  errno = 0;
  float value = std::strtof(text, &end);
  if (errno != 0 || end == text || !std::isfinite(value) || value <= 0.0F) {
    return 20.0F;
  }
  return value;
}

} // namespace

extern "C" int temporal_nrx_infer_v1(const temporal_nrx_request_v1* request,
                                      temporal_nrx_response_v1* response)
{
  if (request == nullptr || response == nullptr) {
    return 1;
  }
  response->llr_grid_written = 0;
  response->llr_range_limit = llr_range_limit_from_env();
  if (request->abi_version != TEMPORAL_NRX_PLUGIN_ABI_VERSION ||
      response->abi_version != TEMPORAL_NRX_PLUGIN_ABI_VERSION) {
    set_error(response, "temporal NRX plugin ABI version mismatch");
    return 2;
  }
  if (request->rx_grid == nullptr || request->channel_estimate == nullptr || response->llr_grid == nullptr) {
    set_error(response, "null tensor pointer in temporal NRX request/response");
    return 3;
  }
  if (request->nof_rx_ports == 0 || request->nof_symbols == 0 || request->nof_subcarriers == 0 ||
      request->nof_bits_per_symbol == 0) {
    set_error(response, "invalid zero-sized PUSCH tensor geometry");
    return 4;
  }
  const uint64_t expected = static_cast<uint64_t>(request->nof_symbols) * request->nof_subcarriers *
                            request->nof_bits_per_symbol;
  if (response->llr_grid_capacity < expected) {
    set_error(response, "LLR response buffer is too small");
    return 5;
  }

  ensure_python_runtime();
  if (!Py_IsInitialized()) {
    set_error(response, "failed to initialize embedded Python runtime");
    return 6;
  }

  PyGILState_STATE gil = PyGILState_Ensure();
  std::call_once(python_module_once, initialize_python_module);
  if (!init_error.empty() || infer_callable == nullptr) {
    set_error(response, init_error.empty() ? "embedded Python runtime unavailable" : init_error);
    PyGILState_Release(gil);
    return 6;
  }

  int rc = 0;
  PyObject* rx = nullptr;
  PyObject* h = nullptr;
  PyObject* args = nullptr;
  PyObject* result = nullptr;
  PyArrayObject* llr = nullptr;

  rx = complex_array_view(request->rx_grid,
                          request->nof_rx_ports,
                          request->nof_symbols,
                          request->nof_subcarriers);
  h = complex_array_view(request->channel_estimate,
                         request->nof_rx_ports,
                         request->nof_symbols,
                         request->nof_subcarriers);
  if (rx == nullptr || h == nullptr) {
    set_error(response, "failed to expose srsRAN tensors to NumPy: " + python_error_string());
    rc = 7;
    goto done;
  }

  args = Py_BuildValue("(OOHKKI)",
                       rx,
                       h,
                       static_cast<unsigned short>(request->crnti),
                       static_cast<unsigned long long>(request->slot_index),
                       static_cast<unsigned long long>(request->dmrs_symbol_mask),
                       static_cast<unsigned int>(request->nof_bits_per_symbol));
  if (args == nullptr) {
    set_error(response, "failed to create Python inference arguments: " + python_error_string());
    rc = 8;
    goto done;
  }

  result = PyObject_CallObject(infer_callable, args);
  if (result == nullptr) {
    set_error(response, "temporal NRX Python inference failed: " + python_error_string());
    rc = 9;
    goto done;
  }
  llr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(result, NPY_FLOAT32, NPY_ARRAY_CARRAY_RO));
  if (llr == nullptr) {
    set_error(response, "temporal NRX result is not a contiguous float32 array: " + python_error_string());
    rc = 10;
    goto done;
  }
  if (PyArray_NDIM(llr) != 3 ||
      static_cast<uint32_t>(PyArray_DIM(llr, 0)) != request->nof_symbols ||
      static_cast<uint32_t>(PyArray_DIM(llr, 1)) != request->nof_subcarriers ||
      static_cast<uint32_t>(PyArray_DIM(llr, 2)) != request->nof_bits_per_symbol) {
    set_error(response, "temporal NRX result must have shape [symbols,subcarriers,bits_per_symbol]");
    rc = 11;
    goto done;
  }

  std::memcpy(response->llr_grid, PyArray_DATA(llr), expected * sizeof(float));
  response->llr_grid_written = expected;

done:
  Py_XDECREF(llr);
  Py_XDECREF(result);
  Py_XDECREF(args);
  Py_XDECREF(h);
  Py_XDECREF(rx);
  PyGILState_Release(gil);
  return rc;
}
