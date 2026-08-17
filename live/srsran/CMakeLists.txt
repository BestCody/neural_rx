cmake_minimum_required(VERSION 3.16)
project(temporal_nrx_srsran_plugin LANGUAGES CXX)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

add_library(temporal_nrx_python_plugin SHARED srsran_temporal_nrx_python_plugin.cpp)
target_compile_features(temporal_nrx_python_plugin PRIVATE cxx_std_14)
target_include_directories(temporal_nrx_python_plugin PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(temporal_nrx_python_plugin PRIVATE Python3::Python)
set_target_properties(temporal_nrx_python_plugin PROPERTIES
  OUTPUT_NAME temporal_nrx_python
  POSITION_INDEPENDENT_CODE ON
)
