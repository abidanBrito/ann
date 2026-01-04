add_executable(axon_demo main.cpp)

target_link_libraries(axon_demo PRIVATE axon_core)

set_target_properties(axon_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
