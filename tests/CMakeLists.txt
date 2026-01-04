add_executable(axon_tests
  neuron_test.cpp
  network_test.cpp
  activation_test.cpp
  criterion_test.cpp
)

target_link_libraries(axon_tests PRIVATE
  axon::core
  GTest::gtest_main
)

set_target_properties(axon_tests PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

include(GoogleTest)
gtest_discover_tests(axon_tests)
