add_executable(ann_demo main.cpp)

target_link_libraries(ann_demo PRIVATE ann_core)

set_target_properties(ann_demo PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)
