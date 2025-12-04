cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

include("${CMAKE_DIR}/compilerSettings.cmake" NO_POLICY_SCOPE)

# targets
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(simulation)
add_subdirectory(rhi)

# build lib
add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME} INTERFACE
    Digital-Twin-Core
    Digital-Twin-Simulation
    Digital-Twin-Runtime
    Digital-Twin-RHI
)

# build bindings
nanobind_add_module(${PROJECT_NAME}-Python ${SRC_DIR}/python/bindings.cpp)

target_include_directories(${PROJECT_NAME}-Python PRIVATE
  ${SRC_DIR}/
)

target_link_libraries(${PROJECT_NAME}-Python PRIVATE
    Digital-Twin-Simulation
    Digital-Twin-Runtime
    Digital-Twin-Core
    Digital-Twin-RHI
    nanobind-static
)

set_target_properties(${PROJECT_NAME}-Python PROPERTIES
    OUTPUT_NAME "DigitalTwin"
)

add_custom_command(TARGET ${PROJECT_NAME}-Python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${PROJECT_NAME}-Python>
        ${ROOT_DIR}/sandbox-python/DigitalTwin${PYTHON_MODULE_EXTENSION}
    COMMENT "Copying python module to sandbox-python"
)