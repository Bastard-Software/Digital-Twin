cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-Python)

include("${CMAKE_DIR}/CompilerSettings.cmake" NO_POLICY_SCOPE)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

pybind11_add_module(${PROJECT_NAME}
    Bindings.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE Digital-Twin)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${TWIN_DIR}/
) 

set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "digital_twin"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${CMAKE_BINARY_DIR}/python_packages
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${PROJECT_NAME}>
        ${ROOT_DIR}/Sandbox-Python/digital_twin${PYTHON_MODULE_EXTENSION}
    COMMENT "Copying Python module to Sandbox-Python"
)

message(STATUS "Python bindings configured")
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python version: ${Python_VERSION}")