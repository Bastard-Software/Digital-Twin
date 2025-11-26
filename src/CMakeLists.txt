cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

include("${CMAKE_DIR}/compilerSettings.cmake" NO_POLICY_SCOPE)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# targets
add_subdirectory(Core)

file(GLOB_RECURSE SOURCE_FILES
    "${SRC_DIR}/*.h"
    "${SRC_DIR}/*.hpp"
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.inl"
)
list(FILTER SOURCE_FILES EXCLUDE REGEX "/core/")
list(FILTER SOURCE_FILES EXCLUDE REGEX "/python/bindings.cpp$")

source_group(TREE "${SRC_DIR}" PREFIX "src" FILES ${SOURCE_FILES})

# build lib
add_library(${PROJECT_NAME} STATIC
    ${SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${SRC_DIR}/
  ${EXTERNAL_DIR}/volk/
  ${EXTERNAL_DIR}/glm/
  ${EXTERNAL_DIR}/entt/src/
  ${EXTERNAL_DIR}/spdlog/include/
) 

target_link_libraries(${PROJECT_NAME} PUBLIC
    volk
    entt
    Digital-Twin-Core
)

# build bindings
nanobind_add_module(${PROJECT_NAME}-Python ${SRC_DIR}/python/bindings.cpp)

target_include_directories(${PROJECT_NAME}-Python PRIVATE
  ${SRC_DIR}/
)

target_link_libraries(${PROJECT_NAME}-Python PRIVATE
    ${PROJECT_NAME}
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