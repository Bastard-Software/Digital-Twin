cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(CMAKE_DIR "${CMAKE_SOURCE_DIR}/cmake")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

include("${CMAKE_DIR}/compilerSettings.cmake" NO_POLICY_SCOPE)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# add dependencies
add_subdirectory(${EXTERNAL_DIR})

# add main project cmake
add_subdirectory(${SRC_DIR})

if(CMAKE_PROJECT_NAME STREQUAL "Digital-Twin")
    add_subdirectory(tests)
endif()

# add sandbox
add_subdirectory(${ROOT_DIR}/sandbox)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)