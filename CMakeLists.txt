cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

set(CMAKE_DIR "${CMAKE_SOURCE_DIR}/cmake")
set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(TWIN_DIR "${CMAKE_SOURCE_DIR}/Digital-Twin")

include("${CMAKE_DIR}/CompilerSettings.cmake" NO_POLICY_SCOPE)

add_subdirectory(${ROOT_DIR}/Digital-Twin)
add_subdirectory(${ROOT_DIR}/Sandbox)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)