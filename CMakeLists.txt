cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

# --- Path Configuration ---
set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(CMAKE_DIR "${CMAKE_SOURCE_DIR}/cmake")
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
set(EXTERNAL_DIR "${CMAKE_SOURCE_DIR}/external")

# --- Output Directory Configuration ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

# --- Compiler Settings ---
include("${CMAKE_DIR}/compilerSettings.cmake" NO_POLICY_SCOPE)

# --- Dependencies ---
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Add external dependencies (imgui, glfw, nanobind, etc.)
add_subdirectory(${EXTERNAL_DIR})

# --- Main Projects ---
# 1. Build the Engine (DLL) + Python Bindings
add_subdirectory(${SRC_DIR})

# 2. Build the Editor Application
add_subdirectory(apps/Gaudi)

# --- Optional Sandbox/Tests ---
if(EXISTS "${ROOT_DIR}/sandbox/CMakeLists.txt")
    add_subdirectory(${ROOT_DIR}/sandbox)
endif()

if(CMAKE_PROJECT_NAME STREQUAL "Digital-Twin")
    if(EXISTS "${ROOT_DIR}/tests/CMakeLists.txt")
        add_subdirectory(tests)
    endif()
endif()

# Enable IDE folders view
set_property(GLOBAL PROPERTY USE_FOLDERS ON)