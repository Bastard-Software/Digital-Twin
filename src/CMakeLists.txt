cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-Engine)

# -----------------------------------------------------------------------------
# 1. File Globbing
# -----------------------------------------------------------------------------

# Scan for implementation files within SRC_DIR
file(GLOB_RECURSE ENGINE_SOURCES 
    "${SRC_DIR}/*.cpp" 
    "${SRC_DIR}/*.hpp"
    "${SRC_DIR}/*.h"
)

# Scan for public header files within INCLUDE_DIR
file(GLOB_RECURSE ENGINE_HEADERS 
    "${INCLUDE_DIR}/*.hpp" 
    "${INCLUDE_DIR}/*.h"
)

# -----------------------------------------------------------------------------
# 2. Visual Studio Folder Structure
# -----------------------------------------------------------------------------

# Create virtual folders in the IDE to match the filesystem
source_group(TREE "${SRC_DIR}" PREFIX "Source Files" FILES ${ENGINE_SOURCES})
source_group(TREE "${INCLUDE_DIR}" PREFIX "Header Files" FILES ${ENGINE_HEADERS})

# -----------------------------------------------------------------------------
# 3. OBJECT LIBRARY (Compilation Unit)
# -----------------------------------------------------------------------------
# We build an OBJECT library first. This compiles the source files but does not 
# link them into a final DLL yet. This allows us to reuse these compiled 
# objects in both the main Engine DLL and Unit Tests.
add_library(Digital-Twin-Obj OBJECT 
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

# Required for object libraries that will eventually be linked into a shared library
set_property(TARGET Digital-Twin-Obj PROPERTY POSITION_INDEPENDENT_CODE ON)

# --- Include Directories ---
target_include_directories(Digital-Twin-Obj PUBLIC "${INCLUDE_DIR}")
target_include_directories(Digital-Twin-Obj PUBLIC "${SRC_DIR}")

# --- Compile Definitions ---
# DT_BUILD_DLL: Used in C++ to handle __declspec(dllexport)
target_compile_definitions(Digital-Twin-Obj PRIVATE DT_BUILD_DLL)
# VK_NO_PROTOTYPES: Required by volk
target_compile_definitions(Digital-Twin-Obj PUBLIC VK_NO_PROTOTYPES)

# --- Dependency Linking (For Compilation) ---

# PUBLIC DEPENDENCIES
# Libraries that are part of our public API (e.g., used in public headers).
# We link them here so the Object Library can find their headers.
target_link_libraries(Digital-Twin-Obj PUBLIC 
    imgui
    glm
    volk
    VulkanMemoryAllocator
    entt
    spdlog::spdlog
    shaderc_combined
    spirv-reflect
    glfw
)

# Move this target to a folder named "Internal" so it doesn't clutter the root.
set_property(TARGET Digital-Twin-Obj PROPERTY FOLDER "Internal")

# -----------------------------------------------------------------------------
# 4. SHARED LIBRARY (The Final DLL)
# -----------------------------------------------------------------------------
add_library(Digital-Twin SHARED
    ${ENGINE_SOURCES}
    ${ENGINE_HEADERS}
)

target_include_directories(Digital-Twin PUBLIC "${INCLUDE_DIR}")
target_include_directories(Digital-Twin PRIVATE "${SRC_DIR}")

target_compile_definitions(Digital-Twin PRIVATE DT_BUILD_DLL)
# VK_NO_PROTOTYPES: Required by volk
target_compile_definitions(Digital-Twin PUBLIC VK_NO_PROTOTYPES)

# --- Dependency Linking (For Linker) ---
# We must re-declare dependencies here so the linker includes their binary code.

# A. PUBLIC (Visible to Consumers like Editor)
# Consumers will automatically inherit include paths for these.
target_link_libraries(Digital-Twin PUBLIC 
    imgui
    glm
)

# B. PRIVATE (Hidden from Consumers)
# These libraries are baked into the DLL, but their headers/symbols 
# are not exposed to the Editor or Sandbox.
target_link_libraries(Digital-Twin PRIVATE
    volk
    VulkanMemoryAllocator
    entt
    spdlog::spdlog
    shaderc_combined
    spirv-reflect
    glfw
)

# -----------------------------------------------------------------------------
# 5. Python Bindings (nanobind)
# -----------------------------------------------------------------------------
# Creates the .pyd/.so module
nanobind_add_module(Digital-Twin-Python "${SRC_DIR}/python/Bindings.cpp")

# Link against the public Engine DLL interface
target_link_libraries(Digital-Twin-Python PRIVATE Digital-Twin)

# Set the import name for Python (e.g., 'import DigitalTwin')
set_target_properties(Digital-Twin-Python PROPERTIES OUTPUT_NAME "DigitalTwin")