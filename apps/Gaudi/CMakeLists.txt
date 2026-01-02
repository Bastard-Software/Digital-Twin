cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-Gaudi)

# -----------------------------------------------------------------------------
# 1. File Globbing & Grouping
# -----------------------------------------------------------------------------

file(GLOB_RECURSE EDITOR_SOURCES "*.cpp" "*.hpp" "*.h")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "Editor Files" FILES ${EDITOR_SOURCES})

# -----------------------------------------------------------------------------
# 2. Executable Target
# -----------------------------------------------------------------------------

add_executable(Digital-Twin-Gaudi ${EDITOR_SOURCES})

# -----------------------------------------------------------------------------
# 3. Includes & Linking
# -----------------------------------------------------------------------------

# Explicitly add the include directory.
target_include_directories(Digital-Twin-Gaudi PRIVATE "${INCLUDE_DIR}")

# Link against the Engine DLL
target_link_libraries(Digital-Twin-Gaudi PRIVATE Digital-Twin)

# Note: The executable will be generated in build/bin, sitting next to Digital-Twin.dll