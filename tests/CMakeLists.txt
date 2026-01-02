cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-Tests)

# Enable CTest functionality
enable_testing()

# -----------------------------------------------------------------------------
# 1. Source Collection
# -----------------------------------------------------------------------------
file(GLOB_RECURSE TEST_SOURCES "*.cpp")

# -----------------------------------------------------------------------------
# 2. Test Executable
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${TEST_SOURCES})

# -----------------------------------------------------------------------------
# 3. Includes & Linking
# -----------------------------------------------------------------------------

# Allow tests to see "src" (to include private headers like RHI/Vulkan.h)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}      # Access to internal headers
    ${INCLUDE_DIR}  # Access to public headers
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    # Link Google Test Main (handles main() function automatically)
    GTest::gtest_main
    
    # Link the INTERNAL OBJECTS of the engine.
    # This gives access to private symbols not exported in the DLL.
    Digital-Twin-Obj
)

# -----------------------------------------------------------------------------
# 4. Test Registration
# -----------------------------------------------------------------------------
# Register the test with CTest
add_test(NAME AllTests COMMAND ${PROJECT_NAME})

# Ensure the test exe ends up in /bin (next to assets/shaders if needed)
set_target_properties(${PROJECT_NAME} PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# IDE Organization
set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "tests")