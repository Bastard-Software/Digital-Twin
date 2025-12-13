cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-Tests)

enable_testing()

file(GLOB_RECURSE TEST_SOURCES "*.cpp")

add_executable(${PROJECT_NAME} ${TEST_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ROOT_DIR}/src
    ${EXTERNAL_DIR}/googletest/googletest/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    GTest::gtest
    Digital-Twin-Core       
    Digital-Twin-Runtime
    Digital-Twin-RHI
    Digital-Twin-Streaming
    Digital-Twin-Compute
    Digital-Twin-Simulation
    Digital-Twin-Renderer
)

add_test(NAME AllTests COMMAND ${PROJECT_NAME})

set_property(TARGET ${PROJECT_NAME} PROPERTY FOLDER "tests")