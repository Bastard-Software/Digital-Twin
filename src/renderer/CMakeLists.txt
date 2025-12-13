cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-Renderer)

file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.hpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "" FILES ${SOURCE_FILES})

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/.. 
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    Digital-Twin-Core
    Digital-Twin-RHI
    Digital-Twin-Platform
    Digital-Twin-Simulation
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets
    ${CMAKE_BINARY_DIR}/assets
    COMMENT "Auto-copying assets to build directory..."
)