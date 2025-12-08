cmake_minimum_required(VERSION 3.24)

project(Digital-Twin-RHI)

file(GLOB_RECURSE SOURCE_FILES "*.cpp" "*.hpp")

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "" FILES ${SOURCE_FILES})

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/.. 
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    Digital-Twin-Core
    volk                
    VulkanMemoryAllocator
    shaderc_combined
    spirv-reflect
    glfw
)