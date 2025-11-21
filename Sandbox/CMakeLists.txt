cmake_minimum_required(VERSION 3.24)

project(Sandbox)
file(GLOB_RECURSE FILES *.hpp *.cpp)

include("${CMAKE_DIR}/compilerSettings.cmake" NO_POLICY_SCOPE)

add_executable(${PROJECT_NAME} ${FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${SRC_DIR}/
) 

target_link_libraries(${PROJECT_NAME} PRIVATE
    Digital-Twin
)