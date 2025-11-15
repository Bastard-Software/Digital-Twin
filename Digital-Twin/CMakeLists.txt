cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

include("${CMAKE_DIR}/CompilerSettings.cmake" NO_POLICY_SCOPE)

set(INC_DIRS
    "${ROOT_DIR}/${PROJECT_NAME}/*.h"
    "${ROOT_DIR}/${PROJECT_NAME}/*.hpp"
    "${ROOT_DIR}/${PROJECT_NAME}/*.inl"
)

set(SRC_DIRS
    "${ROOT_DIR}/${PROJECT_NAME}/*.h"
    "${ROOT_DIR}/${PROJECT_NAME}/*.hpp"
    "${ROOT_DIR}/${PROJECT_NAME}/*.inl"
    "${ROOT_DIR}/${PROJECT_NAME}/*.cpp"
)

include("${CMAKE_DIR}/GroupFiles.cmake" NO_POLICY_SCOPE)

add_library(${PROJECT_NAME} STATIC
    ${INCLUDE_FILES}
    ${SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${TWIN_DIR}/
  ${THIRD_PARTY_DIR}/volk/
  ${THIRD_PARTY_DIR}/glm/
) 

target_link_libraries(${PROJECT_NAME} PRIVATE
    volk
)