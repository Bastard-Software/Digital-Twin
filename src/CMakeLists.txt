cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

include("${CMAKE_DIR}/compilerSettings.cmake" NO_POLICY_SCOPE)

file(GLOB_RECURSE SOURCE_FILES
    "${SRC_DIR}/*.h"
    "${SRC_DIR}/*.hpp"
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.inl"
)

set(STATIC_SOURCE_FILES "")
foreach(source_file ${SOURCE_FILES})
    # Wyklucz pliki w katalogu python/bindings
    if(NOT source_file MATCHES ".*python.*bindings.*\\.cpp$")
        list(APPEND STATIC_SOURCE_FILES ${source_file})
    endif()
endforeach()

source_group(TREE "${SRC_DIR}" PREFIX "src" FILES ${STATIC_SOURCE_FILES})

# build lib
add_library(${PROJECT_NAME} STATIC
    ${STATIC_SOURCE_FILES}
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${SRC_DIR}/
  ${EXTERNAL_DIR}/volk/
  ${EXTERNAL_DIR}/glm/
  ${EXTERNAL_DIR}/entt/src/
) 

target_link_libraries(${PROJECT_NAME} PUBLIC
    volk
    entt
)

# build bindings
pybind11_add_module(${PROJECT_NAME}-Python ${SRC_DIR}/python/bindings.cpp)

target_include_directories(${PROJECT_NAME}-Python PRIVATE
  ${SRC_DIR}/
)

target_link_libraries(${PROJECT_NAME}-Python PRIVATE
    ${PROJECT_NAME}
)

set_target_properties(${PROJECT_NAME}-Python PROPERTIES
    OUTPUT_NAME "DigitalTwin"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

add_custom_command(TARGET ${PROJECT_NAME}-Python POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:${PROJECT_NAME}-Python>
        ${ROOT_DIR}/sandbox-python/DigitalTwin${PYTHON_MODULE_EXTENSION}
    COMMENT "Copying python module to sandbox-python"
)