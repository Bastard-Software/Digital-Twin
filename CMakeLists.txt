cmake_minimum_required(VERSION 3.24)

project(Digital-Twin)

set(CMAKE_DIR "${CMAKE_SOURCE_DIR}/cmake")
set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(TWIN_DIR "${CMAKE_SOURCE_DIR}/Digital-Twin")
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/ThirdParty")

include("${CMAKE_DIR}/CompilerSettings.cmake" NO_POLICY_SCOPE)

add_subdirectory(${ROOT_DIR}/Digital-Twin)
add_subdirectory(${ROOT_DIR}/Sandbox)

if(EXISTS ${THIRD_PARTY_DIR}/pybind11)
    add_subdirectory(ThirdParty/pybind11)
    message(STATUS "Using bundled pybind11")
else()
    message(FATAL_ERROR "pybind11 not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${THIRD_PARTY_DIR}/volk)
    if(WIN32)
        set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
        message(STATUS "Platform: Windows - using VK_USE_PLATFORM_WIN32_KHR")
    else()
        message(WARNING "Unknown platform - Vulkan may not work correctly")
    endif()

    add_subdirectory(ThirdParty/volk)
    set_property(TARGET volk PROPERTY FOLDER "ThirdParty")
    message(STATUS "Using bundled volk")
else()
    message(FATAL_ERROR "volk not found. Run: python scripts/fetch_dependencies.py")
endif()

add_subdirectory(${ROOT_DIR}/Digital-Twin-Python)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)