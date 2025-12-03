if(EXISTS ${EXTERNAL_DIR}/nanobind)
    add_subdirectory(${EXTERNAL_DIR}/nanobind)

    message(STATUS "Using bundled nanobind")
else()
    message(FATAL_ERROR "nanobind not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${EXTERNAL_DIR}/volk)
    if(WIN32)
        set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
        message(STATUS "Platform: Windows - using VK_USE_PLATFORM_WIN32_KHR")
    else()
        message(WARNING "Unknown platform - Vulkan may not work correctly")
    endif()

    add_subdirectory(${EXTERNAL_DIR}/volk)
    set_property(TARGET volk PROPERTY FOLDER "external")
    message(STATUS "Using bundled volk")
else()
    message(FATAL_ERROR "volk not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${EXTERNAL_DIR}/VulkanMemoryAllocator)
    add_library(VulkanMemoryAllocator INTERFACE)
    target_include_directories(VulkanMemoryAllocator INTERFACE 
        ${EXTERNAL_DIR}/VulkanMemoryAllocator/include
    )
    
    target_compile_definitions(VulkanMemoryAllocator INTERFACE
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=0
    )

    set_property(TARGET VulkanMemoryAllocator PROPERTY FOLDER "external")
    message(STATUS "Using bundled VulkanMemoryAllocator")
else()
    message(FATAL_ERROR "VulkanMemoryAllocator not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${EXTERNAL_DIR}/entt)
    add_library(entt INTERFACE)
    target_include_directories(entt INTERFACE ${EXTERNAL_DIR}/entt/src)
    target_compile_definitions(entt INTERFACE ENTT_USE_ATOMIC=ON)
    
    set_property(TARGET entt PROPERTY FOLDER "external")
    message(STATUS "Using bundled entt")
else()
    message(FATAL_ERROR "entt not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${EXTERNAL_DIR}/spdlog)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${EXTERNAL_DIR}/spdlog)
    set_property(TARGET spdlog PROPERTY FOLDER "external")
    message(STATUS "Using bundled spdlog")
else()
    message(WARNING "spdlog not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${EXTERNAL_DIR}/googletest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    
    add_subdirectory(${EXTERNAL_DIR}/googletest)
    set_property(TARGET gtest PROPERTY FOLDER "external")
    set_property(TARGET gtest_main PROPERTY FOLDER "external")
    set_property(TARGET gmock PROPERTY FOLDER "external")
    set_property(TARGET gmock_main PROPERTY FOLDER "external")
    
    message(STATUS "Using bundled googletest")
else()
    message(WARNING "googletest not found. Run: python scripts/fetch_dependencies.py")
endif()