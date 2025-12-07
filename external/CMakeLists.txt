# Funkcja pomocnicza: Rekurencyjnie ustawia folder dla wszystkich targetów w katalogu
function(set_target_folder_recursive current_dir folder_name)
    # Pobierz wszystkie targety w bieżącym katalogu
    get_property(targets DIRECTORY "${current_dir}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(t ${targets})
        # Ustaw folder (np. "external/shaderc")
        set_property(TARGET ${t} PROPERTY FOLDER "${folder_name}")
    endforeach()

    # Pobierz podkatalogi i wywołaj funkcję rekurencyjnie
    get_property(subdirs DIRECTORY "${current_dir}" PROPERTY SUBDIRECTORIES)
    foreach(subdir ${subdirs})
        set_target_folder_recursive("${subdir}" "${folder_name}")
    endforeach()
endfunction()

# ==============================================================================
# EXTERNAL LIBRARIES
# ==============================================================================

if(EXISTS ${EXTERNAL_DIR}/nanobind)
    add_subdirectory(${EXTERNAL_DIR}/nanobind)
    message(STATUS "Using bundled nanobind")
else()
    message(FATAL_ERROR "nanobind not found. Run: python scripts/fetch_dependencies.py")
endif()

if(EXISTS ${EXTERNAL_DIR}/volk)
    if(WIN32)
        set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
    endif()
    add_subdirectory(${EXTERNAL_DIR}/volk)
    set_property(TARGET volk PROPERTY FOLDER "external")
    message(STATUS "Using bundled volk")
else()
    message(FATAL_ERROR "volk not found.")
endif()

if(EXISTS ${EXTERNAL_DIR}/VulkanMemoryAllocator)
    add_library(VulkanMemoryAllocator INTERFACE)
    target_include_directories(VulkanMemoryAllocator INTERFACE ${EXTERNAL_DIR}/VulkanMemoryAllocator/include)
    target_compile_definitions(VulkanMemoryAllocator INTERFACE VMA_STATIC_VULKAN_FUNCTIONS=0 VMA_DYNAMIC_VULKAN_FUNCTIONS=0)
    set_property(TARGET VulkanMemoryAllocator PROPERTY FOLDER "external")
    message(STATUS "Using bundled VMA")
else()
    message(FATAL_ERROR "VMA not found.")
endif()

if(EXISTS ${EXTERNAL_DIR}/entt)
    add_library(entt INTERFACE)
    target_include_directories(entt INTERFACE ${EXTERNAL_DIR}/entt/src)
    target_compile_definitions(entt INTERFACE ENTT_USE_ATOMIC=ON)
    set_property(TARGET entt PROPERTY FOLDER "external")
    message(STATUS "Using bundled entt")
else()
    message(FATAL_ERROR "entt not found.")
endif()

if(EXISTS ${EXTERNAL_DIR}/spdlog)
    set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
    set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${EXTERNAL_DIR}/spdlog)
    set_property(TARGET spdlog PROPERTY FOLDER "external")
    message(STATUS "Using bundled spdlog")
else()
    message(WARNING "spdlog not found.")
endif()

if(EXISTS ${EXTERNAL_DIR}/googletest)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${EXTERNAL_DIR}/googletest)
    
    # Używamy naszej funkcji, bo gtest tworzy 4 targety (gtest, gtest_main, gmock, gmock_main)
    set_target_folder_recursive(${EXTERNAL_DIR}/googletest "external/googletest")
    
    message(STATUS "Using bundled googletest")
else()
    message(WARNING "googletest not found.")
endif()

# ==============================================================================
# SHADERC DEPENDENCIES (Order matters!)
# ==============================================================================

# 1. SPIRV-Headers
if(EXISTS ${EXTERNAL_DIR}/spirv-headers)
    add_subdirectory(${EXTERNAL_DIR}/spirv-headers)
    set_target_folder_recursive(${EXTERNAL_DIR}/spirv-headers "external/shaderc/headers")
    message(STATUS "Using bundled spirv-headers")
else()
    message(FATAL_ERROR "spirv-headers not found. Run fetch_dependencies.py")
endif()

# 2. SPIRV-Tools (Needs headers)
if(EXISTS ${EXTERNAL_DIR}/spirv-tools)
    # Wyłączamy zbędne narzędzia, żeby zmniejszyć liczbę projektów
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE) # To usunie spirv-as, spirv-dis itp.
    set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
    
    add_subdirectory(${EXTERNAL_DIR}/spirv-tools)
    
    # Sprzątamy to co zostało
    set_target_folder_recursive(${EXTERNAL_DIR}/spirv-tools "external/shaderc/tools")
    
    message(STATUS "Using bundled spirv-tools")
else()
    message(FATAL_ERROR "spirv-tools not found. Run fetch_dependencies.py")
endif()

# 3. glslang
if(EXISTS ${EXTERNAL_DIR}/glslang)
    set(GLSLANG_TESTS OFF CACHE BOOL "" FORCE)
    set(GLSLANG_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(ENABLE_GLSLANG_BINARIES OFF CACHE BOOL "" FORCE) # Wyłączamy glslangValidator.exe w solucji
    
    add_subdirectory(${EXTERNAL_DIR}/glslang)
    
    set_target_folder_recursive(${EXTERNAL_DIR}/glslang "external/shaderc/glslang")
    
    message(STATUS "Using bundled glslang")
else()
    message(FATAL_ERROR "glslang not found. Run fetch_dependencies.py")
endif()

# 4. Shaderc (Final)
if(EXISTS ${EXTERNAL_DIR}/shaderc)
    set(SHADERC_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
    set(SHADERC_SKIP_COPYRIGHT_CHECK ON CACHE BOOL "" FORCE)
    set(SHADERC_ENABLE_SHARED_CRT ON CACHE BOOL "" FORCE)
    
    add_subdirectory(${EXTERNAL_DIR}/shaderc)
    
    # To posprząta "shaderc_combined", "shaderc_util" i inne
    set_target_folder_recursive(${EXTERNAL_DIR}/shaderc "external/shaderc")
    
    message(STATUS "Using bundled shaderc")
else()
    message(FATAL_ERROR "shaderc not found. Run fetch_dependencies.py")
endif()

# --- SPIRV-REFLECT ---
if(EXISTS ${EXTERNAL_DIR}/spirv-reflect)
    add_library(spirv-reflect STATIC 
        ${EXTERNAL_DIR}/spirv-reflect/spirv_reflect.c
        ${EXTERNAL_DIR}/spirv-reflect/spirv_reflect.h
    )
    target_include_directories(spirv-reflect PUBLIC ${EXTERNAL_DIR}/spirv-reflect)
    set_property(TARGET spirv-reflect PROPERTY FOLDER "external")
    message(STATUS "Using bundled spirv-reflect")
else()
    message(FATAL_ERROR "spirv-reflect not found. Run: python scripts/fetch_dependencies.py")
endif()