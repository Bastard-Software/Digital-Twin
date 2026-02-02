set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    # /W4 - High warning level
    # /permissive- - Enforce strict standards conformance
    # /wd4251 - Disable warning: "class 'type' needs to have dll-interface to be used by clients of class 'type2'"
    # /wd4275 - Disable warning: "non - DLL-interface class used as base for DLL-interface class"
    add_compile_options(/W4 /permissive- /wd4251 /wd4275)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_CONFIGURATION_TYPES MATCHES Debug)
    add_compile_definitions(DT_DEBUG)
endif()

add_compile_definitions(VK_NO_PROTOTYPES)