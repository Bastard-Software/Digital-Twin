set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug OR CMAKE_CONFIGURATION_TYPES MATCHES Debug)
    add_compile_definitions(DT_DEBUG)
endif()

add_compile_definitions(VK_NO_PROTOTYPES)