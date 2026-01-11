#include "core/Core.h"

#define VMA_IMPLEMENTATION

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#    define VMA_STATIC_VULKAN_FUNCTIONS 0
#endif

#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#    define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#endif

#if defined( _MSC_VER )
#    pragma warning( push )
#    pragma warning( disable : 4100 ) // unreferenced formal parameter
#    pragma warning( disable : 4189 ) // local variable is initialized but not referenced
#    pragma warning( disable : 4127 ) // conditional expression is constant
#    pragma warning( disable : 4324 ) // structure was padded due to alignment specifier
#    pragma warning( disable : 4505 ) // unreferenced local function has been removed
#endif

#include <vk_mem_alloc.h>

#if defined( _MSC_VER )
#    pragma warning( pop )
#endif