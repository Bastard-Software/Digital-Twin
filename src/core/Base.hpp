#pragma once

#include <cstdint>
#include <memory>

namespace DigitalTwin
{
    using bool_t    = bool;
    using float32_t = float;
    using float64_t = double;

    // Temporoary aliases for ref count types
    template<typename T>
    using Scope = std::unique_ptr<T>;

    template<typename T, typename... Args>
    constexpr Scope<T> CreateScope( Args&&... args )
    {
        return std::make_unique<T>( std::forward<Args>( args )... );
    }

    template<typename T>
    using Ref = std::shared_ptr<T>;

    template<typename T, typename... Args>
    constexpr Ref<T> CreateRef( Args&&... args )
    {
        return std::make_shared<T>( std::forward<Args>( args )... );
    }
}

#include "Core/Log.hpp"

#if defined( _MSC_VER )
#    define DT_DEBUGBREAK() __debugbreak()
#elif defined( __linux__ ) || defined( __APPLE__ )
#    include <signal.h>
#    define DT_DEBUGBREAK() raise( SIGTRAP )
#else
#    define DT_DEBUGBREAK()
#endif

#ifdef DT_DEBUG
#    define DT_ENABLE_ASSERTS
#endif

#ifdef DT_ENABLE_ASSERTS
#    define DT_ASSERT( x, ... )                                                                                                                      \
        {                                                                                                                                            \
            if( !( x ) )                                                                                                                             \
            {                                                                                                                                        \
                DT_ERROR( "Assertion Failed: {0}", __VA_ARGS__ );                                                                                    \
                DT_DEBUGBREAK();                                                                                                                     \
            }                                                                                                                                        \
        }
#    define DT_CORE_ASSERT( x, ... )                                                                                                                 \
        {                                                                                                                                            \
            if( !( x ) )                                                                                                                             \
            {                                                                                                                                        \
                DT_CORE_ERROR( "Assertion Failed: {0}", __VA_ARGS__ );                                                                               \
                DT_DEBUGBREAK();                                                                                                                     \
            }                                                                                                                                        \
        }
#else
#    define DT_ASSERT( x, ... )
#    define DT_CORE_ASSERT( x, ... )
#endif