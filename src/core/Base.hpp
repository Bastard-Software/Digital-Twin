#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

namespace DigitalTwin
{
    using bool_t    = bool;
    using float32_t = float;
    using float64_t = double;

    // Error codes
    enum class Result : int32_t
    {
        SUCCESS         = 0,
        FAIL            = -1,
        NOT_IMPLEMENTED = -2,
        INVALID_ARGS    = -3,
        TIMEOUT         = -4,
        OUT_OF_MEMORY   = -10
    };

    inline std::string_view toString( Result result )
    {
        switch( result )
        {
            case Result::SUCCESS:
                return "SUCCESS";
            case Result::FAIL:
                return "FAIL";
            case Result::NOT_IMPLEMENTED:
                return "NOT_IMPLEMENTED";
            case Result::INVALID_ARGS:
                return "INVALID_ARGS";
            case Result::TIMEOUT:
                return "TIMEOUT";
            case Result::OUT_OF_MEMORY:
                return "OUT_OF_MEMORY";
            default:
                return "UNKNOWN";
        }
    }

    template<typename T>
    using HeapArray = std::vector<T>;

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

    template<typename T>
    using Weak = std::weak_ptr<T>;
} // namespace DigitalTwin

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

#ifdef DT_DEBUG
#    define DT_CHECK( x )                                                                                                                            \
        {                                                                                                                                            \
            Result r = ( x );                                                                                                                        \
            if( r != ::DigitalTwin::Result::SUCCESS )                                                                                                \
            {                                                                                                                                        \
                DT_CORE_ERROR( "Check Failed: {0}", ::DigitalTwin::toString( r ) );                                                                  \
                DT_DEBUGBREAK();                                                                                                                     \
            }                                                                                                                                        \
        }
#else
#    define DT_CHECK( x ) ( x )
#endif
