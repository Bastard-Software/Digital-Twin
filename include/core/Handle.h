#pragma once
#include <cstdint>
#include <functional>

namespace DigitalTwin
{
    /**
     * @brief Base Handle structure representing a resource reference.
     * Consists of an Index (32-bit) and a Generation (32-bit) for safe versioning.
     */
    struct Handle
    {
        uint64_t value = 0;

        Handle() = default;
        Handle( uint32_t index, uint32_t generation )
        {
            // Pack index and generation into a single 64-bit integer
            value = ( ( uint64_t )generation << 32 ) | index;
        }

        inline uint32_t GetIndex() const { return ( uint32_t )( value & 0xFFFFFFFF ); }
        inline uint32_t GetGeneration() const { return ( uint32_t )( value >> 32 ); }
        inline bool     IsValid() const { return value != 0; }

        // Boolean operator for easy checking: if (handle) ...
        explicit operator bool() const { return IsValid(); }

        inline bool operator==( const Handle& other ) const { return value == other.value; }
        inline bool operator!=( const Handle& other ) const { return value != other.value; }
        inline bool operator<( const Handle& other ) const { return value < other.value; }
    };

} // namespace DigitalTwin

// Enable hashing for Handles so they can be used as keys in std::unordered_map
namespace std
{
    template<>
    struct hash<DigitalTwin::Handle>
    {
        size_t operator()( const DigitalTwin::Handle& h ) const { return hash<uint64_t>()( h.value ); }
    };
} // namespace std

/**
 * @brief Macro to define a strong type handle derived from the base Handle struct.
 * Usage: DEFINE_HANDLE( MyHandle );
 */
#define DEFINE_HANDLE( Name )                                                                                                                        \
    struct Name : public ::DigitalTwin::Handle                                                                                                       \
    {                                                                                                                                                \
        using Handle::Handle;                                                                                                                        \
        static const Name Invalid;                                                                                                                   \
    };                                                                                                                                               \
    inline const Name Name::Invalid = Name( 0, 0 )