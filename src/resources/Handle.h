#pragma once
#include <cstdint>
#include <functional>

namespace DigitalTwin
{

    // Base Handle structure: [ Generation (32 bits) | Index (32 bits) ]
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

        inline bool operator==( const Handle& other ) const { return value == other.value; }
        inline bool operator!=( const Handle& other ) const { return value != other.value; }
        // Needed for map ordering
        inline bool operator<( const Handle& other ) const { return value < other.value; }
    };

// Macro to define strongly typed handles.
// Using 'inline static const' (C++17) allows defining the Invalid constant
// directly in the header, avoiding the need for a separate .cpp definition.
#define DEFINE_HANDLE( Name )                                                                                                                        \
    struct Name : public Handle                                                                                                                      \
    {                                                                                                                                                \
        using Handle::Handle;                                                                                                                        \
        static const Name Invalid;                                                                                                                   \
    };                                                                                                                                               \
    inline const Name Name::Invalid = Name( 0, 0 );

    // Define all handles types here
    DEFINE_HANDLE( TextureHandle );
    DEFINE_HANDLE( BufferHandle );
    DEFINE_HANDLE( ShaderHandle );
    DEFINE_HANDLE( SamplerHandle );
    DEFINE_HANDLE( MeshHandle );
    DEFINE_HANDLE( ComputePipelineHandle );
    DEFINE_HANDLE( GraphicsPipelineHandle );

} // namespace DigitalTwin

// Specialization for std::hash to allow Handles in std::unordered_map
namespace std
{
    template<>
    struct hash<DigitalTwin::Handle>
    {
        size_t operator()( const DigitalTwin::Handle& h ) const { return hash<uint64_t>()( h.value ); }
    };

    // We can also specialize for derived types if needed, or rely on implicit casting to base Handle used as key.
    // For strict type safety in maps, you might want to specialize each, but typically casting is fine.
} // namespace std