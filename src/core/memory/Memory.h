#pragma once
#include <cstddef>
#include <cstdint>

namespace DigitalTwin
{

    // Helper struct to track allocation details in Debug mode
    struct MemoryBlockInfo
    {
        size_t      size;
        const char* file;
        uint32_t    line;
    };

    // Interface for all allocators
    class IAllocator
    {
    public:
        virtual ~IAllocator() = default;

        virtual void* Allocate( size_t size, const char* file, uint32_t line ) = 0;
        virtual void  Free( void* ptr )                                        = 0;
    };

} // namespace DigitalTwin