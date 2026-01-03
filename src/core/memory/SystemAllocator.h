#pragma once
#include "core/memory/Memory.h"
#include <cstdlib>

namespace DigitalTwin
{
    class MemorySystem;

    class SystemAllocator : public IAllocator
    {
    public:
        SystemAllocator( MemorySystem* owner );
        ~SystemAllocator() = default;

        void* Allocate( size_t size, const char* file, uint32_t line ) override;
        void  Free( void* ptr ) override;

    private:
        MemorySystem* m_owner;
    };

} // namespace DigitalTwin