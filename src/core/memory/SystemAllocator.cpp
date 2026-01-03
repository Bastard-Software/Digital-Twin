#include "core/memory/SystemAllocator.h"

#include "core/memory/MemorySystem.h"

namespace DigitalTwin
{

    SystemAllocator::SystemAllocator( MemorySystem* owner )
        : m_owner( owner )
    {
    }

    void* SystemAllocator::Allocate( size_t size, const char* file, uint32_t line )
    {
        void* ptr = std::malloc( size );
        if( ptr && m_owner )
        {
            m_owner->TrackAllocation( ptr, size, file, line );
        }

        return ptr;
    }

    void SystemAllocator::Free( void* ptr )
    {
        if( !ptr )
            return;

        if( m_owner )
        {
            m_owner->TrackDeallocation( ptr );
        }
        std::free( ptr );
    }

} // namespace DigitalTwin