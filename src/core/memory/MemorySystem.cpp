#include "core/memory/MemorySystem.h"

#include "core/Log.h" // We use our new Logger here!

namespace DigitalTwin
{

    MemorySystem::MemorySystem()
        : m_systemAllocator( this )
    {
    }

    MemorySystem::~MemorySystem()
    {
    }

    void MemorySystem::Initialize()
    {
        DT_INFO( "Memory System Initialized." );
#ifdef DT_DEBUG
        m_totalAllocated = 0;
        m_allocations.clear();
#endif
    }

    void MemorySystem::Shutdown()
    {
#ifdef DT_DEBUG
        std::lock_guard<std::mutex> lock( m_mutex );

        if( !m_allocations.empty() )
        {
            DT_ERROR( "Memory Leaks Detected! Count: {0}, Total Bytes: {1}", m_allocations.size(), m_totalAllocated );
            for( const auto& [ ptr, info ]: m_allocations )
            {
                DT_ERROR( " - Leak: {0} bytes at {1}:{2}", info.size, info.file, info.line );
                // TODO: Do we wnat to use free here or let user fix the leak
            }
        }
        else
        {
            DT_INFO( "Memory System Shutdown. No leaks detected." );
        }
        m_allocations.clear();
#else
        DT_INFO( "Memory System Shutdown." );
#endif
    }

    void MemorySystem::TrackAllocation( void* ptr, size_t size, const char* file, uint32_t line )
    {
#ifdef DT_DEBUG
        std::lock_guard<std::mutex> lock( m_mutex );
        if( ptr )
        {
            m_allocations[ ptr ] = { size, file, line };
            m_totalAllocated += size;
        }
#endif
    }

    void MemorySystem::TrackDeallocation( void* ptr )
    {
#ifdef DT_DEBUG
        std::lock_guard<std::mutex> lock( m_mutex );
        if( ptr )
        {
            auto it = m_allocations.find( ptr );
            if( it != m_allocations.end() )
            {
                m_totalAllocated -= it->second.size;
                m_allocations.erase( it );
            }
            else
            {
                DT_ERROR( "Attempted to free unknown pointer: {0}", ptr );
            }
        }
#endif
    }

} // namespace DigitalTwin