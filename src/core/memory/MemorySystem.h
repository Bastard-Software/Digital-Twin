#pragma once
#include "core/memory/Memory.h"
#include "core/memory/SystemAllocator.h"
#include <mutex>
#include <unordered_map>

namespace DigitalTwin
{

    class MemorySystem
    {
    public:
        MemorySystem();
        ~MemorySystem();

        void Initialize();
        void Shutdown();

        IAllocator* GetSystemAllocator() { return &m_systemAllocator; }

        // Debug Tracking Functions
        void TrackAllocation( void* ptr, size_t size, const char* file, uint32_t line );
        void TrackDeallocation( void* ptr );

#ifdef DT_DEBUG
        size_t GetAllocationCount()
        {
            std::lock_guard<std::mutex> lock( m_mutex );
            return m_allocations.size();
        }
#endif

    private:
        SystemAllocator m_systemAllocator;

        // Tracking Data (Debug Only)
#ifdef DT_DEBUG
        std::mutex m_mutex;
        // Map pointer address -> Block Info for O(1) lookup
        std::unordered_map<void*, MemoryBlockInfo> m_allocations;
        size_t                                     m_totalAllocated = 0;
#endif
    };

} // namespace DigitalTwin