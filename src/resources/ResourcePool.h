#pragma once
#include "core/Log.h"
#include <DigitalTwinTypes.h>
#include <functional>
#include <memory>
#include <queue>
#include <type_traits>
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Generic pool for managing resources with generational handles.
     * Owns the memory via std::unique_ptr.
     */
    template<typename T, typename HandleType, typename Deleter = std::default_delete<T>>
    class ResourcePool
    {
        struct Slot
        {
            std::unique_ptr<T, Deleter> resource   = nullptr;
            uint32_t                    generation = 1; // 0 is reserved for Invalid handles
        };

    public:
        /**
         * @brief Inserts a resource into the pool.
         * @param resource The resource to manage (pool takes ownership).
         * @return A strongly typed handle to the resource.
         */
        HandleType Insert( std::unique_ptr<T, Deleter> resource )
        {
            uint32_t index;
            if( !m_freeIndices.empty() )
            {
                index = m_freeIndices.front();
                m_freeIndices.pop();
            }
            else
            {
                index = static_cast<uint32_t>( m_slots.size() );
                m_slots.emplace_back();
            }

            Slot& slot    = m_slots[ index ];
            slot.resource = std::move( resource );
            // Generation is already set correctly (incremented during removal)

            return HandleType( index, slot.generation );
        }

        /**
         * @brief Retrieves a raw pointer to the resource.
         * @return Pointer to T or nullptr if handle is invalid/stale.
         */
        T* Get( HandleType handle ) const
        {
            if( !handle.IsValid() )
                return nullptr;

            uint32_t index = handle.GetIndex();
            if( index >= m_slots.size() )
                return nullptr;

            const Slot& slot = m_slots[ index ];
            // Validation: Check if the handle's generation matches the slot's generation
            if( slot.generation != handle.GetGeneration() )
            {
                return nullptr; // Handle is stale (refers to a deleted object)
            }

            return slot.resource.get();
        }

        /**
         * @brief Removes a resource from the pool and returns ownership.
         * This is used by the ResourceManager to move the resource to the deletion queue.
         */
        std::unique_ptr<T, Deleter> Remove( HandleType handle )
        {
            if( !handle.IsValid() )
                return nullptr;

            uint32_t index = handle.GetIndex();
            if( index >= m_slots.size() )
                return nullptr;

            Slot& slot = m_slots[ index ];
            if( slot.generation != handle.GetGeneration() )
            {
                DT_WARN( "[ResourcePool] Attempted to remove invalid/stale handle." );
                return nullptr;
            }

            // Move resource out
            std::unique_ptr<T, Deleter> resource = std::move( slot.resource );

            // Increment generation so any existing handles to this slot become invalid
            slot.generation++;

            // Mark index as free
            m_freeIndices.push( index );

            return resource;
        }

        void Clear()
        {
            m_slots.clear();
            std::queue<uint32_t> empty;
            std::swap( m_freeIndices, empty );
        }

        template<typename Func>
        void ForEach( Func func )
        {
            for( auto& slot: m_slots )
            {
                if( slot.resource )
                { // Only iterate active resources
                    func( slot.resource.get() );
                }
            }
        }

    private:
        std::vector<Slot>    m_slots;
        std::queue<uint32_t> m_freeIndices;
    };
} // namespace DigitalTwin