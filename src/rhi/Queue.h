#pragma once

#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <volk.h>

namespace DigitalTwin
{
    /**
     * @brief Wrapper around a Vulkan Queue.
     * Includes a Timeline Semaphore for synchronization.
     * Lifecycle is managed by the Device (RAII).
     */
    class Queue
    {
    public:
        Queue( VkDevice device, const VolkDeviceTable& table, uint32_t familyIndex, QueueType type );
        ~Queue();

        VkQueue   GetHandle() const { return m_queue; }
        uint32_t  GetFamilyIndex() const { return m_familyIndex; }
        QueueType GetType() const { return m_type; }

        /**
         * @brief Checks if the timeline semaphore has reached the specified value.
         * @return True if the GPU execution has passed this point.
         */
        bool_t IsValueCompleted( uint64_t value ) const;

        /**
         * @brief Returns the value that will be signaled by the next submission.
         * Effectively, the 'current' time of the CPU regarding submissions.
         */
        uint64_t GetLastSubmittedValue() const { return m_nextValue - 1; }

        /**
         * @brief Returns the timeline semaphore associated with this queue.
         * Used for CPU-GPU and GPU-GPU synchronization.
         */
        VkSemaphore GetTimelineSemaphore() const { return m_timelineSemaphore; }

        /**
         * @brief Waits until the queue is idle.
         */
        void WaitIdle() const;

    private:
        VkDevice               m_device; // Not owned
        const VolkDeviceTable& m_api;    // Reference to Device's function table
        VkQueue                m_queue;  // Native Handle
        uint32_t               m_familyIndex;
        QueueType              m_type;

        VkSemaphore m_timelineSemaphore; // Owned, created in constructor
        uint64_t    m_nextValue = 1;     // Next wait value for the timeline
    };
} // namespace DigitalTwin