#pragma once

#include "core/Base.hpp"
#include <mutex>
#include <volk.h>

namespace DigitalTwin
{
    enum class QueueType
    {
        GRAPHICS,
        COMPUTE,
        TRANSFER,
        _MAX_ENUM,
    };

    // Description of a semaphore to wait on
    struct QueueWaitInfo
    {
        VkSemaphore           semaphore;
        uint64_t              value;     // Target value for Timeline, ignored (0) for Binary
        VkPipelineStageFlags2 stageMask; // Pipeline stage that blocks waiting for this semaphore
    };

    // Description of a semaphore to signal
    struct QueueSignalInfo
    {
        VkSemaphore           semaphore;
        uint64_t              value;     // Signal value for Timeline, ignored (0) for Binary
        VkPipelineStageFlags2 stageMask; // Pipeline stage that signals this semaphore
    };

    // Consolidated submit information
    struct SubmitInfo
    {
        std::vector<VkCommandBuffer> commandBuffers;
        std::vector<QueueWaitInfo>   waitSemaphores;
        std::vector<QueueSignalInfo> signalSemaphores;
    };

    class Queue
    {
    public:
        Queue( VkDevice device, const VolkDeviceTable& table, uint32_t queueFamilyIndex, QueueType type );
        ~Queue();

        /**
         * @brief Submits command buffers to the queue with advanced synchronization (Vulkan 1.3).
         * Uses vkQueueSubmit2. No Fence needed - rely on Timeline Semaphore for CPU sync.
         * * @param info Struct containing command buffers and user-defined semaphores.
         * @param outSignalValue [Out, Optional] Returns the value of the internal timeline semaphore for this submission.
         * @return Result::SUCCESS or Result::FAIL.
         */
        Result Submit( const SubmitInfo& info, uint64_t* outSignalValue = nullptr );
        Result Submit( VkCommandBuffer commandBuffer, uint64_t& outSignalValue );


        bool_t      IsValueCompleted( uint64_t fenceValue );
        VkQueue     GetHandle() const { return m_queue; }
        uint32_t    GetFamilyIndex() const { return m_queueFamilyIndex; }
        QueueType   GetType() const { return m_type; }
        uint64_t    GetLastSubmittedValue() const { return m_nextValue - 1; }
        VkSemaphore GetTimelineSemaphore() const { return m_timelineSemaphore; }

    private:
        VkDevice               m_device;
        const VolkDeviceTable& m_api;
        VkQueue                m_queue;
        uint32_t               m_queueFamilyIndex;
        QueueType              m_type;
        std::mutex             m_mutex;

        VkSemaphore m_timelineSemaphore;
        uint64_t    m_nextValue;
    };
} // namespace DigitalTwin