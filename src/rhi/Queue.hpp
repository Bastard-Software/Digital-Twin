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

    class Queue
    {
    public:
        Queue( VkDevice device, const VolkDeviceTable& table, uint32_t queueFamilyIndex, QueueType type );
        ~Queue();

        /**
         * @brief Submits a command buffer to the queue.
         * @param commandBuffer The Vulkan command buffer to submit.
         * @param outSignalValue [Out] The timeline semaphore value that will be signaled upon completion.
         * @return Result::SUCCESS or Result::FAIL.
         */
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