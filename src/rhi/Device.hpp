#pragma once
#include "core/Base.hpp"
#include "rhi/CommandPool.hpp"
#include "rhi/CommandQueue.hpp"
#include <vk_mem_alloc.h>
#include <volk.h>

namespace DigitalTwin
{
    struct DeviceDesc
    {
        bool_t headless = false;
    };

    class Device
    {
    public:
        Device( VkPhysicalDevice physicalDevice );
        ~Device();

        Result Init( DeviceDesc desc );
        void   Shutdown();

        /**
         * @brief Waits for a specific value on the queue's timeline semaphore.
         * @param queue The queue to wait on.
         * @param waitValue The value to wait for.
         * @param timeout Timeout in nanoseconds.
         * @return Result::SUCCESS, Result::TIMEOUT, or Result::FAIL.
         */
        Result WaitForQueue( Ref<CommandQueue> queue, uint64_t waitValue, uint64_t timeout = UINT64_MAX );

        // Getters for queues (may point to the same object if hardware has a single queue family)
        Ref<CommandQueue> GetGraphicsQueue() const { return m_graphicsQueue; }
        Ref<CommandQueue> GetComputeQueue() const { return m_computeQueue; }
        Ref<CommandQueue> GetTransferQueue() const { return m_transferQueue; }

        VkDevice               GetHandle() const { return m_device; }
        VkPhysicalDevice       GetPhysicalDevice() const { return m_physicalDevice; }
        VmaAllocator           GetAllocator() const { return m_allocator; }
        const VolkDeviceTable& GetAPI() const { return m_api; }

    private:
        struct QueueFamilyIndices;
        QueueFamilyIndices FindQueueFamilies( VkPhysicalDevice device );

    private:
        VkPhysicalDevice m_physicalDevice;
        VkDevice         m_device;
        VmaAllocator     m_allocator;
        VolkDeviceTable  m_api;
        DeviceDesc       m_desc;

        Ref<CommandQueue> m_graphicsQueue;
        Ref<CommandQueue> m_computeQueue;
        Ref<CommandQueue> m_transferQueue;

        std::unordered_map<std::thread::id, Ref<CommandPool>> m_commandPools;
        std::mutex                                            m_poolMutex;
    };

} // namespace DigitalTwin