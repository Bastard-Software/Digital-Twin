#pragma once
#include "rhi/RHITypes.h"

#include "rhi/CommandBuffer.h"
#include "rhi/DescriptorAllocator.h"
#include <memory>
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    /**
     * @brief Manages per-thread resources.
     * Owns a CommandPool, CommandBuffer wrappers, and DescriptorAllocator.
     */
    class ThreadContext
    {
    public:
        ThreadContext() = default;
        ~ThreadContext();

        Result Initialize( VkDevice device, const VolkDeviceTable* api, QueueType type, uint32_t qfNdx );
        void   Shutdown();

        // Resets pools (called at frame start for main thread, or internally for recycling)
        void Reset();

        // Returns Handle to Command Buffer (valid until Reset)
        CommandBufferHandle CreateCommandBuffer();

        // Retrieve wrapper from handle
        CommandBuffer* GetCommandBuffer( CommandBufferHandle handle );

        DescriptorAllocator& GetDescriptorAllocator() { return *m_descriptorAllocator; }

    private:
        VkDevice               m_device = VK_NULL_HANDLE;
        const VolkDeviceTable* m_api    = nullptr;
        QueueType              m_type;

        VkCommandPool              m_commandPool = VK_NULL_HANDLE;
        Scope<DescriptorAllocator> m_descriptorAllocator;

        // Pool of CommandBuffer wrappers. Resetting the pool allows reuse.
        std::vector<CommandBuffer> m_commandBuffers;
        uint32_t                   m_activeCmdBufferCount = 0;
    };
} // namespace DigitalTwin