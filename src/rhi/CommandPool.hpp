#pragma once

#include "core/Base.hpp"
#include <volk.h>

namespace DigitalTwin
{
    class CommandPool
    {
    public:
        CommandPool( VkDevice device, const VolkDeviceTable& api, uint32_t queueFamilyIndex );
        ~CommandPool();

        VkCommandBuffer Allocate();
        void            Reset();

    private:
        VkDevice               m_device;
        const VolkDeviceTable& m_api;
        VkCommandPool          m_commandPool;
    };
} // namespace DigitalTwin