#include "rhi/CommandPool.hpp"

namespace DigitalTwin
{
    CommandPool::CommandPool( VkDevice device, const VolkDeviceTable& api, uint32_t queueFamilyIndex )
        : m_device( device )
        , m_api( api )
        , m_commandPool( VK_NULL_HANDLE )
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if( vkCreateCommandPool( m_device, &poolInfo, nullptr, &m_commandPool ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create thread-local Command Pool for family index: {}", queueFamilyIndex );
        }
    }

    CommandPool::~CommandPool()
    {
        if( m_commandPool != VK_NULL_HANDLE )
        {
            vkDestroyCommandPool( m_device, m_commandPool, nullptr );
        }
    }

    VkCommandBuffer CommandPool::Allocate()
    {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool        = m_commandPool;
        allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        if( m_api.vkAllocateCommandBuffers( m_device, &allocInfo, &commandBuffer ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to allocate command buffer!" );
            return VK_NULL_HANDLE;
        }

        return commandBuffer;
    }

    void CommandPool::Reset()
    {
        m_api.vkResetCommandPool( m_device, m_commandPool, 0 );
    }

} // namespace DigitalTwin