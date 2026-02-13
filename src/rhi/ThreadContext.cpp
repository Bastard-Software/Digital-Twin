#include "rhi/ThreadContext.h"

#include "core/Log.h"

namespace DigitalTwin
{
    ThreadContext::~ThreadContext()
    {
        Shutdown();
    }

    Result ThreadContext::Initialize( VkDevice device, const VolkDeviceTable* api, QueueType type, uint32_t qfNdx )
    {
        m_device = device;
        m_api    = api;
        m_type   = type;

        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex        = qfNdx;

        if( m_api->vkCreateCommandPool( m_device, &poolInfo, nullptr, &m_commandPool ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create ThreadContext Command Pool!" );
            return Result::FAIL;
        }

        m_descriptorAllocator = std::make_unique<DescriptorAllocator>( device, api );
        m_descriptorAllocator->Initialize();

        return Result::SUCCESS;
    }

    void ThreadContext::Shutdown()
    {
        if( m_descriptorAllocator )
        {
            m_descriptorAllocator->Shutdown();
            m_descriptorAllocator.reset();
        }

        if( m_commandPool != VK_NULL_HANDLE )
        {
            m_api->vkDestroyCommandPool( m_device, m_commandPool, nullptr );
            m_commandPool = VK_NULL_HANDLE;
        }
        m_commandBuffers.clear();
    }

    void ThreadContext::Reset()
    {
        if( m_commandPool )
        {
            m_api->vkResetCommandPool( m_device, m_commandPool, 0 );
        }
        if( m_descriptorAllocator )
        {
            m_descriptorAllocator->ResetPools();
        }

        // Reset count, but keep wrapper objects in vector for reuse
        m_activeCmdBufferCount = 0;
    }

    CommandBufferHandle ThreadContext::CreateCommandBuffer()
    {
        // 1. Check if we can reuse an existing wrapper
        if( m_activeCmdBufferCount < m_commandBuffers.size() )
        {
            CommandBuffer& buf = m_commandBuffers[ m_activeCmdBufferCount ];

            // Re-initialize logic. Note: The VkCommandBuffer handle from the pool remains valid
            // and is in the 'Initial' state because we called vkResetCommandPool.
            // We just need to update the wrapper's metadata.
            buf.Initialize( buf.GetHandle(), m_type, m_device, m_api );

            uint32_t index = m_activeCmdBufferCount++;
            return CommandBufferHandle( index, 1 );
        }

        // 2. Allocate new Vulkan Handle and wrapper
        VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool                 = m_commandPool;
        allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount          = 1;

        VkCommandBuffer vkCmd = VK_NULL_HANDLE;
        if( m_api->vkAllocateCommandBuffers( m_device, &allocInfo, &vkCmd ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to allocate Vulkan Command Buffer!" );
            return CommandBufferHandle::Invalid;
        }

        CommandBuffer newBuf;
        newBuf.Initialize( vkCmd, m_type, m_device, m_api );

        m_commandBuffers.push_back( newBuf );

        uint32_t index = m_activeCmdBufferCount++;
        return CommandBufferHandle( index, 1 );
    }

    CommandBuffer* ThreadContext::GetCommandBuffer( CommandBufferHandle handle )
    {
        if( !handle.IsValid() )
            return nullptr;
        uint32_t idx = handle.GetIndex();
        if( idx >= m_activeCmdBufferCount )
            return nullptr;
        return &m_commandBuffers[ idx ];
    }
} // namespace DigitalTwin