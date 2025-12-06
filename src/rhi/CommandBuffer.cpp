#include "rhi/CommandBuffer.hpp"

namespace DigitalTwin
{
    CommandBuffer::CommandBuffer( VkDevice device, const VolkDeviceTable* api, VkCommandPool pool, VkCommandBuffer buffer )
        : m_device( device )
        , m_api( api )
        , m_pool( pool )
        , m_commandBuffer( buffer )
    {
    }

    CommandBuffer::~CommandBuffer()
    {
        if( m_commandBuffer != VK_NULL_HANDLE )
        {
            m_api->vkFreeCommandBuffers( m_device, m_pool, 1, &m_commandBuffer );
        }
    }

    void CommandBuffer::Begin( VkCommandBufferUsageFlags flags )
    {
        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags                    = flags;
        m_api->vkBeginCommandBuffer( m_commandBuffer, &beginInfo );
    }

    void CommandBuffer::End()
    {
        m_api->vkEndCommandBuffer( m_commandBuffer );
    }

    void CommandBuffer::BindComputePipeline( VkPipeline pipeline )
    {
        ( void )pipeline;
        // NOP
    }

    void CommandBuffer::Dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ )
    {
        ( void )groupCountX;
        ( void )groupCountY;
        ( void )groupCountZ;
        // NOP
    }

    void CommandBuffer::BindDescriptorSets( VkPipelineLayout layout, uint32_t firstSet, const std::vector<VkDescriptorSet>& sets )
    {
        ( void )layout;
        ( void )firstSet;
        ( void )sets;
        // NOP
    }

    void CommandBuffer::PipelineBarrier( VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags dependencyFlags,
                                         const std::vector<VkMemoryBarrier>& memoryBarriers, const std::vector<VkBufferMemoryBarrier>& bufferBarriers,
                                         const std::vector<VkImageMemoryBarrier>& imageBarriers )
    {
        ( void )srcStage;
        ( void )dstStage;
        ( void )dependencyFlags;
        ( void )memoryBarriers;
        ( void )bufferBarriers;
        ( void )imageBarriers;
        // NOP
    }

} // namespace DigitalTwin