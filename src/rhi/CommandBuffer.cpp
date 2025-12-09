#include "rhi/CommandBuffer.hpp"

#include "rhi/Buffer.hpp"
#include "rhi/Pipeline.hpp"

namespace DigitalTwin
{
    CommandBuffer::CommandBuffer( VkDevice device, const VolkDeviceTable* api, VkCommandPool pool, VkCommandBuffer buffer, QueueType type )
        : m_device( device )
        , m_api( api )
        , m_pool( pool )
        , m_commandBuffer( buffer )
        , m_type( type )
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

    void CommandBuffer::BeginRendering( const RenderingInfo& info )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "BeginRendering requires a GRAPHICS queue!" );

        std::vector<VkRenderingAttachmentInfo> colorAttachments;
        for( const auto& att: info.colorAttachments )
        {
            VkRenderingAttachmentInfo vkAtt = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
            vkAtt.imageView                 = att.imageView;
            vkAtt.imageLayout               = att.layout;
            vkAtt.loadOp                    = att.loadOp;
            vkAtt.storeOp                   = att.storeOp;
            vkAtt.clearValue                = att.clearValue;
            colorAttachments.push_back( vkAtt );
        }

        VkRenderingInfo renderInfo      = { VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderInfo.renderArea           = info.renderArea;
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = static_cast<uint32_t>( colorAttachments.size() );
        renderInfo.pColorAttachments    = colorAttachments.data();

        VkRenderingAttachmentInfo depthAtt = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        if( info.useDepth )
        {
            depthAtt.imageView          = info.depthAttachment.imageView;
            depthAtt.imageLayout        = info.depthAttachment.layout;
            depthAtt.loadOp             = info.depthAttachment.loadOp;
            depthAtt.storeOp            = info.depthAttachment.storeOp;
            depthAtt.clearValue         = info.depthAttachment.clearValue;
            renderInfo.pDepthAttachment = &depthAtt;
        }

        m_api->vkCmdBeginRendering( m_commandBuffer, &renderInfo );
    }

    void CommandBuffer::EndRendering()
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "EndRendering requires a GRAPHICS queue!" );
        m_api->vkCmdEndRendering( m_commandBuffer );
    }

    void CommandBuffer::SetViewport( float x, float y, float width, float height, float minDepth, float maxDepth )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "SetViewport requires a GRAPHICS queue!" );
        VkViewport viewport{};
        viewport.x        = x;
        viewport.y        = y;
        viewport.width    = width;
        viewport.height   = height;
        viewport.minDepth = minDepth;
        viewport.maxDepth = maxDepth;
        m_api->vkCmdSetViewport( m_commandBuffer, 0, 1, &viewport );
    }

    void CommandBuffer::SetScissor( int32_t x, int32_t y, uint32_t width, uint32_t height )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "SetScissor requires a GRAPHICS queue!" );
        VkRect2D scissor{};
        scissor.offset = { x, y };
        scissor.extent = { width, height };
        m_api->vkCmdSetScissor( m_commandBuffer, 0, 1, &scissor );
    }

    void CommandBuffer::BindComputePipeline( Ref<ComputePipeline> pipeline )
    {
        DT_CORE_ASSERT( m_type == QueueType::COMPUTE || m_type == QueueType::GRAPHICS, "BindComputePipeline requires COMPUTE or GRAPHICS queue!" );
        m_api->vkCmdBindPipeline( m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline->GetHandle() );
    }

    void CommandBuffer::BindGraphicsPipeline( Ref<GraphicsPipeline> pipeline )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "BindGraphicsPipeline requires a GRAPHICS queue!" );
        m_api->vkCmdBindPipeline( m_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline->GetHandle() );
    }

    void CommandBuffer::BindDescriptorSets( VkPipelineBindPoint bindPoint, VkPipelineLayout layout, uint32_t firstSet,
                                            const std::vector<VkDescriptorSet>& sets )
    {
        if( !sets.empty() )
        {
            m_api->vkCmdBindDescriptorSets( m_commandBuffer, bindPoint, layout, firstSet, static_cast<uint32_t>( sets.size() ), sets.data(), 0,
                                            nullptr );
        }
    }

    void CommandBuffer::BindIndexBuffer( Ref<Buffer> buffer, VkDeviceSize offset, VkIndexType indexType )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "BindIndexBuffer requires a GRAPHICS queue!" );
        m_api->vkCmdBindIndexBuffer( m_commandBuffer, buffer->GetHandle(), offset, indexType );
    }

    void CommandBuffer::Dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ )
    {
        DT_CORE_ASSERT( m_type == QueueType::COMPUTE || m_type == QueueType::GRAPHICS, "Dispatch requires COMPUTE or GRAPHICS queue!" );
        m_api->vkCmdDispatch( m_commandBuffer, groupCountX, groupCountY, groupCountZ );
    }

    void CommandBuffer::Draw( uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "Draw requires a GRAPHICS queue!" );
        m_api->vkCmdDraw( m_commandBuffer, vertexCount, instanceCount, firstVertex, firstInstance );
    }

    void CommandBuffer::DrawIndexed( uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset, uint32_t firstInstance )
    {
        DT_CORE_ASSERT( m_type == QueueType::GRAPHICS, "DrawIndexed requires a GRAPHICS queue!" );
        m_api->vkCmdDrawIndexed( m_commandBuffer, indexCount, instanceCount, firstIndex, vertexOffset, firstInstance );
    }

    void CommandBuffer::PipelineBarrier( VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags dependencyFlags,
                                         const std::vector<VkMemoryBarrier>& memoryBarriers, const std::vector<VkBufferMemoryBarrier>& bufferBarriers,
                                         const std::vector<VkImageMemoryBarrier>& imageBarriers )
    {
        // Pipeline barriers are generally valid on all queues (Transfer, Compute, Graphics),
        // though specific stages might be restricted (e.g., COLOR_ATTACHMENT_OUTPUT on Compute queue is invalid).
        // For simplicity, we don't assert stages here, relying on validation layers.

        m_api->vkCmdPipelineBarrier( m_commandBuffer, srcStage, dstStage, dependencyFlags, static_cast<uint32_t>( memoryBarriers.size() ),
                                     memoryBarriers.data(), static_cast<uint32_t>( bufferBarriers.size() ), bufferBarriers.data(),
                                     static_cast<uint32_t>( imageBarriers.size() ), imageBarriers.data() );
    }

    void CommandBuffer::TransitionImageLayout( VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags aspectMask )
    {
        VkImageMemoryBarrier barrier            = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        barrier.oldLayout                       = oldLayout;
        barrier.newLayout                       = newLayout;
        barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
        barrier.image                           = image;
        barrier.subresourceRange.aspectMask     = aspectMask;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;

        VkPipelineStageFlags sourceStage      = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VkPipelineStageFlags destinationStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;

        // Optimized barriers for common cases
        if( oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL )
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            sourceStage           = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if( oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL )
        {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            sourceStage           = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage      = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        }
        // General fallback
        else
        {
            barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        }

        m_api->vkCmdPipelineBarrier( m_commandBuffer, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier );
    }
} // namespace DigitalTwin