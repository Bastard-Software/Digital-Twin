#include "rhi/CommandBuffer.h"

#include "core/Log.h"
#include "rhi/Buffer.h"
#include "rhi/Texture.h"

// Macros to strip validation checks in Release builds
#ifdef DT_DEBUG
#    define DT_CMD_CHECK_TYPE( req, name )                                                                                                           \
        if( !ValidateType( req, name ) )                                                                                                             \
        return
#    define DT_CMD_CHECK_STATE( req, name )                                                                                                          \
        if( !ValidateState( req, name ) )                                                                                                            \
        return Result::FAIL
#    define DT_CMD_SET_STATE( s ) m_state = s
#else
#    define DT_CMD_CHECK_TYPE( req, name )
#    define DT_CMD_CHECK_STATE( req, name )
#    define DT_CMD_SET_STATE( s )
#endif

namespace DigitalTwin
{
    void CommandBuffer::Initialize( VkCommandBuffer handle, QueueType type, VkDevice device, const VolkDeviceTable* api )
    {
        m_handle = handle;
        m_type   = type;
        m_device = device;
        m_api    = api;
        DT_CMD_SET_STATE( State::Initial );
    }

    Result CommandBuffer::Begin()
    {
#ifdef DT_DEBUG
        if( m_state == State::Recording )
        {
            DT_WARN( "CommandBuffer::Begin called on already recording buffer." );
            return Result::FAIL;
        }
#endif

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if( m_api->vkBeginCommandBuffer( m_handle, &beginInfo ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to begin command buffer recording." );
            return Result::FAIL;
        }

        DT_CMD_SET_STATE( State::Recording );
        return Result::SUCCESS;
    }

    Result CommandBuffer::End()
    {
        DT_CMD_CHECK_STATE( State::Recording, "End" );

        if( m_api->vkEndCommandBuffer( m_handle ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to end command buffer recording." );
            return Result::FAIL;
        }

        DT_CMD_SET_STATE( State::Executable );
        return Result::SUCCESS;
    }

    void CommandBuffer::Reset()
    {
        // Usually handled by pool reset, but good for explicit reuse
        m_api->vkResetCommandBuffer( m_handle, 0 );
        DT_CMD_SET_STATE( State::Initial );
    }

    void CommandBuffer::BeginRendering( const VkRenderingInfo& renderingInfo )
    {
        DT_CMD_CHECK_TYPE( QueueType::GRAPHICS, "BeginRendering" );
        m_api->vkCmdBeginRendering( m_handle, &renderingInfo );
    }

    void CommandBuffer::EndRendering()
    {
        DT_CMD_CHECK_TYPE( QueueType::GRAPHICS, "EndRendering" );
        m_api->vkCmdEndRendering( m_handle );
    }

    void CommandBuffer::SetViewport( float x, float y, float width, float height, float minDepth, float maxDepth )
    {
        DT_CMD_CHECK_TYPE( QueueType::GRAPHICS, "SetViewport" );
        VkViewport vp{ x, y, width, height, minDepth, maxDepth };
        m_api->vkCmdSetViewport( m_handle, 0, 1, &vp );
    }

    void CommandBuffer::SetScissor( int32_t x, int32_t y, uint32_t width, uint32_t height )
    {
        DT_CMD_CHECK_TYPE( QueueType::GRAPHICS, "SetScissor" );
        VkRect2D scissor{ { x, y }, { width, height } };
        m_api->vkCmdSetScissor( m_handle, 0, 1, &scissor );
    }

    void CommandBuffer::Draw( uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance )
    {
        DT_CMD_CHECK_TYPE( QueueType::GRAPHICS, "Draw" );
        m_api->vkCmdDraw( m_handle, vertexCount, instanceCount, firstVertex, firstInstance );
    }

    void CommandBuffer::Dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ )
    {
#ifdef DT_DEBUG
        if( m_type == QueueType::TRANSFER )
        {
            DT_ERROR( "Dispatch disallowed on Transfer Queue" );
            return;
        }
#endif
        m_api->vkCmdDispatch( m_handle, groupCountX, groupCountY, groupCountZ );
    }

    void CommandBuffer::CopyBuffer( Buffer* src, Buffer* dst, VkDeviceSize size, VkDeviceSize srcOffset, VkDeviceSize dstOffset )
    {
        VkBufferCopy region{};
        region.srcOffset = srcOffset;
        region.dstOffset = dstOffset;
        region.size      = size;
        m_api->vkCmdCopyBuffer( m_handle, src->GetHandle(), dst->GetHandle(), 1, &region );
    }

    void CommandBuffer::PipelineBarrier( VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage, VkDependencyFlags dependencyFlags,
                                         uint32_t memoryBarrierCount, const VkMemoryBarrier2* pMemoryBarriers, uint32_t bufferMemoryBarrierCount,
                                         const VkBufferMemoryBarrier2* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
                                         const VkImageMemoryBarrier2* pImageMemoryBarriers )
    {
        VkDependencyInfo info         = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        info.dependencyFlags          = dependencyFlags;
        info.memoryBarrierCount       = memoryBarrierCount;
        info.pMemoryBarriers          = pMemoryBarriers;
        info.bufferMemoryBarrierCount = bufferMemoryBarrierCount;
        info.pBufferMemoryBarriers    = pBufferMemoryBarriers;
        info.imageMemoryBarrierCount  = imageMemoryBarrierCount;
        info.pImageMemoryBarriers     = pImageMemoryBarriers;

        m_api->vkCmdPipelineBarrier2( m_handle, &info );
    }

    void CommandBuffer::ClearColorImage( Texture* texture, VkImageLayout layout, const VkClearColorValue& color )
    {
#ifdef DT_DEBUG
        if( m_type == QueueType::TRANSFER )
        {
            DT_ERROR( "ClearColorImage disallowed on Transfer Queue" );
            return;
        }
#endif
        VkImageSubresourceRange range{};
        range.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        range.baseMipLevel   = 0;
        range.levelCount     = 1;
        range.baseArrayLayer = 0;
        range.layerCount     = 1;

        m_api->vkCmdClearColorImage( m_handle, texture->GetHandle(), layout, &color, 1, &range );
    }

    bool CommandBuffer::ValidateType( QueueType requiredType, const char* operationName ) const
    {
        if( m_type != requiredType )
        {
            if( requiredType == QueueType::GRAPHICS && m_type != QueueType::GRAPHICS )
            {
                DT_ERROR( "CommandBuffer: Invalid operation '{}'. Requires GRAPHICS queue.", operationName );
                return false;
            }
        }
        return true;
    }

    bool CommandBuffer::ValidateState( State requiredState, const char* operationName ) const
    {
        if( m_state != requiredState )
        {
            DT_ERROR( "CommandBuffer: Invalid state for '{}'.", operationName );
            return false;
        }
        return true;
    }
} // namespace DigitalTwin