#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <volk.h>

namespace DigitalTwin
{
    class Buffer;
    class Texture;

    class CommandBuffer
    {
    public:
        // Debug-only state tracking
        enum class State : uint8_t
        {
            Initial,
            Recording,
            Executable,
            Pending,
            Invalid
        };

        CommandBuffer()  = default;
        ~CommandBuffer() = default;

        // Light initialization instead of constructor
        void Initialize( VkCommandBuffer handle, QueueType type, VkDevice device, const VolkDeviceTable* api );

        // --- Lifetime ---
        Result Begin();
        Result End();
        void   Reset();

        // --- Commands ---

        void SetPipeline( ComputePipeline* pipeline );
        void SetPipeline( GraphicsPipeline* pipeline );
        void SetBindingGroup( BindingGroup* group, VkPipelineLayout layout, VkPipelineBindPoint bindPoint );
        void SetIndexBuffer( Buffer* buffer, VkDeviceSize offset = 0, VkIndexType indexType = VK_INDEX_TYPE_UINT32 );

        void PushConstants( VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, uint32_t size, const void* pValues );

        void BeginRendering( const VkRenderingInfo& renderingInfo );
        void EndRendering();

        void SetViewport( float x, float y, float width, float height, float minDepth = 0.0f, float maxDepth = 1.0f );
        void SetScissor( int32_t x, int32_t y, uint32_t width, uint32_t height );

        void Draw( uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance );
        void DrawIndexedIndirect( Buffer* indirectBuffer, VkDeviceSize offset, uint32_t drawCount, uint32_t stride );
        void DrawIndexedIndirectCount( Buffer* indirectBuffer, VkDeviceSize offset, Buffer* countBuffer, VkDeviceSize countBufferOffset,
                                       uint32_t maxDrawCount, uint32_t stride );

        void Dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ );

        void CopyBuffer( Buffer* src, Buffer* dst, VkDeviceSize size, VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0 );

        void PipelineBarrier( VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage, VkDependencyFlags dependencyFlags,
                              uint32_t memoryBarrierCount, const VkMemoryBarrier2* pMemoryBarriers, uint32_t bufferMemoryBarrierCount,
                              const VkBufferMemoryBarrier2* pBufferMemoryBarriers, uint32_t imageMemoryBarrierCount,
                              const VkImageMemoryBarrier2* pImageMemoryBarriers );
        void PipelineBarrier( const VkDependencyInfo* pDependencyInfo );

        void ClearColorImage( Texture* texture, VkImageLayout layout, const VkClearColorValue& color );

        void ResetQueryPool( VkQueryPool queryPool, uint32_t firstQuery, uint32_t queryCount );
        void WriteTimestamp( VkPipelineStageFlagBits stage, VkQueryPool queryPool, uint32_t query );
        void BeginQuery( VkQueryPool queryPool, uint32_t query, VkQueryControlFlags flags = 0 );
        void EndQuery( VkQueryPool queryPool, uint32_t query );

        // --- Getters ---
        VkCommandBuffer GetHandle() const { return m_handle; }
        QueueType       GetType() const { return m_type; }
        State           GetState() const { return m_state; }

    private:
        // Validation helpers (compiled out in Release)
        bool ValidateType( QueueType requiredType, const char* operationName ) const;
        bool ValidateState( State requiredState, const char* operationName ) const;

    private:
        VkCommandBuffer        m_handle = VK_NULL_HANDLE;
        QueueType              m_type   = QueueType::GRAPHICS;
        VkDevice               m_device = VK_NULL_HANDLE;
        const VolkDeviceTable* m_api    = nullptr;
        State                  m_state  = State::Initial;
    };
} // namespace DigitalTwin