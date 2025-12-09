#pragma once
#include "core/Base.hpp"
#include "rhi/Queue.hpp" // Need QueueType enum
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    class Pipeline;
    class ComputePipeline;
    class GraphicsPipeline;
    class Buffer;
    class Texture;

    struct RenderingAttachmentInfo
    {
        VkImageView         imageView;
        VkAttachmentLoadOp  loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        VkAttachmentStoreOp storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
        VkClearValue        clearValue = { 0.0f, 0.0f, 0.0f, 1.0f };
        VkImageLayout       layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    };

    struct RenderingInfo
    {
        VkRect2D                             renderArea;
        std::vector<RenderingAttachmentInfo> colorAttachments;
        bool                                 useDepth = false;
        RenderingAttachmentInfo              depthAttachment;
    };

    class CommandBuffer
    {
    public:
        // Constructor now takes QueueType to validate capabilities
        CommandBuffer( VkDevice device, const VolkDeviceTable* api, VkCommandPool pool, VkCommandBuffer buffer, QueueType type );
        ~CommandBuffer();

        // --- Lifecycle ---
        void Begin( VkCommandBufferUsageFlags flags = 0 );
        void End();

        // --- Dynamic Rendering (Requires Graphics Queue) ---
        void BeginRendering( const RenderingInfo& info );
        void EndRendering();

        // --- State Setup (Requires Graphics Queue) ---
        void SetViewport( float x, float y, float width, float height, float minDepth = 0.0f, float maxDepth = 1.0f );
        void SetScissor( int32_t x, int32_t y, uint32_t width, uint32_t height );

        // --- Pipelines & Binding ---
        void BindComputePipeline( Ref<ComputePipeline> pipeline );   // Compute or Graphics Queue
        void BindGraphicsPipeline( Ref<GraphicsPipeline> pipeline ); // Graphics Queue Only

        void BindDescriptorSets( VkPipelineBindPoint bindPoint, VkPipelineLayout layout, uint32_t firstSet,
                                 const std::vector<VkDescriptorSet>& sets );

        void BindIndexBuffer( Ref<Buffer> buffer, VkDeviceSize offset, VkIndexType indexType );

        // --- Push Constants ---
        template<typename T>
        void PushConstants( VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t offset, const T& data )
        {
            m_api->vkCmdPushConstants( m_commandBuffer, layout, stageFlags, offset, sizeof( T ), &data );
        }

        // --- Dispatch & Draw ---
        void Dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ ); // Compute/Graphics

        void Draw( uint32_t vertexCount, uint32_t instanceCount, uint32_t firstVertex, uint32_t firstInstance ); // Graphics Only
        void DrawIndexed( uint32_t indexCount, uint32_t instanceCount, uint32_t firstIndex, int32_t vertexOffset,
                          uint32_t firstInstance ); // Graphics Only

        // --- Synchronization ---
        void PipelineBarrier( VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags dependencyFlags,
                              const std::vector<VkMemoryBarrier>& memoryBarriers, const std::vector<VkBufferMemoryBarrier>& bufferBarriers,
                              const std::vector<VkImageMemoryBarrier>& imageBarriers );

        void TransitionImageLayout( VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                                    VkImageAspectFlags aspectMask = VK_IMAGE_ASPECT_COLOR_BIT );

        VkCommandBuffer GetHandle() const { return m_commandBuffer; }
        QueueType       GetType() const { return m_type; }

    public:
        // --- Disable Copying (RAII) ---
        CommandBuffer( const CommandBuffer& )            = delete;
        CommandBuffer& operator=( const CommandBuffer& ) = delete;

        // Allow Moving
        CommandBuffer( CommandBuffer&& )            = default;
        CommandBuffer& operator=( CommandBuffer&& ) = default;

    private:
        VkDevice               m_device;
        const VolkDeviceTable* m_api;
        VkCommandPool          m_pool;
        VkCommandBuffer        m_commandBuffer;
        QueueType              m_type;
    };
} // namespace DigitalTwin