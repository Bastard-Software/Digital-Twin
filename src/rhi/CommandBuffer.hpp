#pragma once
#include "core/Base.hpp"
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    class CommandBuffer
    {
    public:
        CommandBuffer( VkDevice device, const VolkDeviceTable* api, VkCommandPool pool, VkCommandBuffer buffer );
        ~CommandBuffer();

        CommandBuffer( const CommandBuffer& )                = delete;
        CommandBuffer& operator=( const CommandBuffer& )     = delete;
        CommandBuffer( CommandBuffer&& ) noexcept            = default;
        CommandBuffer& operator=( CommandBuffer&& ) noexcept = default;

        // --- Lifecycle ---
        void Begin( VkCommandBufferUsageFlags flags = 0 );
        void End();

        // --- Compute & Dispatch ---
        void BindComputePipeline( VkPipeline pipeline );
        void Dispatch( uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ );

        // --- Resources ---
        void BindDescriptorSets( VkPipelineLayout layout, uint32_t firstSet, const std::vector<VkDescriptorSet>& sets );

        // --- Synchronization ---
        void PipelineBarrier( VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags dependencyFlags,
                              const std::vector<VkMemoryBarrier>& memoryBarriers, const std::vector<VkBufferMemoryBarrier>& bufferBarriers,
                              const std::vector<VkImageMemoryBarrier>& imageBarriers );

        // --- Getters ---
        VkCommandBuffer GetHandle() const { return m_commandBuffer; }

    private:
        VkDevice        m_device;
        const VolkDeviceTable* m_api;
        VkCommandPool   m_pool;
        VkCommandBuffer m_commandBuffer;
    };
} // namespace DigitalTwin