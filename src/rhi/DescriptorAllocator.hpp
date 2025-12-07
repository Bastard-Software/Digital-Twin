#pragma once
#include "core/Base.hpp"
#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    class DescriptorAllocator
    {
    public:
        // Constructor initializes the allocator with device handle and API table
        DescriptorAllocator( VkDevice device, const VolkDeviceTable* api );
        ~DescriptorAllocator();

        // Disable copying
        DescriptorAllocator( const DescriptorAllocator& )            = delete;
        DescriptorAllocator& operator=( const DescriptorAllocator& ) = delete;

        // Cleans up all created pools. Called in destructor.
        void Shutdown();

        /**
         * @brief Allocates a descriptor set from the current pool.
         * If the pool is full, it gets a new one from the reserve or creates it.
         * * @param layout The layout of the descriptor set to allocate.
         * @param outSet [Out] Handle to the allocated descriptor set.
         * @return Result::SUCCESS or Result::OUT_OF_MEMORY
         */
        Result Allocate( VkDescriptorSetLayout layout, VkDescriptorSet& outSet );

        /**
         * @brief Resets all used pools and moves them to the free list.
         * Does NOT destroy the underlying Vulkan pools, just resets them for reuse.
         * Call this at the start of a frame (if using frame-based allocators).
         */
        void ResetPools();

    private:
        // Helper to pick a valid pool (from free list or create new)
        VkDescriptorPool GrabPool();

        // Internal helper to create a fresh Vulkan pool
        VkDescriptorPool CreatePool( uint32_t count, VkDescriptorPoolCreateFlags flags );

    private:
        VkDevice               m_device;
        const VolkDeviceTable* m_api;

        VkDescriptorPool              m_currentPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorPool> m_usedPools;
        std::vector<VkDescriptorPool> m_freePools;
    };
} // namespace DigitalTwin