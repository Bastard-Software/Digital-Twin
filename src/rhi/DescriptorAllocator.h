#pragma once
#include "rhi/RHITypes.h"

#include <vector>
#include <volk.h>

namespace DigitalTwin
{
    /**
     * @brief Manages dynamic allocation of Vulkan Descriptor Sets.
     * Automatically creates new Descriptor Pools as needed.
     * Uses internal Volk device table for API calls.
     */
    class DescriptorAllocator
    {
    public:
        struct PoolSizeRatio
        {
            VkDescriptorType type;
            float            ratio; // Multiplier for the pool size
        };

        /**
         * @brief Constructor requiring device and API table.
         * @param device The logical Vulkan device.
         * @param api The function table for this device (Volk).
         */
        DescriptorAllocator( VkDevice device, const VolkDeviceTable* api );
        ~DescriptorAllocator();

        /**
         * @brief Initializes default pool ratios.
         */
        void Initialize();

        /**
         * @brief Destroys all managed pools.
         */
        void Shutdown();

        /**
         * @brief Resets all pools, making all allocated sets invalid.
         * Should be called at the beginning of a frame.
         * Does NOT destroy the internal Vulkan pools, just resets them for reuse.
         */
        void ResetPools();

        /**
         * @brief Allocates a descriptor set from the current pool.
         * If the pool is full, a new pool is requested or created.
         * @param layout The layout to allocate.
         * @param outSet Pointer to store the resulting set.
         * @return Result::SUCCESS if allocation succeeded.
         */
        Result Allocate( VkDescriptorSetLayout layout, VkDescriptorSet& outSet );

    private:
        VkDescriptorPool GrabPool();
        VkDescriptorPool CreatePool( uint32_t count, VkDescriptorPoolCreateFlags flags );

    private:
        VkDevice               m_device = VK_NULL_HANDLE;
        const VolkDeviceTable* m_api    = nullptr;

        VkDescriptorPool              m_currentPool = VK_NULL_HANDLE;
        std::vector<VkDescriptorPool> m_usedPools;
        std::vector<VkDescriptorPool> m_freePools;

        // Configuration for pool sizes
        std::vector<PoolSizeRatio> m_ratios;
    };
} // namespace DigitalTwin