#include "rhi/DescriptorAllocator.hpp"

#include <array>

namespace DigitalTwin
{
    // Configuration for pool sizes.
    // When a new pool is created, it will hold this many descriptors of each type.
    static constexpr uint32_t MAX_SETS_PER_POOL = 1000;

    static const std::vector<VkDescriptorPoolSize> POOL_SIZES = { { VK_DESCRIPTOR_TYPE_SAMPLER, 500 },
                                                                  { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4000 },
                                                                  { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4000 },
                                                                  { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                                                                  { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                                                                  { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                                                                  { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2000 },
                                                                  { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2000 },
                                                                  { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                                                                  { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                                                                  { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 500 } };

    DescriptorAllocator::DescriptorAllocator( VkDevice device, const VolkDeviceTable* api )
        : m_device( device )
        , m_api( api )
    {
        DT_CORE_ASSERT( m_api, "VolkDeviceTable is null!" );
    }

    DescriptorAllocator::~DescriptorAllocator()
    {
        Shutdown();
    }

    void DescriptorAllocator::Shutdown()
    {
        // Destroy all pools in both lists
        for( auto p: m_freePools )
        {
            m_api->vkDestroyDescriptorPool( m_device, p, nullptr );
        }
        for( auto p: m_usedPools )
        {
            m_api->vkDestroyDescriptorPool( m_device, p, nullptr );
        }

        m_freePools.clear();
        m_usedPools.clear();
        m_currentPool = VK_NULL_HANDLE;
    }

    Result DescriptorAllocator::Allocate( VkDescriptorSetLayout layout, VkDescriptorSet& outSet )
    {
        // Initialize current pool if needed
        if( m_currentPool == VK_NULL_HANDLE )
        {
            m_currentPool = GrabPool();
            m_usedPools.push_back( m_currentPool );
        }

        VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.pNext                       = nullptr;
        allocInfo.descriptorPool              = m_currentPool;
        allocInfo.descriptorSetCount          = 1;
        allocInfo.pSetLayouts                 = &layout;

        // Try to allocate from the current pool
        VkResult result = m_api->vkAllocateDescriptorSets( m_device, &allocInfo, &outSet );

        // Handle allocation failure (Pool Full or Fragmentation)
        if( result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTATION )
        {
            // Current pool is full, grab a new one
            m_currentPool = GrabPool();
            m_usedPools.push_back( m_currentPool );

            // Retry allocation with the new pool
            allocInfo.descriptorPool = m_currentPool;
            result                   = m_api->vkAllocateDescriptorSets( m_device, &allocInfo, &outSet );
        }

        if( result != VK_SUCCESS )
        {
            DT_CORE_ERROR( "Failed to allocate descriptor set! Error: {}", ( int )result );
            return Result::OUT_OF_MEMORY; // Or map specific Vulkan errors to generic Result
        }

        return Result::SUCCESS;
    }

    void DescriptorAllocator::ResetPools()
    {
        // Reset all used pools and move them to the free list
        for( auto p: m_usedPools )
        {
            m_api->vkResetDescriptorPool( m_device, p, 0 );
            m_freePools.push_back( p );
        }

        m_usedPools.clear();
        m_currentPool = VK_NULL_HANDLE;
    }

    VkDescriptorPool DescriptorAllocator::GrabPool()
    {
        // Reuse an existing pool if available
        if( !m_freePools.empty() )
        {
            VkDescriptorPool pool = m_freePools.back();
            m_freePools.pop_back();
            return pool;
        }

        // Otherwise create a new one
        return CreatePool( MAX_SETS_PER_POOL, 0 );
    }

    VkDescriptorPool DescriptorAllocator::CreatePool( uint32_t count, VkDescriptorPoolCreateFlags flags )
    {
        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.flags                      = flags;
        poolInfo.maxSets                    = count;
        poolInfo.poolSizeCount              = static_cast<uint32_t>( POOL_SIZES.size() );
        poolInfo.pPoolSizes                 = POOL_SIZES.data();

        VkDescriptorPool pool;
        if( m_api->vkCreateDescriptorPool( m_device, &poolInfo, nullptr, &pool ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create descriptor pool!" );
            return VK_NULL_HANDLE;
        }

        return pool;
    }
} // namespace DigitalTwin