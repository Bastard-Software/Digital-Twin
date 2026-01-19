#include "rhi/DescriptorAllocator.h"

#include "core/Log.h"

namespace DigitalTwin
{
    DescriptorAllocator::DescriptorAllocator( VkDevice device, const VolkDeviceTable* api )
        : m_device( device )
        , m_api( api )
    {
    }

    DescriptorAllocator::~DescriptorAllocator()
    {
    }

    void DescriptorAllocator::Initialize()
    {
        // Define default ratios for a general-purpose engine.
        // These ratios determine how many descriptors of each type are in a single pool.
        m_ratios = { { VK_DESCRIPTOR_TYPE_SAMPLER, 0.5f },
                     { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4.0f },
                     { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4.0f },
                     { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1.0f },
                     { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1.0f },
                     { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1.0f },
                     { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2.0f },
                     { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2.0f },
                     { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1.0f },
                     { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1.0f },
                     { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 0.5f } };
    }

    void DescriptorAllocator::Shutdown()
    {
        // Destroy all pools created by this allocator using m_api
        for( auto p: m_freePools )
        {
            m_api->vkDestroyDescriptorPool( m_device, p, nullptr );
        }
        for( auto p: m_usedPools )
        {
            m_api->vkDestroyDescriptorPool( m_device, p, nullptr );
        }
        if( m_currentPool != VK_NULL_HANDLE )
        {
            m_api->vkDestroyDescriptorPool( m_device, m_currentPool, nullptr );
        }

        m_freePools.clear();
        m_usedPools.clear();
        m_currentPool = VK_NULL_HANDLE;
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

        // Also reset the current pool and mark it as free, so we start fresh next frame
        if( m_currentPool != VK_NULL_HANDLE )
        {
            m_api->vkResetDescriptorPool( m_device, m_currentPool, 0 );
            m_freePools.push_back( m_currentPool );
            m_currentPool = VK_NULL_HANDLE;
        }
    }

    Result DescriptorAllocator::Allocate( VkDescriptorSetLayout layout, VkDescriptorSet& outSet )
    {
        // Initialize current pool if needed
        if( m_currentPool == VK_NULL_HANDLE )
        {
            m_currentPool = GrabPool();
            if( m_currentPool == VK_NULL_HANDLE )
                return Result::FAIL;
        }

        VkDescriptorSetAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.pNext                       = nullptr;
        allocInfo.descriptorPool              = m_currentPool;
        allocInfo.descriptorSetCount          = 1;
        allocInfo.pSetLayouts                 = &layout;

        VkResult result = m_api->vkAllocateDescriptorSets( m_device, &allocInfo, &outSet );

        // If the pool is full or fragmented, try to get a new one
        if( result == VK_ERROR_OUT_OF_POOL_MEMORY || result == VK_ERROR_FRAGMENTATION )
        {
            // Move the full pool to used list
            m_usedPools.push_back( m_currentPool );

            // Grab a new pool
            m_currentPool = GrabPool();
            if( m_currentPool == VK_NULL_HANDLE )
                return Result::FAIL;

            allocInfo.descriptorPool = m_currentPool;

            // Retry allocation
            result = m_api->vkAllocateDescriptorSets( m_device, &allocInfo, &outSet );
        }

        if( result != VK_SUCCESS )
        {
            DT_ERROR( "Failed to allocate descriptor set! Vulkan Error: {}", ( int )result );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    VkDescriptorPool DescriptorAllocator::GrabPool()
    {
        // Reuse an existing free pool if available
        if( !m_freePools.empty() )
        {
            VkDescriptorPool pool = m_freePools.back();
            m_freePools.pop_back();
            return pool;
        }
        else
        {
            // Create a new pool with base size of 1000 (scaled by ratios)
            return CreatePool( 1000, 0 );
        }
    }

    VkDescriptorPool DescriptorAllocator::CreatePool( uint32_t count, VkDescriptorPoolCreateFlags flags )
    {
        std::vector<VkDescriptorPoolSize> sizes;
        sizes.reserve( m_ratios.size() );

        for( const auto& ratio: m_ratios )
        {
            sizes.push_back( { ratio.type, static_cast<uint32_t>( ratio.ratio * count ) } );
        }

        VkDescriptorPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        poolInfo.flags                      = flags;
        poolInfo.maxSets                    = count;
        poolInfo.poolSizeCount              = static_cast<uint32_t>( sizes.size() );
        poolInfo.pPoolSizes                 = sizes.data();

        VkDescriptorPool newPool;
        if( m_api->vkCreateDescriptorPool( m_device, &poolInfo, nullptr, &newPool ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create descriptor pool!" );
            return VK_NULL_HANDLE;
        }

        return newPool;
    }

} // namespace DigitalTwin