#include "rhi/Device.hpp"

#include "rhi/RHI.hpp"
#include <set>
#include <sstream>
#include <vector>

namespace DigitalTwin
{
    struct Device::QueueFamilyIndices
    {
        int32_t graphics = -1;
        int32_t compute  = -1;
        int32_t transfer = -1;

        // Helper check to ensure we have at least a graphics/universal queue
        bool IsValid() const { return graphics != -1; }
    };

    Device::Device( VkPhysicalDevice physicalDevice )
        : m_physicalDevice( physicalDevice )
    {
    }

    Device::~Device()
    {
        Shutdown();
    }

    Result Device::Init( DeviceDesc desc )
    {
        m_desc = desc;

        // 1. Find queue family indices (prioritizing Async Compute)
        QueueFamilyIndices indices = FindQueueFamilies( m_physicalDevice );
        if( !indices.IsValid() )
        {
            DT_CORE_CRITICAL( "Failed to find suitable queue families!" );
            return Result::FAIL;
        }

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        // Use a set to avoid duplicates if indices are the same
        std::set<int32_t> uniqueQueueFamilies = { indices.graphics, indices.compute, indices.transfer };
        // Remove -1 if present (fallback logic should eliminate this, but for safety)
        uniqueQueueFamilies.erase( -1 );

        float queuePriority = 1.0f;
        for( int32_t queueFamily: uniqueQueueFamilies )
        {
            VkDeviceQueueCreateInfo queueCreateInfo = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
            queueCreateInfo.queueFamilyIndex        = queueFamily;
            queueCreateInfo.queueCount              = 1;
            queueCreateInfo.pQueuePriorities        = &queuePriority;
            queueCreateInfos.push_back( queueCreateInfo );
        }

        VkPhysicalDeviceFeatures deviceFeatures = {};
        // Here we can enable features like shaderInt64 for compute if needed

        VkPhysicalDeviceVulkan12Features features12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        features12.timelineSemaphore                = VK_TRUE;

        VkPhysicalDeviceVulkan13Features features13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        features13.pNext                            = &features12;
        features13.synchronization2                 = VK_TRUE;
        features13.dynamicRendering                 = VK_TRUE;

        VkPhysicalDeviceFeatures2 deviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        deviceFeatures2.features                  = deviceFeatures;
        deviceFeatures2.pNext                     = &features13;

        std::vector<const char*> extensions;
        if( !m_desc.headless )
        {
            extensions.push_back( VK_KHR_SWAPCHAIN_EXTENSION_NAME );
        }

        VkDeviceCreateInfo createInfo      = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        createInfo.pNext                   = &deviceFeatures2;
        createInfo.queueCreateInfoCount    = static_cast<uint32_t>( queueCreateInfos.size() );
        createInfo.pQueueCreateInfos       = queueCreateInfos.data();
        createInfo.pEnabledFeatures        = nullptr;
        createInfo.enabledExtensionCount   = static_cast<uint32_t>( extensions.size() );
        createInfo.ppEnabledExtensionNames = extensions.data();

        if( vkCreateDevice( m_physicalDevice, &createInfo, nullptr, &m_device ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create logical device!" );
            return Result::FAIL;
        }

        volkLoadDeviceTable( &m_api, m_device );

        // Create queues with aliasing logic (sharing)

        // We always create Graphics (assuming Graphics is the "main" universal queue, even in headless)
        // If headless and no graphics bit, FindQueueFamilies should return compute index as graphics
        m_graphicsQueue = CreateRef<Queue>( m_device, m_api, indices.graphics, QueueType::GRAPHICS );

        // Compute
        if( indices.compute == indices.graphics )
        {
            m_computeQueue = m_graphicsQueue; // Alias: Point to the same object
            DT_CORE_INFO( "Compute Queue aliased to Graphics Queue (Family {})", indices.compute );
        }
        else
        {
            m_computeQueue = CreateRef<Queue>( m_device, m_api, indices.compute, QueueType::COMPUTE );
        }

        // Transfer
        if( indices.transfer == indices.graphics )
        {
            m_transferQueue = m_graphicsQueue; // Alias
        }
        else if( indices.transfer == indices.compute )
        {
            m_transferQueue = m_computeQueue; // Alias to Compute
        }
        else
        {
            m_transferQueue = CreateRef<Queue>( m_device, m_api, indices.transfer, QueueType::TRANSFER );
        }

        // --- VMA Init ---
        VmaVulkanFunctions vmaVulkanFunctions                      = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr                   = vkGetInstanceProcAddr;
        vmaVulkanFunctions.vkGetDeviceProcAddr                     = vkGetDeviceProcAddr;
        vmaVulkanFunctions.vkGetPhysicalDeviceProperties           = vkGetPhysicalDeviceProperties;
        vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties     = vkGetPhysicalDeviceMemoryProperties;
        vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;

        vmaVulkanFunctions.vkAllocateMemory               = m_api.vkAllocateMemory;
        vmaVulkanFunctions.vkFreeMemory                   = m_api.vkFreeMemory;
        vmaVulkanFunctions.vkMapMemory                    = m_api.vkMapMemory;
        vmaVulkanFunctions.vkUnmapMemory                  = m_api.vkUnmapMemory;
        vmaVulkanFunctions.vkFlushMappedMemoryRanges      = m_api.vkFlushMappedMemoryRanges;
        vmaVulkanFunctions.vkInvalidateMappedMemoryRanges = m_api.vkInvalidateMappedMemoryRanges;

        vmaVulkanFunctions.vkBindBufferMemory            = m_api.vkBindBufferMemory;
        vmaVulkanFunctions.vkBindImageMemory             = m_api.vkBindImageMemory;
        vmaVulkanFunctions.vkGetBufferMemoryRequirements = m_api.vkGetBufferMemoryRequirements;
        vmaVulkanFunctions.vkGetImageMemoryRequirements  = m_api.vkGetImageMemoryRequirements;

        vmaVulkanFunctions.vkCreateBuffer  = m_api.vkCreateBuffer;
        vmaVulkanFunctions.vkDestroyBuffer = m_api.vkDestroyBuffer;
        vmaVulkanFunctions.vkCreateImage   = m_api.vkCreateImage;
        vmaVulkanFunctions.vkDestroyImage  = m_api.vkDestroyImage;
        vmaVulkanFunctions.vkCmdCopyBuffer = m_api.vkCmdCopyBuffer;

        vmaVulkanFunctions.vkGetBufferMemoryRequirements2KHR = m_api.vkGetBufferMemoryRequirements2;
        vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR  = m_api.vkGetImageMemoryRequirements2;
        vmaVulkanFunctions.vkBindBufferMemory2KHR            = m_api.vkBindBufferMemory2;
        vmaVulkanFunctions.vkBindImageMemory2KHR             = m_api.vkBindImageMemory2;

        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice         = m_physicalDevice;
        allocatorInfo.device                 = m_device;
        allocatorInfo.instance               = RHI::GetInstance();
        allocatorInfo.vulkanApiVersion       = VK_API_VERSION_1_4; // With api version 1.4, VMA needs some newer functions
        allocatorInfo.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        allocatorInfo.pVulkanFunctions       = &vmaVulkanFunctions;

        if( vmaCreateAllocator( &allocatorInfo, &m_allocator ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create VMA allocator!" );
            return Result::FAIL;
        }

        m_descriptorAllocator = CreateRef<DescriptorAllocator>( m_device, &m_api );

        DT_CORE_INFO( "Logical Device initialized. Queues indices -> G:{0} C:{1} T:{2}", indices.graphics, indices.compute, indices.transfer );

        return Result::SUCCESS;
    }

    void Device::Shutdown()
    {
        if( m_device == VK_NULL_HANDLE )
            return;

        // Clean up descriptor pools first
        if( m_descriptorAllocator )
        {
            m_descriptorAllocator->Shutdown();
            m_descriptorAllocator.reset();
        }

        // Clean up command pools before destroying the device
        {
            std::lock_guard<std::mutex> lock( m_poolMutex );
            m_threadPools.clear(); // Destructors of PoolInfo will call vkDestroyCommandPool using m_api
        }

        // Reset pointers in reverse order.
        // If they are aliases, shared_ptr ensures the object is destroyed only when the last ref is gone.
        m_transferQueue.reset();
        m_computeQueue.reset();
        m_graphicsQueue.reset();

        if( m_allocator != VK_NULL_HANDLE )
        {
            vmaDestroyAllocator( m_allocator );
            m_allocator = VK_NULL_HANDLE;
        }

        m_api.vkDestroyDevice( m_device, nullptr );
        m_device = VK_NULL_HANDLE;
    }

    VkCommandPool Device::GetOrCreateThreadLocalPool( uint32_t queueFamilyIndex )
    {
        std::thread::id tid = std::this_thread::get_id();

        // Lock only for map access/insertion
        std::lock_guard<std::mutex> lock( m_poolMutex );
        auto&                       familyMap = m_threadPools[ tid ];

        // Check if pool already exists for this thread and family
        auto it = familyMap.find( queueFamilyIndex );
        if( it != familyMap.end() )
        {
            return it->second.handle;
        }

        // Create a new pool
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex        = queueFamilyIndex;
        // TRANSIENT_BIT tells the driver that command buffers allocated from this pool
        // will be short-lived (reset often), which is typical for our compute-first approach.
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

        VkCommandPool pool;
        if( m_api.vkCreateCommandPool( m_device, &poolInfo, nullptr, &pool ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create thread-local command pool for thread!" );
            return VK_NULL_HANDLE;
        }

        PoolInfo info;
        info.handle = pool;
        info.device = m_device;
        info.api    = &m_api; // Store the pointer to the device table for destruction

        familyMap[ queueFamilyIndex ] = std::move( info );

        std::stringstream ss;
        ss << tid;
        DT_CORE_TRACE( "Created CommandPool for ThreadID: {} Family: {}", ss.str(), queueFamilyIndex );

        return pool;
    }

    Ref<CommandBuffer> Device::CreateCommandBuffer( QueueType type )
    {
        uint32_t familyIndex = 0;
        switch( type )
        {
            case QueueType::GRAPHICS:
                familyIndex = m_graphicsQueue->GetFamilyIndex();
                break;
            case QueueType::COMPUTE:
                familyIndex = m_computeQueue->GetFamilyIndex();
                break;
            case QueueType::TRANSFER:
                familyIndex = m_transferQueue->GetFamilyIndex();
                break;
            default:
                return nullptr;
        }

        // Get the thread-local pool for this queue family
        VkCommandPool pool = GetOrCreateThreadLocalPool( familyIndex );
        if( pool == VK_NULL_HANDLE )
            return nullptr;

        VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool                 = pool;
        allocInfo.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount          = 1;

        VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;
        if( m_api.vkAllocateCommandBuffers( m_device, &allocInfo, &cmdBuffer ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to allocate command buffer!" );
            return nullptr;
        }

        // Return a wrapper that manages the command buffer lifecycle
        return CreateRef<CommandBuffer>( m_device, &m_api, pool, cmdBuffer );
    }

    Ref<Buffer> Device::CreateBuffer( const BufferDesc& desc )
    {
        // Pass m_api to Buffer constructor
        auto buffer = CreateRef<Buffer>( m_allocator, m_device, &m_api );
        if( buffer->Create( desc ) != Result::SUCCESS )
        {
            return nullptr;
        }
        return buffer;
    }

    Ref<Texture> Device::CreateTexture( const TextureDesc& desc )
    {
        // Pass m_api to Texture constructor
        auto texture = CreateRef<Texture>( m_allocator, m_device, &m_api );
        if( texture->Create( desc ) != Result::SUCCESS )
        {
            return nullptr;
        }
        return texture;
    }

    Ref<Shader> Device::CreateShader( const std::string& filepath )
    {
        return CreateRef<Shader>( m_device, &m_api, filepath );
    }

    Ref<ComputePipeline> Device::CreateComputePipeline( const ComputePipelineDesc& desc )
    {
        return CreateRef<ComputePipeline>( m_device, &m_api, desc );
    }

    Ref<GraphicsPipeline> Device::CreateGraphicsPipeline( const GraphicsPipelineDesc& desc )
    {
        return CreateRef<GraphicsPipeline>( m_device, &m_api, desc );
    }

    Ref<Swapchain> Device::CreateSwapchain( const SwapchainDesc& desc )
    {
        // Assuming graphics queue is used for presentation
        VkQueue  presentQueue  = m_graphicsQueue->GetHandle();
        uint32_t presentFamily = m_graphicsQueue->GetFamilyIndex();

        // RHI::GetInstance() is available globally
        return CreateRef<Swapchain>( m_device, m_physicalDevice, RHI::GetInstance(), presentQueue, presentFamily, &m_api, desc );
    }

    Result Device::AllocateDescriptor( VkDescriptorSetLayout layout, VkDescriptorSet& outSet )
    {
        if( !m_descriptorAllocator )
        {
            return Result::FAIL;
        }
        return m_descriptorAllocator->Allocate( layout, outSet );
    }

    void Device::ResetDescriptorPools()
    {
        if( m_descriptorAllocator )
        {
            m_descriptorAllocator->ResetPools();
        }
    }

    Ref<Texture> Device::CreateTexture1D( uint32_t width, VkFormat format, TextureUsage usage )
    {
        auto texture = CreateRef<Texture>( m_allocator, m_device, &m_api );
        if( texture->Create1D( width, format, usage ) != Result::SUCCESS )
        {
            return nullptr;
        }
        return texture;
    }

    Ref<Texture> Device::CreateTexture2D( uint32_t width, uint32_t height, VkFormat format, TextureUsage usage )
    {
        auto texture = CreateRef<Texture>( m_allocator, m_device, &m_api );
        if( texture->Create2D( width, height, format, usage ) != Result::SUCCESS )
        {
            return nullptr;
        }
        return texture;
    }

    Ref<Texture> Device::CreateTexture3D( uint32_t width, uint32_t height, uint32_t depth, VkFormat format, TextureUsage usage )
    {
        auto texture = CreateRef<Texture>( m_allocator, m_device, &m_api );
        if( texture->Create3D( width, height, depth, format, usage ) != Result::SUCCESS )
        {
            return nullptr;
        }
        return texture;
    }

    Result Device::WaitForQueue( Ref<Queue> queue, uint64_t waitValue, uint64_t timeout )
    {
        if( !queue )
            return Result::FAIL;

        VkSemaphore timelineSem = queue->GetTimelineSemaphore();

        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.semaphoreCount      = 1;
        waitInfo.pSemaphores         = &timelineSem;
        waitInfo.pValues             = &waitValue;

        VkResult result = m_api.vkWaitSemaphores( m_device, &waitInfo, timeout );

        if( result == VK_SUCCESS )
            return Result::SUCCESS;
        if( result == VK_TIMEOUT )
            return Result::TIMEOUT;

        DT_CORE_ERROR( "WaitForQueue failed for value {}! Error: {}", waitValue, ( int )result );
        return Result::FAIL;
    }

    Device::QueueFamilyIndices Device::FindQueueFamilies( VkPhysicalDevice device )
    {
        QueueFamilyIndices indices;
        uint32_t           queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties( device, &queueFamilyCount, nullptr );

        std::vector<VkQueueFamilyProperties> queueFamilies( queueFamilyCount );
        vkGetPhysicalDeviceQueueFamilyProperties( device, &queueFamilyCount, queueFamilies.data() );

        // 1. Find Graphics (Must support GRAPHICS, compute and transfer are usually implicitly supported)
        for( int i = 0; i < ( int )queueFamilies.size(); ++i )
        {
            if( ( queueFamilies[ i ].queueFlags & VK_QUEUE_GRAPHICS_BIT ) )
            {
                indices.graphics = i;
                break; // Take the first Graphics family found
            }
        }

        // 2. Find Dedicated Compute (Compute BIT, but NO Graphics BIT)
        for( int i = 0; i < ( int )queueFamilies.size(); ++i )
        {
            const auto& flags = queueFamilies[ i ].queueFlags;
            if( ( flags & VK_QUEUE_COMPUTE_BIT ) && !( flags & VK_QUEUE_GRAPHICS_BIT ) )
            {
                indices.compute = i;
                break;
            }
        }

        // Fallback: If no dedicated compute, take any COMPUTE BIT (different from graphics if possible)
        if( indices.compute == -1 )
        {
            for( int i = 0; i < ( int )queueFamilies.size(); ++i )
            {
                if( ( queueFamilies[ i ].queueFlags & VK_QUEUE_COMPUTE_BIT ) && i != indices.graphics )
                {
                    indices.compute = i;
                    break;
                }
            }
        }

        // Final fallback: Use graphics queue
        if( indices.compute == -1 )
        {
            indices.compute = indices.graphics;
        }

        // 3. Find Dedicated Transfer (Transfer BIT, no Graphics and no Compute)
        for( int i = 0; i < ( int )queueFamilies.size(); ++i )
        {
            const auto& flags = queueFamilies[ i ].queueFlags;
            if( ( flags & VK_QUEUE_TRANSFER_BIT ) && !( flags & VK_QUEUE_GRAPHICS_BIT ) && !( flags & VK_QUEUE_COMPUTE_BIT ) )
            {
                indices.transfer = i;
                break;
            }
        }

        // Fallback: Transfer on separate compute queue (if compute is distinct)
        if( indices.transfer == -1 && indices.compute != indices.graphics )
        {
            // Check if compute supports transfer (usually yes)
            // In Vulkan, Graphics/Compute queues typically support Transfer implicitly,
            // but checking the flag is good practice.
            VkQueueFlags computeFlags = queueFamilies[ indices.compute ].queueFlags;
            if( computeFlags & VK_QUEUE_TRANSFER_BIT )
            {
                indices.transfer = indices.compute;
            }
        }

        // Final fallback
        if( indices.transfer == -1 )
        {
            indices.transfer = indices.graphics;
        }

        return indices;
    }
} // namespace DigitalTwin