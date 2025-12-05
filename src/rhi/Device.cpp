#include "rhi/Device.hpp"

#include "rhi/RHI.hpp"
#include <set>
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
        features12.pNext                            = &features12;
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
        m_graphicsQueue = CreateRef<CommandQueue>( m_device, m_api, indices.graphics, QueueType::GRAPHICS );

        // Compute
        if( indices.compute == indices.graphics )
        {
            m_computeQueue = m_graphicsQueue; // Alias: Point to the same object
            DT_CORE_INFO( "Compute Queue aliased to Graphics Queue (Family {})", indices.compute );
        }
        else
        {
            m_computeQueue = CreateRef<CommandQueue>( m_device, m_api, indices.compute, QueueType::COMPUTE );
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
            m_transferQueue = CreateRef<CommandQueue>( m_device, m_api, indices.transfer, QueueType::TRANSFER );
        }

        // --- VMA Init ---
        VmaVulkanFunctions vmaVulkanFunctions                  = {};
        vmaVulkanFunctions.vkGetInstanceProcAddr               = vkGetInstanceProcAddr;
        vmaVulkanFunctions.vkGetDeviceProcAddr                 = vkGetDeviceProcAddr;
        vmaVulkanFunctions.vkGetPhysicalDeviceProperties       = vkGetPhysicalDeviceProperties;
        vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
        vmaVulkanFunctions.vkAllocateMemory                    = m_api.vkAllocateMemory;
        vmaVulkanFunctions.vkFreeMemory                        = m_api.vkFreeMemory;
        vmaVulkanFunctions.vkMapMemory                         = m_api.vkMapMemory;
        vmaVulkanFunctions.vkUnmapMemory                       = m_api.vkUnmapMemory;
        vmaVulkanFunctions.vkFlushMappedMemoryRanges           = m_api.vkFlushMappedMemoryRanges;
        vmaVulkanFunctions.vkInvalidateMappedMemoryRanges      = m_api.vkInvalidateMappedMemoryRanges;
        vmaVulkanFunctions.vkBindBufferMemory                  = m_api.vkBindBufferMemory;
        vmaVulkanFunctions.vkBindImageMemory                   = m_api.vkBindImageMemory;
        vmaVulkanFunctions.vkGetBufferMemoryRequirements       = m_api.vkGetBufferMemoryRequirements;
        vmaVulkanFunctions.vkGetImageMemoryRequirements        = m_api.vkGetImageMemoryRequirements;
        vmaVulkanFunctions.vkCreateBuffer                      = m_api.vkCreateBuffer;
        vmaVulkanFunctions.vkDestroyBuffer                     = m_api.vkDestroyBuffer;
        vmaVulkanFunctions.vkCreateImage                       = m_api.vkCreateImage;
        vmaVulkanFunctions.vkDestroyImage                      = m_api.vkDestroyImage;
        vmaVulkanFunctions.vkCmdCopyBuffer                     = m_api.vkCmdCopyBuffer;

        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.physicalDevice         = m_physicalDevice;
        allocatorInfo.device                 = m_device;
        allocatorInfo.instance               = RHI::GetInstance();
        allocatorInfo.vulkanApiVersion       = VK_API_VERSION_1_4;
        allocatorInfo.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        allocatorInfo.pVulkanFunctions       = &vmaVulkanFunctions;

        if( vmaCreateAllocator( &allocatorInfo, &m_allocator ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create VMA allocator!" );
            return Result::FAIL;
        }

        DT_CORE_INFO( "Logical Device initialized. Queues indices -> G:{0} C:{1} T:{2}", indices.graphics, indices.compute, indices.transfer );

        return Result::SUCCESS;
    }

    void Device::Shutdown()
    {
        if( m_device == VK_NULL_HANDLE )
            return;

        {
            std::lock_guard<std::mutex> lock( m_poolMutex );
            m_commandPools.clear();
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

    Result Device::WaitForQueue( Ref<CommandQueue> queue, uint64_t waitValue, uint64_t timeout )
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