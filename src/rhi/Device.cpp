#include "rhi/Device.h"

#include "core/Log.h"
#include "rhi/Buffer.h"
#include "rhi/Pipeline.h"
#include "rhi/Queue.h"
#include "rhi/Sampler.h"
#include "rhi/Shader.h"
#include "rhi/Texture.h"
#include "rhi/ThreadContext.h"
#include <set>
#include <vector>

namespace DigitalTwin
{
    struct Device::QueueFamilyIndices
    {
        int32_t graphics = -1;
        int32_t compute  = -1;
        int32_t transfer = -1;

        bool IsValid() const { return graphics != -1; }
    };

    Device::Device( VkPhysicalDevice physicalDevice, VkInstance instance )
        : m_physicalDevice( physicalDevice )
        , m_instance( instance )
    {
    }

    Device::~Device()
    {
    }

    Result Device::Initialize( const DeviceDesc& desc )
    {
        m_desc = desc;

        // 1. Find queue families
        QueueFamilyIndices indices = FindQueueFamilies( m_physicalDevice );
        if( !indices.IsValid() )
        {
            DT_ERROR( "Failed to find suitable queue families!" );
            return Result::FAIL;
        }

        // 2. Setup Queue Create Info
        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<int32_t>                    uniqueQueueFamilies = { indices.graphics, indices.compute, indices.transfer };
        if( uniqueQueueFamilies.count( -1 ) )
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

        // 3. Create Logical Device
        VkPhysicalDeviceFeatures         deviceFeatures = {};
        VkPhysicalDeviceVulkan12Features features12     = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        features12.timelineSemaphore                    = VK_TRUE;
        features12.bufferDeviceAddress                  = VK_TRUE;

        VkPhysicalDeviceVulkan13Features features13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
        features13.pNext                            = &features12;
        features13.synchronization2                 = VK_TRUE;
        features13.dynamicRendering                 = VK_TRUE;

        VkPhysicalDeviceVulkan14Features features14 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
        features14.pNext                            = &features13;

        VkPhysicalDeviceFeatures2 deviceFeatures2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        deviceFeatures2.features                  = deviceFeatures;
        deviceFeatures2.pNext                     = &features13;

        std::vector<const char*> deviceExtensions;
        if( !m_desc.headless )
        {
            deviceExtensions.push_back( VK_KHR_SWAPCHAIN_EXTENSION_NAME );
        }
#ifdef __APPLE__
        deviceExtensions.push_back( "VK_KHR_portability_subset" );
#endif

        VkDeviceCreateInfo createInfo      = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        createInfo.pNext                   = &features14;
        createInfo.queueCreateInfoCount    = static_cast<uint32_t>( queueCreateInfos.size() );
        createInfo.pQueueCreateInfos       = queueCreateInfos.data();
        createInfo.pEnabledFeatures        = &deviceFeatures;
        createInfo.enabledExtensionCount   = static_cast<uint32_t>( deviceExtensions.size() );
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if( vkCreateDevice( m_physicalDevice, &createInfo, nullptr, &m_device ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to create logical device!" );
            return Result::FAIL;
        }

        // 4. Load Device Table (Volk)
        volkLoadDeviceTable( &m_api, m_device );

        // 5. Create Queues
        // Clear previous state just in case
        m_ownedQueues.clear();

        // --- Graphics Queue ---
        {
            auto gfxScope   = CreateScope<Queue>( m_device, m_api, indices.graphics, QueueType::GRAPHICS );
            m_graphicsQueue = gfxScope.get();                 // Store raw pointer
            m_ownedQueues.push_back( std::move( gfxScope ) ); // Transfer ownership
        }

        // --- Compute Queue ---
        if( indices.compute == indices.graphics )
        {
            DT_INFO( "RHI: Compute queue aliased to Graphics queue." );
            m_computeQueue = m_graphicsQueue; // Alias the raw pointer
        }
        else
        {
            auto compScope = CreateScope<Queue>( m_device, m_api, indices.compute, QueueType::COMPUTE );
            m_computeQueue = compScope.get();
            m_ownedQueues.push_back( std::move( compScope ) );
        }

        // --- Transfer Queue ---
        if( indices.transfer == indices.graphics )
        {
            m_transferQueue = m_graphicsQueue; // Alias to Graphics
        }
        else if( indices.transfer == indices.compute )
        {
            m_transferQueue = m_computeQueue; // Alias to Compute
        }
        else
        {
            auto transScope = CreateScope<Queue>( m_device, m_api, indices.transfer, QueueType::TRANSFER );
            m_transferQueue = transScope.get();
            m_ownedQueues.push_back( std::move( transScope ) );
        }

        // 6. VMA Allocator
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
        allocatorInfo.instance               = m_instance;
        allocatorInfo.vulkanApiVersion       = VK_API_VERSION_1_4; // With api version 1.4, VMA needs some newer functions
        allocatorInfo.flags                  = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
        allocatorInfo.pVulkanFunctions       = &vmaVulkanFunctions;

        if( vmaCreateAllocator( &allocatorInfo, &m_allocator ) != VK_SUCCESS )
        {
            DT_CRITICAL( "Failed to create VMA allocator!" );
            return Result::FAIL;
        }

        DT_INFO( "Device Initialized. Queues: G:{0} C:{1} T:{2}", indices.graphics, indices.compute, indices.transfer );
        return Result::SUCCESS;
    }

    void Device::Shutdown()
    {
        if( m_device != VK_NULL_HANDLE )
        {
            WaitIdle();

            // 1. Destroy VMA Allocator
            if( m_allocator != VK_NULL_HANDLE )
            {
                vmaDestroyAllocator( m_allocator );
                m_allocator = VK_NULL_HANDLE;
            }

            // 2. Destroy Thread Contexts
            m_threadContexts.clear();

            // 3. Destroy all queues immediately.
            // Queue destructors call vkDestroySemaphore, so VkDevice must still be valid here.
            m_ownedQueues.clear();

            // 4. Clear raw pointers to prevent dangling usage (safety)
            m_graphicsQueue = nullptr;
            m_computeQueue  = nullptr;
            m_transferQueue = nullptr;

            // 5. Destroy Device
            m_api.vkDestroyDevice( m_device, nullptr );
            m_device = VK_NULL_HANDLE;
            DT_INFO( "Device Shutdown." );
        }
    }

    Result Device::CreateBuffer( const BufferDesc& desc, Buffer* buffer )
    {
        if( !buffer )
        {
            DT_ERROR( "CreateBuffer: buffer pointer is null!" );
            return Result::FAIL;
        }

        return buffer->Create( desc );
    }

    void Device::DestroyBuffer( Buffer* buffer )
    {
        if( buffer )
        {
            buffer->Destroy();
        }
    }

    Result Device::CreateTexture( const TextureDesc& desc, Texture* texture )
    {
        if( !texture )
        {
            DT_ERROR( "CreateTexture: texture pointer is null!" );
            return Result::FAIL;
        }

        return texture->Create( desc );
    }

    void Device::DestroyTexture( Texture* texture )
    {
        if( texture )
        {
            texture->Destroy();
        }
    }

    Result Device::CreateSampler( const SamplerDesc& desc, Sampler* sampler )
    {
        if( !sampler )
        {
            DT_ERROR( "CreateSampler: sampler pointer is null!" );
            return Result::FAIL;
        }

        return sampler->Create( desc );
    }

    void Device::DestroySampler( Sampler* sampler )
    {
        if( sampler )
        {
            sampler->Destroy();
        }
    }

    Result Device::CreateShader( const std::string& filepath, Shader* shader )
    {
        if( !shader )
        {
            DT_ERROR( "CreateSampler: sampler pointer is null!" );
            return Result::FAIL;
        }

        return shader->Create( filepath );
    }

    void Device::DestroyShader( Shader* shader )
    {
        if( shader )
        {
            shader->Destroy();
        }
    }

    Result Device::CreateComputePipeline( const ComputePipelineNativeDesc& desc, ComputePipeline* pipeline )
    {
        if( !pipeline )
        {
            DT_ERROR( "CreateComputePipeline: pipeline pointer is null!" );
            return Result::FAIL;
        }

        return pipeline->Create( desc );
    }

    void Device::DestroyComputePipeline( ComputePipeline* pipeline )
    {
        if( pipeline )
        {
            pipeline->Destroy();
        }
    }

    Result Device::CreateGraphicsPipeline( const GraphicsPipelineNativeDesc& desc, GraphicsPipeline* pipeline )
    {
        if( !pipeline )
        {
            DT_ERROR( "CreateGraphicsPipeline: pipeline pointer is null!" );
            return Result::FAIL;
        }

        return pipeline->Create( desc );
    }

    void Device::DestroyGraphicsPipeline( GraphicsPipeline* pipeline )
    {
        if( pipeline )
        {
            pipeline->Destroy();
        }
    }

    void Device::WaitIdle()
    {
        if( m_device )
        {
            m_api.vkDeviceWaitIdle( m_device );
        }
    }

    Result Device::WaitForSemaphores( const std::vector<VkSemaphore>& semaphores, const std::vector<uint64_t>& values )
    {
        if( semaphores.empty() || semaphores.size() != values.size() )
        {
            return Result::FAIL;
        }

        VkSemaphoreWaitInfo waitInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO };
        waitInfo.flags               = 0;
        waitInfo.semaphoreCount      = ( uint32_t )semaphores.size();
        waitInfo.pSemaphores         = semaphores.data();
        waitInfo.pValues             = values.data();
        VkResult res                 = m_api.vkWaitSemaphores( m_device, &waitInfo, UINT64_MAX );

        if( res != VK_SUCCESS )
        {
            DT_ERROR( "WaitForSemaphores failed!" );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    ThreadContextHandle Device::CreateThreadContext( QueueType type )
    {
        auto ctx = CreateScope<ThreadContext>();

        uint32_t familyIndex = 0;
        if( type == QueueType::GRAPHICS )
            familyIndex = m_graphicsQueue->GetFamilyIndex();
        else if( type == QueueType::COMPUTE )
            familyIndex = m_computeQueue->GetFamilyIndex();
        else if( type == QueueType::TRANSFER )
            familyIndex = m_transferQueue->GetFamilyIndex();
        else
        {
            DT_ERROR( "Invalid QueueType specified for ThreadContext creation!" );
            return ThreadContextHandle::Invalid;
        }

        if( ctx->Initialize( m_device, &m_api, type, familyIndex ) != Result::SUCCESS )
        {
            return ThreadContextHandle::Invalid;
        }

        std::lock_guard<std::mutex> lock( m_threadContextMutex );

        uint32_t index = ( uint32_t )m_threadContexts.size();
        m_threadContexts.push_back( std::move( ctx ) );

        return ThreadContextHandle( index, 1 );
    }

    ThreadContext* Device::GetThreadContext( ThreadContextHandle handle )
    {
        std::lock_guard<std::mutex> lock( m_threadContextMutex );

        if( !handle.IsValid() || handle.GetIndex() >= m_threadContexts.size() )
            return nullptr;
        return m_threadContexts[ handle.GetIndex() ].get();
    }

    Device::QueueFamilyIndices Device::FindQueueFamilies( VkPhysicalDevice device )
    {
        QueueFamilyIndices indices;
        uint32_t           count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties( device, &count, nullptr );

        std::vector<VkQueueFamilyProperties> families( count );
        vkGetPhysicalDeviceQueueFamilyProperties( device, &count, families.data() );

        // 1. Graphics
        for( int i = 0; i < ( int )families.size(); ++i )
        {
            if( families[ i ].queueFlags & VK_QUEUE_GRAPHICS_BIT )
            {
                indices.graphics = i;
                break;
            }
        }

        // 2. Dedicated Compute
        for( int i = 0; i < ( int )families.size(); ++i )
        {
            if( ( families[ i ].queueFlags & VK_QUEUE_COMPUTE_BIT ) && !( families[ i ].queueFlags & VK_QUEUE_GRAPHICS_BIT ) )
            {
                indices.compute = i;
                break;
            }
        }
        if( indices.compute == -1 )
        {
            for( int i = 0; i < ( int )families.size(); ++i )
                if( families[ i ].queueFlags & VK_QUEUE_COMPUTE_BIT )
                {
                    indices.compute = i;
                    break;
                }
        }

        // 3. Dedicated Transfer
        for( int i = 0; i < ( int )families.size(); ++i )
        {
            if( ( families[ i ].queueFlags & VK_QUEUE_TRANSFER_BIT ) && !( families[ i ].queueFlags & VK_QUEUE_GRAPHICS_BIT ) &&
                !( families[ i ].queueFlags & VK_QUEUE_COMPUTE_BIT ) )
            {
                indices.transfer = i;
                break;
            }
        }
        if( indices.transfer == -1 )
        {
            if( indices.compute != -1 )
                indices.transfer = indices.compute;
            else
                indices.transfer = indices.graphics;
        }

        return indices;
    }

} // namespace DigitalTwin