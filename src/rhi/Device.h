#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include "rhi/Queue.h"
#include <vector>
#include <vma/vk_mem_alloc.h>
#include <volk.h>

namespace DigitalTwin
{
    class Device
    {
    public:
        Device( VkPhysicalDevice physicalDevice, VkInstance instance );
        ~Device();

        Result Initialize( const DeviceDesc& desc );
        void   Shutdown();

        Result CreateBuffer( const BufferDesc& desc, Buffer* buffer );
        void   DestroyBuffer( Buffer* buffer );

        Result CreateTexture( const TextureDesc& desc, Texture* texture );
        void   DestroyTexture( Texture* texture );

        Result CreateSampler( const SamplerDesc& desc, Sampler* sampler );
        void   DestroySampler( Sampler* sampler );

        Result CreateShader( const std::string& filepath, Shader* shader );
        void   DestroyShader( Shader* shader );

        Result CreateComputePipeline( const ComputePipelineNativeDesc& desc, ComputePipeline* pipeline );
        void   DestroyComputePipeline( ComputePipeline* pipeline );

        Result CreateGraphicsPipeline( const GraphicsPipelineNativeDesc& desc, GraphicsPipeline* pipeline );
        void   DestroyGraphicsPipeline( GraphicsPipeline* pipeline );

        ThreadContextHandle CreateThreadContext();
        ThreadContext*      GetThreadContext( ThreadContextHandle handle );

        VkDevice         GetHandle() const { return m_device; }
        VkPhysicalDevice GetPhysicalDevice() const { return m_physicalDevice; }
        VmaAllocator     GetAllocator() const { return m_allocator; }

        // Returns the function table loaded for this specific device
        const VolkDeviceTable& GetAPI() const { return m_api; }

        Queue* GetGraphicsQueue() const { return m_graphicsQueue; }
        Queue* GetComputeQueue() const { return m_computeQueue; }
        Queue* GetTransferQueue() const { return m_transferQueue; }

        void WaitIdle();

    private:
        struct QueueFamilyIndices;
        QueueFamilyIndices FindQueueFamilies( VkPhysicalDevice device );

    private:
        VkInstance       m_instance;
        VkPhysicalDevice m_physicalDevice;
        VkDevice         m_device = VK_NULL_HANDLE;
        VolkDeviceTable  m_api    = {}; // Function table for this device
        DeviceDesc       m_desc;

        // Vma Allocator
        VmaAllocator m_allocator = VK_NULL_HANDLE;

        // Queues
        Queue* m_graphicsQueue = nullptr;
        Queue* m_computeQueue  = nullptr;
        Queue* m_transferQueue = nullptr;

        // To control lifetime of owned queues
        std::vector<Scope<Queue>>         m_ownedQueues;
        std::vector<Scope<ThreadContext>> m_threadContexts;
        std::mutex                        m_threadContextMutex;
    };
} // namespace DigitalTwin