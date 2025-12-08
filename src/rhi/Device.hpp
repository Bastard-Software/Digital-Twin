#pragma once
#include "core/Base.hpp"
#include "rhi/Buffer.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/DescriptorAllocator.hpp"
#include "rhi/Pipeline.hpp"
#include "rhi/Queue.hpp"
#include "rhi/Shader.hpp"
#include "rhi/Swapchain.hpp"
#include "rhi/Texture.hpp"
#include <map>
#include <mutex>
#include <thread>
#include <vk_mem_alloc.h>
#include <volk.h>

namespace DigitalTwin
{
    struct DeviceDesc
    {
        bool_t headless = false;
    };

    class Device
    {
    public:
        Device( VkPhysicalDevice physicalDevice );
        ~Device();

        Result Init( DeviceDesc desc );
        void   Shutdown();

        Ref<CommandBuffer>    CreateCommandBuffer( QueueType type );
        Ref<Buffer>           CreateBuffer( const BufferDesc& desc );
        Ref<Texture>          CreateTexture( const TextureDesc& desc );
        Ref<Shader>           CreateShader( const std::string& filepath );
        Ref<ComputePipeline>  CreateComputePipeline( const ComputePipelineDesc& desc );
        Ref<GraphicsPipeline> CreateGraphicsPipeline( const GraphicsPipelineDesc& desc );
        Ref<Swapchain>        CreateSwapchain( const SwapchainDesc& desc );

        Result AllocateDescriptor( VkDescriptorSetLayout layout, VkDescriptorSet& outSet );
        void   ResetDescriptorPools();

        // Convenience wrappers for textures
        Ref<Texture> CreateTexture1D( uint32_t width, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                                      TextureUsage usage = TextureUsage::SAMPLED | TextureUsage::STORAGE );

        Ref<Texture> CreateTexture2D( uint32_t width, uint32_t height, VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
                                      TextureUsage usage = TextureUsage::SAMPLED | TextureUsage::STORAGE );

        Ref<Texture> CreateTexture3D( uint32_t width, uint32_t height, uint32_t depth, VkFormat format = VK_FORMAT_R32_SFLOAT,
                                      TextureUsage usage = TextureUsage::SAMPLED | TextureUsage::STORAGE );

        /**
         * @brief Waits for a specific value on the queue's timeline semaphore.
         * @param queue The queue to wait on.
         * @param waitValue The value to wait for.
         * @param timeout Timeout in nanoseconds.
         * @return Result::SUCCESS, Result::TIMEOUT, or Result::FAIL.
         */
        Result WaitForQueue( Ref<Queue> queue, uint64_t waitValue, uint64_t timeout = UINT64_MAX );

        // Getters for queues (may point to the same object if hardware has a single queue family)
        Ref<Queue> GetGraphicsQueue() const { return m_graphicsQueue; }
        Ref<Queue> GetComputeQueue() const { return m_computeQueue; }
        Ref<Queue> GetTransferQueue() const { return m_transferQueue; }

        VkDevice               GetHandle() const { return m_device; }
        VkPhysicalDevice       GetPhysicalDevice() const { return m_physicalDevice; }
        VmaAllocator           GetAllocator() const { return m_allocator; }
        const VolkDeviceTable& GetAPI() const { return m_api; }

    private:
        struct QueueFamilyIndices;
        QueueFamilyIndices FindQueueFamilies( VkPhysicalDevice device );

        VkCommandPool GetOrCreateThreadLocalPool( uint32_t queueFamilyIndex );

    private:
        VkPhysicalDevice m_physicalDevice;
        VkDevice         m_device;
        VmaAllocator     m_allocator;
        VolkDeviceTable  m_api;
        DeviceDesc       m_desc;

        Ref<Queue> m_graphicsQueue;
        Ref<Queue> m_computeQueue;
        Ref<Queue> m_transferQueue;

        Ref<DescriptorAllocator> m_descriptorAllocator;

        // --- Thread Local Command Pools Management ---
        struct PoolInfo
        {
            VkCommandPool          handle = VK_NULL_HANDLE;
            VkDevice               device = VK_NULL_HANDLE;
            const VolkDeviceTable* api    = nullptr; // Pointer to the device dispatch table

            PoolInfo() = default;

            // Move constructor
            PoolInfo( PoolInfo&& other ) noexcept
                : handle( other.handle )
                , device( other.device )
                , api( other.api )
            {
                other.handle = VK_NULL_HANDLE;
                other.api    = nullptr;
            }

            // Move assignment
            PoolInfo& operator=( PoolInfo&& other ) noexcept
            {
                if( this != &other )
                {
                    // Clean up current resource using the stored API table
                    if( handle && api )
                    {
                        api->vkDestroyCommandPool( device, handle, nullptr );
                    }

                    handle = other.handle;
                    device = other.device;
                    api    = other.api;

                    other.handle = VK_NULL_HANDLE;
                    other.api    = nullptr;
                }
                return *this;
            }

            // Destructor
            ~PoolInfo()
            {
                // CORRECTED: Using the device-specific table for destruction
                if( handle && api )
                {
                    api->vkDestroyCommandPool( device, handle, nullptr );
                }
            }
        };

        // Map: ThreadID -> (QueueFamilyIndex -> PoolInfo)
        std::map<std::thread::id, std::map<uint32_t, PoolInfo>> m_threadPools;
        std::mutex                                              m_poolMutex;
    };
} // namespace DigitalTwin
