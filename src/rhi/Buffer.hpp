#pragma once
#include "core/Base.hpp"
#include <vk_mem_alloc.h>
#include <volk.h>

namespace DigitalTwin
{
    // Defines the intended usage and memory location of the buffer
    enum class BufferType
    {
        UPLOAD,
        READBACK,
        STORAGE,
        UNIFORM,
        VERTEX,
        INDEX,
        INDIRECT
    };

    struct BufferDesc
    {
        VkDeviceSize size = 0;
        BufferType   type = BufferType::STORAGE;

        // Optional: Additional Vulkan usage flags (e.g., if we want TRANSFER_SRC on a STORAGE buffer)
        VkBufferUsageFlags additionalUsage = 0;
    };

    class Buffer
    {
    public:
        Buffer( VmaAllocator allocator, VkDevice device, const VolkDeviceTable* api );
        ~Buffer();

        Result Create( const BufferDesc& desc );
        void   Destroy();

        // Only allowe in UPLOAD and READBACK buffers
        void* Map();
        void  Unmap();
        void  Write( const void* data, size_t size, size_t offset = 0 );
        void  Read( void* outData, size_t size, size_t offset = 0 );

        VkDescriptorBufferInfo GetDescriptorInfo( VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE ) const;

        VkBuffer     GetHandle() const { return m_buffer; }
        VkDeviceSize GetSize() const { return m_size; }
        BufferType   GetType() const { return m_type; }
        uint64_t     GetDeviceAddress() const;

    public:
        // Disable copying (RAII), allow moving
        Buffer( const Buffer& )            = delete;
        Buffer& operator=( const Buffer& ) = delete;
        Buffer( Buffer&& other ) noexcept;
        Buffer& operator=( Buffer&& other ) noexcept;

    private:
        VmaAllocator           m_allocator;
        VkDevice               m_device;
        const VolkDeviceTable* m_api;
        VkBuffer               m_buffer     = VK_NULL_HANDLE;
        VmaAllocation          m_allocation = VK_NULL_HANDLE;
        VkDeviceSize           m_size       = 0;
        BufferType             m_type       = BufferType::STORAGE;
        void*                  m_mappedData = nullptr;
    };
} // namespace DigitalTwin