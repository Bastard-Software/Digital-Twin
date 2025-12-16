#include "rhi/Buffer.hpp"

#include <cstring>

namespace DigitalTwin
{
    Buffer::Buffer( VmaAllocator allocator, VkDevice device, const VolkDeviceTable* api )
        : m_allocator( allocator )
        , m_device( device )
        , m_api( api )
    {
    }

    Buffer::Buffer( Buffer&& other ) noexcept
        : m_allocator( other.m_allocator )
        , m_device( other.m_device )
        , m_buffer( other.m_buffer )
        , m_allocation( other.m_allocation )
        , m_size( other.m_size )
        , m_type( other.m_type )
        , m_mappedData( other.m_mappedData )
    {
        other.m_buffer     = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_mappedData = nullptr;
    }

    Buffer& Buffer::operator=( Buffer&& other ) noexcept
    {
        if( this != &other )
        {
            Destroy();
            m_allocator  = other.m_allocator;
            m_device     = other.m_device;
            m_buffer     = other.m_buffer;
            m_allocation = other.m_allocation;
            m_size       = other.m_size;
            m_type       = other.m_type;
            m_mappedData = other.m_mappedData;

            other.m_buffer     = VK_NULL_HANDLE;
            other.m_allocation = VK_NULL_HANDLE;
            other.m_mappedData = nullptr;
        }
        return *this;
    }

    Buffer::~Buffer()
    {
        Destroy();
    }

    Result Buffer::Create( const BufferDesc& desc )
    {
        m_size = desc.size;
        m_type = desc.type;

        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size               = desc.size;
        bufferInfo.sharingMode        = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo allocInfo = {};

        // Automatically select usage flags based on abstract BufferType
        switch( desc.type )
        {
            case BufferType::UPLOAD:
                // CPU -> GPU (Staging).
                // VMA_MEMORY_USAGE_CPU_TO_GPU guarantees HOST_VISIBLE and HOST_COHERENT.
                bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                allocInfo.usage  = VMA_MEMORY_USAGE_CPU_TO_GPU;
                allocInfo.flags  = VMA_ALLOCATION_CREATE_MAPPED_BIT;
                break;

            case BufferType::READBACK:
                // GPU -> CPU.
                // VMA_MEMORY_USAGE_GPU_TO_CPU guarantees HOST_VISIBLE and preferably HOST_CACHED.
                bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                allocInfo.usage  = VMA_MEMORY_USAGE_GPU_TO_CPU;
                allocInfo.flags  = VMA_ALLOCATION_CREATE_MAPPED_BIT;
                break;

            case BufferType::STORAGE:
                // Standard SSBO on GPU (VRAM).
                // Included TRANSFER bits for data upload/download.
                // Included DEVICE_ADDRESS for pointer-like access in shaders.
                bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                break;

            case BufferType::UNIFORM:
                // Constant buffer (UBO)
                bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                allocInfo.usage  = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                break;

            case BufferType::MESH:
                // Buffer for mesh data containing indces and GPU is able to write there
                bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                allocInfo.usage  = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                break;

            case BufferType::INDIRECT:
                // Indirect Buffer must be STORAGE so Compute Shader can generate commands into it.
                // Included INDIRECT_BUFFER_BIT for vkCmdDispatchIndirect / vkCmdDrawIndirect.
                bufferInfo.usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT; // TODO: Do we need this flags?
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                break;
            case BufferType::ATOMIC_COUNTER:
                // Atomic Counter is just a small STORAGE buffer (4 bytes).
                bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
                allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                break;
        }

        VmaAllocationInfo resultInfo;
        if( vmaCreateBuffer( m_allocator, &bufferInfo, &allocInfo, &m_buffer, &m_allocation, &resultInfo ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create buffer!" );
            return Result::FAIL;
        }

        if( allocInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT )
        {
            m_mappedData = resultInfo.pMappedData;
        }

        return Result::SUCCESS;
    }

    void Buffer::Destroy()
    {
        if( m_buffer != VK_NULL_HANDLE )
        {
            m_mappedData = nullptr;
            vmaDestroyBuffer( m_allocator, m_buffer, m_allocation );
            m_buffer     = VK_NULL_HANDLE;
            m_allocation = VK_NULL_HANDLE;
        }
    }

    void* Buffer::Map()
    {
        DT_ASSERT( m_type == BufferType::UPLOAD || m_type == BufferType::READBACK, "Mapping is only allowed for UPLOAD and READBACK buffers!" );

        if( m_mappedData )
            return m_mappedData;

        void* data;
        if( vmaMapMemory( m_allocator, m_allocation, &data ) != VK_SUCCESS )
        {
            DT_CORE_ERROR( "Failed to map buffer memory!" );
            return nullptr;
        }
        return data;
    }

    void Buffer::Unmap()
    {
        DT_ASSERT( m_type == BufferType::UPLOAD || m_type == BufferType::READBACK, "Mapping is only allowed for UPLOAD and READBACK buffers!" );

        if( !m_mappedData )
        {
            vmaUnmapMemory( m_allocator, m_allocation );
        }
    }

    void Buffer::Write( const void* data, size_t size, size_t offset )
    {
        DT_ASSERT( m_type == BufferType::UPLOAD || m_type == BufferType::READBACK, "Host writting is only allowed for UPLOAD and READBACK buffers!" );

        void* ptr = Map();
        if( ptr )
        {
            uint8_t* target = static_cast<uint8_t*>( ptr ) + offset;
            memcpy( target, data, size );
        }
    }

    void Buffer::Read( void* outData, size_t size, size_t offset )
    {
        DT_ASSERT( m_type == BufferType::UPLOAD || m_type == BufferType::READBACK, "Host reading is only allowed for UPLOAD and READBACK buffers!" );

        void* ptr = Map();
        if( ptr )
        {
            // Invalidate cache to ensure we read the latest GPU writes
            vmaInvalidateAllocation( m_allocator, m_allocation, offset, size );
            const uint8_t* source = static_cast<const uint8_t*>( ptr ) + offset;
            memcpy( outData, source, size );
        }
    }

    Result Buffer::Invalidate( size_t size, size_t offset )
    {
        if( m_allocation )
        {
            VkDeviceSize dataSize = ( size == VK_WHOLE_SIZE ) ? m_size : size;

            VkResult res = vmaInvalidateAllocation( m_allocator, m_allocation, offset, dataSize );
            if( res != VK_SUCCESS )
            {
                DT_CORE_ERROR( "Buffer Invalidate failed!" );
                return Result::FAIL;
            }
            return Result::SUCCESS;
        }
        return Result::FAIL;
    }

    VkDescriptorBufferInfo Buffer::GetDescriptorInfo( VkDeviceSize offset, VkDeviceSize range ) const
    {
        VkDescriptorBufferInfo info{};
        info.buffer = m_buffer;
        info.offset = offset;
        info.range  = range;
        return info;
    }

    uint64_t Buffer::GetDeviceAddress() const
    {
        VkBufferDeviceAddressInfo info = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        info.buffer                    = m_buffer;
        return m_api->vkGetBufferDeviceAddress( m_device, &info );
    }
} // namespace DigitalTwin