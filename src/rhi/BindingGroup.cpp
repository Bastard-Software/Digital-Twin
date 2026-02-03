#include "rhi/BindingGroup.h"

#include "core/Log.h"
#include "rhi/Buffer.h"
#include "rhi/DescriptorAllocator.h"
#include "rhi/Device.h"
#include "rhi/Sampler.h"
#include "rhi/Texture.h"

namespace DigitalTwin
{

    BindingGroup::BindingGroup( Device* device, DescriptorAllocator* allocator, VkDescriptorSetLayout layout, uint32_t setIndex )
        : m_device( device )
        , m_allocator( allocator )
        , m_layout( layout )
        , m_setIndex( setIndex )
    {
    }

    BindingGroup::~BindingGroup()
    {
        // Note: We don't free individual descriptor sets here.
        // They are released when the DescriptorPool (owned by DescriptorAllocator) is reset or destroyed.
    }

    void BindingGroup::Bind( uint32_t binding, Buffer* buffer, VkDeviceSize offset, VkDeviceSize range )
    {
        if( !buffer )
            return;

        BindingInfo info{};
        info.binding           = binding;
        info.bufferInfo.buffer = buffer->GetHandle();
        info.bufferInfo.offset = offset;
        info.bufferInfo.range  = ( range == VK_WHOLE_SIZE ) ? buffer->GetSize() : range;

        // Infer descriptor type from buffer type
        // TODO: Validate against shader reflection data for robustness
        if( buffer->GetType() == BufferType::UNIFORM )
            info.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        else
            info.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

        m_pendingBindings[ binding ] = info;
        m_dirty                      = true;
    }

    void BindingGroup::Bind( uint32_t binding, Texture* texture, VkImageLayout layout )
    {
        if( !texture )
            return;

        BindingInfo info{};
        info.binding               = binding;
        info.imageInfo.imageView   = texture->GetView();
        info.imageInfo.imageLayout = layout;
        info.imageInfo.sampler     = VK_NULL_HANDLE;

        // Infer type: GENERAL layout usually implies storage image
        if( layout == VK_IMAGE_LAYOUT_GENERAL )
            info.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else
            info.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;

        m_pendingBindings[ binding ] = info;
        m_dirty                      = true;
    }

    void BindingGroup::Bind( uint32_t binding, Texture* texture, Sampler* sampler, VkImageLayout layout )
    {
        if( !texture || !sampler )
            return;

        BindingInfo info{};
        info.binding               = binding;
        info.imageInfo.imageView   = texture->GetView();
        info.imageInfo.imageLayout = layout;
        info.imageInfo.sampler     = sampler->GetHandle();
        info.type                  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;

        m_pendingBindings[ binding ] = info;
        m_dirty                      = true;
    }

    Result BindingGroup::Build()
    {
        // Optimization: If nothing changed and set is allocated, do nothing
        if( !m_dirty && m_set != VK_NULL_HANDLE )
            return Result::SUCCESS;

        // 1. Allocate Descriptor Set if needed
        if( m_set == VK_NULL_HANDLE )
        {
            if( m_allocator->Allocate( m_layout, m_set ) != Result::SUCCESS )
            {
                DT_ERROR( "BindingGroup: Failed to allocate Descriptor Set!" );
                return Result::FAIL;
            }
        }

        // 2. Prepare Vulkan Writes
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve( m_pendingBindings.size() );

        for( auto& [ binding, info ]: m_pendingBindings )
        {
            VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            write.dstSet               = m_set;
            write.dstBinding           = binding;
            write.dstArrayElement      = 0;
            write.descriptorCount      = 1; // Arrays not supported yet
            write.descriptorType       = info.type;

            if( info.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || info.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER )
            {
                write.pBufferInfo = &info.bufferInfo;
            }
            else
            {
                write.pImageInfo = &info.imageInfo;
            }

            writes.push_back( write );
        }

        // 3. Update GPU
        if( !writes.empty() )
        {
            const auto& api = m_device->GetAPI();
            api.vkUpdateDescriptorSets( m_device->GetHandle(), ( uint32_t )writes.size(), writes.data(), 0, nullptr );
        }

        m_dirty = false;
        return Result::SUCCESS;
    }

} // namespace DigitalTwin