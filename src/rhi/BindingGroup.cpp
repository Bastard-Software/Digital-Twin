#include "rhi/BindingGroup.hpp"

#include "core/Log.hpp"
#include "rhi/Device.hpp"

namespace DigitalTwin
{
    // Helper to map our internal ShaderResourceType to Vulkan Descriptor Type
    static VkDescriptorType MapResourceTypeToVulkan( ShaderResourceType type )
    {
        switch( type )
        {
            case ShaderResourceType::UNIFORM_BUFFER:
                return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            case ShaderResourceType::STORAGE_BUFFER:
                return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            case ShaderResourceType::SAMPLED_IMAGE:
                return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            case ShaderResourceType::STORAGE_IMAGE:
                return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            default:
                return VK_DESCRIPTOR_TYPE_MAX_ENUM;
        }
    }

    BindingGroup::BindingGroup( Ref<Device> device, VkDescriptorSet set, const ShaderReflectionData& layoutMap )
        : m_device( device )
        , m_set( set )
        , m_layoutMap( layoutMap )
    {
        m_bufferInfos.reserve( 16 );
        m_pendingWrites.reserve( 16 );
    }

    void BindingGroup::Set( const std::string& name, Ref<Buffer> buffer )
    {
        // 1. Validate resource existence in shader layout
        auto it = m_layoutMap.find( name );
        if( it == m_layoutMap.end() )
        {
            DT_CORE_WARN( "[BindingGroup] Shader does not contain resource named '{}'. Ignored.", name );
            return;
        }

        const ShaderResource& resourceInfo = it->second;

        // 2. Prepare Buffer Info
        VkDescriptorBufferInfo& bufInfo = m_bufferInfos.emplace_back();
        bufInfo.buffer                  = buffer->GetHandle();
        bufInfo.offset                  = 0;
        bufInfo.range                   = VK_WHOLE_SIZE;

        // 3. Prepare Write Descriptor
        VkWriteDescriptorSet write = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        write.dstSet               = m_set;
        write.dstBinding           = resourceInfo.binding;
        write.dstArrayElement      = 0;
        write.descriptorType       = MapResourceTypeToVulkan( resourceInfo.type );
        write.descriptorCount      = 1;
        write.pBufferInfo          = &bufInfo;

        m_pendingWrites.push_back( write );
    }

    void BindingGroup::Build()
    {
        if( m_pendingWrites.empty() )
            return;

        // Commit updates to the device
        m_device->UpdateDescriptorSets( m_pendingWrites );

        // Clear pending writes, but keep capacity.
        m_pendingWrites.clear();
        m_bufferInfos.clear();
    }
} // namespace DigitalTwin