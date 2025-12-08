#include "rhi/Sampler.hpp"

namespace DigitalTwin
{
    Sampler::Sampler( VkDevice device, const VolkDeviceTable* api, const SamplerDesc& desc )
        : m_device( device )
        , m_api( api )
    {
        VkSamplerCreateInfo createInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        createInfo.magFilter           = desc.magFilter;
        createInfo.minFilter           = desc.minFilter;
        createInfo.addressModeU        = desc.addressModeU;
        createInfo.addressModeV        = desc.addressModeV;
        createInfo.addressModeW        = desc.addressModeW;
        createInfo.mipmapMode          = desc.mipmapMode;

        // Defaults for typical use cases
        createInfo.anisotropyEnable        = VK_FALSE;
        createInfo.maxAnisotropy           = 1.0f;
        createInfo.borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        createInfo.unnormalizedCoordinates = VK_FALSE;
        createInfo.compareEnable           = VK_FALSE;
        createInfo.compareOp               = VK_COMPARE_OP_ALWAYS;
        createInfo.minLod                  = 0.0f;
        createInfo.maxLod                  = VK_LOD_CLAMP_NONE;

        if( m_api->vkCreateSampler( m_device, &createInfo, nullptr, &m_sampler ) != VK_SUCCESS )
        {
            DT_CORE_CRITICAL( "Failed to create sampler!" );
        }
    }

    Sampler::~Sampler()
    {
        if( m_sampler != VK_NULL_HANDLE )
        {
            m_api->vkDestroySampler( m_device, m_sampler, nullptr );
        }
    }
} // namespace DigitalTwin