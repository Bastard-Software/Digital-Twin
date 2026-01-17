#include "rhi/Sampler.h"

#include "core/Log.h"

namespace DigitalTwin
{
    Sampler::Sampler( VkDevice device, const VolkDeviceTable* api )
        : m_device( device )
        , m_api( api )
    {
    }

    Sampler::~Sampler()
    {
    }

    Result Sampler::Create( const SamplerDesc& desc )
    {
        m_desc = desc;

        VkSamplerCreateInfo createInfo = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        createInfo.magFilter           = m_desc.magFilter;
        createInfo.minFilter           = m_desc.minFilter;
        createInfo.addressModeU        = m_desc.addressModeU;
        createInfo.addressModeV        = m_desc.addressModeV;
        createInfo.addressModeW        = m_desc.addressModeW;
        createInfo.mipmapMode          = m_desc.mipmapMode;

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
            DT_CRITICAL( "Failed to create sampler!" );
        }

        return Result();
    }

    void Sampler::Destroy()
    {
        if( m_sampler != VK_NULL_HANDLE )
        {
            m_api->vkDestroySampler( m_device, m_sampler, nullptr );
        }
    }

    Sampler::Sampler( Sampler&& other ) noexcept
        : m_device( other.m_device )
        , m_api( other.m_api )
        , m_sampler( other.m_sampler )
        , m_desc( other.m_desc )
    {
        other.m_sampler = VK_NULL_HANDLE;
    }

    Sampler& Sampler::operator=( Sampler&& other ) noexcept
    {
        if( this != &other )
        {
            m_device  = other.m_device;
            m_api     = other.m_api;
            m_sampler = other.m_sampler;
            m_desc    = other.m_desc;

            other.m_sampler = VK_NULL_HANDLE;
        }
        return *this;
    }

} // namespace DigitalTwin