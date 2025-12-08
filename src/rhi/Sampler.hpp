#pragma once
#include "core/Base.hpp"
#include <volk.h>

namespace DigitalTwin
{
    // Common sampler configurations
    struct SamplerDesc
    {
        VkFilter             magFilter    = VK_FILTER_LINEAR;
        VkFilter             minFilter    = VK_FILTER_LINEAR;
        VkSamplerAddressMode addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerAddressMode addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        VkSamplerAddressMode addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        // Mipmapping defaults
        VkSamplerMipmapMode mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    };

    class Sampler
    {
    public:
        Sampler( VkDevice device, const VolkDeviceTable* api, const SamplerDesc& desc );
        ~Sampler();

        VkSampler GetHandle() const { return m_sampler; }

        // Copy disabled
        Sampler( const Sampler& )            = delete;
        Sampler& operator=( const Sampler& ) = delete;

    private:
        VkDevice               m_device;
        const VolkDeviceTable* m_api;
        VkSampler              m_sampler = VK_NULL_HANDLE;
    };
} // namespace DigitalTwin