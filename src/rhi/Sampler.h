#pragma once
#include "rhi/RHITypes.h"

#include "core/Core.h"
#include <volk.h>

namespace DigitalTwin
{

    class Sampler
    {
    public:
        Sampler( VkDevice device, const VolkDeviceTable* api );
        ~Sampler();

        Result Create( const SamplerDesc& desc );
        void   Destroy();

        VkSampler GetHandle() const { return m_sampler; }

    public:
        // Disable copying (RAII), allow moving
        Sampler( const Sampler& )            = delete;
        Sampler& operator=( const Sampler& ) = delete;
        Sampler( Sampler&& other ) noexcept;
        Sampler& operator=( Sampler&& other ) noexcept;

    private:
        VkDevice               m_device;
        const VolkDeviceTable* m_api;
        VkSampler              m_sampler = VK_NULL_HANDLE;
        SamplerDesc            m_desc;
    };

} // namespace DigitalTwin