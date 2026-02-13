#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>
#include <functional>
#include <memory>
#include <vector>

namespace DigitalTwin
{
    class ResourceManager;

    class Renderer
    {
    public:
        Renderer( Device* device, Swapchain* swapchain, ResourceManager* resourceManager );
        ~Renderer();

        Result Create();
        void   Destroy();

        void BeginUI();
        void EndUI();

        void RecordUIPass( CommandBuffer* cmd, Texture* backbuffer );

        // --- Public API ---
        void  RenderUI( std::function<void()> callback );
        void* GetImGuiTextureID( TextureHandle handle );
        void* GetImGuiContext();

        SamplerHandle GetDefaultSampler() const { return m_defaultSampler; }

    private:
        Device*          m_device;
        Swapchain*       m_swapchain;
        ResourceManager* m_resourceManager;

        // Default resources
        SamplerHandle m_defaultSampler;
        void*         m_imguiDescriptorPool = nullptr;
    };

} // namespace DigitalTwin