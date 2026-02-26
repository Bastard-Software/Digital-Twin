#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>

namespace DigitalTwin
{
    class ResourceManager;

    class RenderTarget
    {
    public:
        RenderTarget( ResourceManager* rm, uint32_t width, uint32_t height );
        ~RenderTarget();

        void Resize( uint32_t width, uint32_t height );
        bool NeedsResize( uint32_t width, uint32_t height ) const;

        void TransitionForRendering( CommandBuffer* cmd );
        void TransitionForSampling( CommandBuffer* cmd );

        TextureHandle   GetColorTexture() const { return m_colorHandle; }
        TextureHandle   GetDepthTexture() const { return m_depthHandle; }

        uint32_t GetWidth() const { return m_width; }
        uint32_t GetHeight() const { return m_height; }

    private:
        void CreateResources();
        void DestroyResources();

    private:
        ResourceManager* m_resourceManager;
        uint32_t         m_width;
        uint32_t         m_height;

        TextureHandle m_colorHandle;
        TextureHandle m_depthHandle;
    };
} // namespace DigitalTwin