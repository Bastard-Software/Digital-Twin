#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>

namespace DigitalTwin
{
    class ResourceManager;

    /**
     * @brief Off-screen render target.
     *
     * When sampleCount == VK_SAMPLE_COUNT_1_BIT (default) the layout is identical to the
     * original: one colour attachment (RENDER_TARGET | SAMPLED) + one depth attachment.
     *
     * When sampleCount == VK_SAMPLE_COUNT_4_BIT the layout becomes:
     *   m_msaaColorHandle    — multisampled colour, RENDER_TARGET only (never sampled directly)
     *   m_msaaDepthHandle    — multisampled depth,  DEPTH_STENCIL_TARGET
     *   m_resolvedColorHandle — single-sample colour, RENDER_TARGET | SAMPLED (resolve target)
     *
     * The resolve is performed implicitly at vkCmdEndRendering via pResolveAttachments
     * (VK_RESOLVE_MODE_AVERAGE_BIT) — no extra blit pass needed.
     */
    class RenderTarget
    {
    public:
        RenderTarget( ResourceManager* rm, uint32_t width, uint32_t height,
                      VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT );
        ~RenderTarget();

        void Resize( uint32_t width, uint32_t height );
        void Resize( uint32_t width, uint32_t height, VkSampleCountFlagBits sampleCount );
        bool NeedsResize( uint32_t width, uint32_t height ) const;

        void TransitionForRendering( CommandBuffer* cmd );
        void TransitionForSampling( CommandBuffer* cmd );

        // -------------------------------------------------------------------
        // Attachment views — used by Renderer to fill VkRenderingAttachmentInfo
        // -------------------------------------------------------------------

        /// View to render into: multisampled colour (MSAA on) or plain colour (MSAA off).
        VkImageView GetColorAttachmentView() const;

        /// Resolve-target view when MSAA is on; VK_NULL_HANDLE when MSAA is off.
        VkImageView GetResolveAttachmentView() const;

        /// Depth view (multisampled when MSAA on, single-sample when MSAA off).
        VkImageView GetDepthAttachmentView() const;

        // -------------------------------------------------------------------
        // Sampled texture — used by ImGui to display the scene
        // -------------------------------------------------------------------

        /// Returns the TextureHandle that ImGui/descriptor-set should sample:
        ///   MSAA on  → resolvedColor (single-sample)
        ///   MSAA off → plain colour
        TextureHandle GetSampledTexture() const;

        VkSampleCountFlagBits GetSampleCount() const { return m_sampleCount; }
        uint32_t              GetWidth() const { return m_width; }
        uint32_t              GetHeight() const { return m_height; }


    private:
        void CreateResources();
        void DestroyResources();

    private:
        ResourceManager*      m_resourceManager;
        uint32_t              m_width;
        uint32_t              m_height;
        VkSampleCountFlagBits m_sampleCount;

        // Single-sample path (sampleCount == VK_SAMPLE_COUNT_1_BIT)
        TextureHandle m_colorHandle; // RENDER_TARGET | SAMPLED
        TextureHandle m_depthHandle; // DEPTH_STENCIL_TARGET

        // MSAA path (sampleCount > VK_SAMPLE_COUNT_1_BIT)
        TextureHandle m_msaaColorHandle;     // RENDER_TARGET (multi-sample, not sampled)
        TextureHandle m_msaaDepthHandle;     // DEPTH_STENCIL_TARGET (multi-sample)
        TextureHandle m_resolvedColorHandle; // RENDER_TARGET | SAMPLED (single-sample resolve target)
    };
} // namespace DigitalTwin
