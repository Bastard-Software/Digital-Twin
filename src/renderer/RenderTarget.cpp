#include "renderer/RenderTarget.h"

#include "resources/ResourceManager.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Texture.h"

namespace DigitalTwin
{
    RenderTarget::RenderTarget( ResourceManager* rm, uint32_t width, uint32_t height,
                                VkSampleCountFlagBits sampleCount )
        : m_resourceManager( rm )
        , m_width( width )
        , m_height( height )
        , m_sampleCount( sampleCount )
    {
        CreateResources();
    }

    RenderTarget::~RenderTarget()
    {
        DestroyResources();
    }

    void RenderTarget::Resize( uint32_t width, uint32_t height )
    {
        Resize( width, height, m_sampleCount );
    }

    void RenderTarget::Resize( uint32_t width, uint32_t height, VkSampleCountFlagBits sampleCount )
    {
        if( width == 0 || height == 0 )
            return;
        if( width == m_width && height == m_height && sampleCount == m_sampleCount )
            return;

        m_width       = width;
        m_height      = height;
        m_sampleCount = sampleCount;
        DestroyResources();
        CreateResources();
    }

    bool RenderTarget::NeedsResize( uint32_t width, uint32_t height ) const
    {
        return ( width != m_width || height != m_height ) && width > 0 && height > 0;
    }

    void RenderTarget::CreateResources()
    {
        if( m_sampleCount == VK_SAMPLE_COUNT_1_BIT )
        {
            // ---------------------------------------------------------------
            // Single-sample path (original behaviour)
            // ---------------------------------------------------------------
            TextureDesc colorDesc;
            colorDesc.width  = m_width;
            colorDesc.height = m_height;
            colorDesc.format = VK_FORMAT_R8G8B8A8_UNORM;
            colorDesc.usage  = TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED;
            m_colorHandle    = m_resourceManager->CreateTexture( colorDesc );

            TextureDesc depthDesc;
            depthDesc.width  = m_width;
            depthDesc.height = m_height;
            depthDesc.format = VK_FORMAT_D32_SFLOAT;
            depthDesc.usage  = TextureUsage::DEPTH_STENCIL_TARGET;
            m_depthHandle    = m_resourceManager->CreateTexture( depthDesc );
        }
        else
        {
            // ---------------------------------------------------------------
            // MSAA path
            // Multisampled images cannot be sampled by shaders, so we keep a
            // separate single-sample resolved colour image for ImGui.
            // ---------------------------------------------------------------
            TextureDesc msaaColorDesc;
            msaaColorDesc.width       = m_width;
            msaaColorDesc.height      = m_height;
            msaaColorDesc.format      = VK_FORMAT_R8G8B8A8_UNORM;
            msaaColorDesc.usage       = TextureUsage::RENDER_TARGET;
            msaaColorDesc.sampleCount = m_sampleCount;
            m_msaaColorHandle         = m_resourceManager->CreateTexture( msaaColorDesc );

            TextureDesc msaaDepthDesc;
            msaaDepthDesc.width       = m_width;
            msaaDepthDesc.height      = m_height;
            msaaDepthDesc.format      = VK_FORMAT_D32_SFLOAT;
            msaaDepthDesc.usage       = TextureUsage::DEPTH_STENCIL_TARGET;
            msaaDepthDesc.sampleCount = m_sampleCount;
            m_msaaDepthHandle         = m_resourceManager->CreateTexture( msaaDepthDesc );

            TextureDesc resolveDesc;
            resolveDesc.width       = m_width;
            resolveDesc.height      = m_height;
            resolveDesc.format      = VK_FORMAT_R8G8B8A8_UNORM;
            resolveDesc.usage       = TextureUsage::RENDER_TARGET | TextureUsage::SAMPLED;
            resolveDesc.sampleCount = VK_SAMPLE_COUNT_1_BIT;
            m_resolvedColorHandle   = m_resourceManager->CreateTexture( resolveDesc );
        }
    }

    void RenderTarget::DestroyResources()
    {
        // Single-sample path
        if( m_colorHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_colorHandle );
        if( m_depthHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_depthHandle );

        // MSAA path
        if( m_msaaColorHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_msaaColorHandle );
        if( m_msaaDepthHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_msaaDepthHandle );
        if( m_resolvedColorHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_resolvedColorHandle );
    }

    // ---------------------------------------------------------------------------
    // Attachment view accessors
    // ---------------------------------------------------------------------------

    VkImageView RenderTarget::GetColorAttachmentView() const
    {
        if( m_sampleCount > VK_SAMPLE_COUNT_1_BIT )
            return m_resourceManager->GetTexture( m_msaaColorHandle )->GetView();
        return m_resourceManager->GetTexture( m_colorHandle )->GetView();
    }

    VkImageView RenderTarget::GetResolveAttachmentView() const
    {
        if( m_sampleCount > VK_SAMPLE_COUNT_1_BIT )
            return m_resourceManager->GetTexture( m_resolvedColorHandle )->GetView();
        return VK_NULL_HANDLE;
    }

    VkImageView RenderTarget::GetDepthAttachmentView() const
    {
        if( m_sampleCount > VK_SAMPLE_COUNT_1_BIT )
            return m_resourceManager->GetTexture( m_msaaDepthHandle )->GetView();
        return m_resourceManager->GetTexture( m_depthHandle )->GetView();
    }

    TextureHandle RenderTarget::GetSampledTexture() const
    {
        if( m_sampleCount > VK_SAMPLE_COUNT_1_BIT )
            return m_resolvedColorHandle;
        return m_colorHandle;
    }

    // ---------------------------------------------------------------------------
    // Barriers
    // ---------------------------------------------------------------------------

    void RenderTarget::TransitionForRendering( CommandBuffer* cmd )
    {
        if( m_sampleCount > VK_SAMPLE_COUNT_1_BIT )
        {
            Texture* msaaColor    = m_resourceManager->GetTexture( m_msaaColorHandle );
            Texture* msaaDepth    = m_resourceManager->GetTexture( m_msaaDepthHandle );
            Texture* resolveColor = m_resourceManager->GetTexture( m_resolvedColorHandle );

            // MSAA colour: transition to colour attachment for rendering
            VkImageMemoryBarrier2 msaaColorBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            msaaColorBarrier.srcStageMask           = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            msaaColorBarrier.srcAccessMask          = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            msaaColorBarrier.dstStageMask           = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            msaaColorBarrier.dstAccessMask          = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            msaaColorBarrier.oldLayout              = VK_IMAGE_LAYOUT_UNDEFINED; // discard previous contents
            msaaColorBarrier.newLayout              = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            msaaColorBarrier.image                  = msaaColor->GetHandle();
            msaaColorBarrier.subresourceRange       = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            // Resolve target: must also be in colour attachment layout for implicit resolve
            VkImageMemoryBarrier2 resolveBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            resolveBarrier.srcStageMask          = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            resolveBarrier.srcAccessMask         = VK_ACCESS_2_SHADER_READ_BIT;
            resolveBarrier.dstStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            resolveBarrier.dstAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            resolveBarrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED; // discard previous contents
            resolveBarrier.newLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            resolveBarrier.image                 = resolveColor->GetHandle();
            resolveBarrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            // MSAA depth
            VkImageMemoryBarrier2 depthBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            depthBarrier.srcStageMask          = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            depthBarrier.srcAccessMask         = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depthBarrier.dstStageMask          = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            depthBarrier.dstAccessMask         = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            depthBarrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
            depthBarrier.newLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthBarrier.image                 = msaaDepth->GetHandle();
            depthBarrier.subresourceRange      = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

            VkImageMemoryBarrier2 barriers[] = { msaaColorBarrier, resolveBarrier, depthBarrier };
            cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 3, barriers );
        }
        else
        {
            // Single-sample path (original)
            Texture* color = m_resourceManager->GetTexture( m_colorHandle );
            Texture* depth = m_resourceManager->GetTexture( m_depthHandle );

            VkImageMemoryBarrier2 colorBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            colorBarrier.srcStageMask          = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            colorBarrier.srcAccessMask         = VK_ACCESS_2_SHADER_READ_BIT;
            colorBarrier.dstStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            colorBarrier.dstAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            colorBarrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
            colorBarrier.newLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            colorBarrier.image                 = color->GetHandle();
            colorBarrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkImageMemoryBarrier2 depthBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            depthBarrier.srcStageMask          = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            depthBarrier.srcAccessMask         = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            depthBarrier.dstStageMask          = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
            depthBarrier.dstAccessMask         = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
            depthBarrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
            depthBarrier.newLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            depthBarrier.image                 = depth->GetHandle();
            depthBarrier.subresourceRange      = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };

            VkImageMemoryBarrier2 barriers[] = { colorBarrier, depthBarrier };
            cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 2, barriers );
        }
    }

    void RenderTarget::TransitionForSampling( CommandBuffer* cmd )
    {
        // Always transition the texture that ImGui will sample.
        // When MSAA is on that is the resolved colour (single-sample); the
        // MSAA colour attachment is never sampled directly.
        Texture* sampled = m_resourceManager->GetTexture( GetSampledTexture() );

        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask          = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask         = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout             = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image                 = sampled->GetHandle();
        barrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier );
    }
} // namespace DigitalTwin
