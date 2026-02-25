#include "renderer/RenderTarget.h"

#include "resources/ResourceManager.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Texture.h"

namespace DigitalTwin
{
    RenderTarget::RenderTarget( ResourceManager* rm, uint32_t width, uint32_t height )
        : m_resourceManager( rm )
        , m_width( width )
        , m_height( height )
    {
        CreateResources();
    }

    RenderTarget::~RenderTarget()
    {
        DestroyResources();
    }

    void RenderTarget::Resize( uint32_t width, uint32_t height )
    {
        if( width == 0 || height == 0 || ( width == m_width && height == m_height ) )
            return;
        m_width  = width;
        m_height = height;
        DestroyResources();
        CreateResources();
    }

    bool RenderTarget::NeedsResize( uint32_t width, uint32_t height ) const
    {
        return ( width != m_width || height != m_height ) && width > 0 && height > 0;
    }

    void RenderTarget::CreateResources()
    {
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

    void RenderTarget::DestroyResources()
    {
        if( m_colorHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_colorHandle );
        if( m_depthHandle.IsValid() )
            m_resourceManager->DestroyTexture( m_depthHandle );
    }

    void RenderTarget::TransitionForRendering( CommandBuffer* cmd )
    {
        Texture* color = m_resourceManager->GetTexture( m_colorHandle );
        Texture* depth = m_resourceManager->GetTexture( m_depthHandle );

        VkImageMemoryBarrier2 colorBarrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        colorBarrier.srcStageMask          = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        colorBarrier.srcAccessMask         = VK_ACCESS_2_SHADER_READ_BIT;
        colorBarrier.dstStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        colorBarrier.dstAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        colorBarrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED; // We don't care about previous contents
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

    void RenderTarget::TransitionForSampling( CommandBuffer* cmd )
    {
        Texture* color = m_resourceManager->GetTexture( m_colorHandle );

        VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier.srcStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier.srcAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstStageMask          = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask         = VK_ACCESS_2_SHADER_READ_BIT;
        barrier.oldLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier.newLayout             = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.image                 = color->GetHandle();
        barrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier );
    }
} // namespace DigitalTwin