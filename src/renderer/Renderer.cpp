#include "renderer/Renderer.hpp"

#include "runtime/Engine.hpp"

namespace DigitalTwin
{

    Renderer::Renderer( Engine& engine )
        : m_resManager( engine.GetResourceManager() )
    {
        if( engine.IsHeadless() )
        {
            m_active = false;
            return;
        }

        m_active = true;
        m_ctx    = CreateScope<RenderContext>( engine.GetDevice(), engine.GetWindow() );
        m_ctx->Init();

        m_simPass = CreateScope<AgentRenderPass>( engine.GetDevice(), m_resManager );
        m_simPass->Init( m_ctx->GetColorFormat(), m_ctx->GetDepthFormat() );

        m_camera = CreateScope<Camera>( 45.0f, static_cast<float>( engine.GetConfig().width ) / static_cast<float>( engine.GetConfig().height ), 0.1f,
                                        1000.0f );
    }

    Renderer::~Renderer()
    {
        if( m_ctx )
            m_ctx->Shutdown();
    }

    void Renderer::Render( const Scene& scene, const std::vector<VkSemaphore>& waitSems, const std::vector<uint64_t>& waitVals )
    {
        if( !m_active )
            return;

        CommandBuffer* cmd = m_ctx->BeginFrame();
        if( !cmd )
            return;

        auto extent = m_ctx->GetSwapchain()->GetExtent();

        // Dynamic Rendering using CommandBuffer Wrapper
        RenderingInfo renderInfo;
        renderInfo.renderArea = { { 0, 0 }, { extent.width, extent.height } };

        RenderingAttachmentInfo colorAtt;
        colorAtt.imageView  = m_ctx->GetSwapchain()->GetImageView( m_ctx->GetCurrentImageIndex() );
        colorAtt.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.clearValue = { 0.1f, 0.1f, 0.1f, 1.0f }; // Dark gray background
        renderInfo.colorAttachments.push_back( colorAtt );

        // Depth (Disabled for now as we don't have depth buffer in Swapchain yet)
        renderInfo.useDepth = true;
        RenderingAttachmentInfo depthAtt;
        depthAtt.imageView = m_ctx->GetDepthTexture()->GetView(); // Get the view from our new texture

        // Ideally, we should transition this layout. For now, we assume the initial transition or do it here.
        // We write to it, so use DEPTH_ATTACHMENT_OPTIMAL.
        depthAtt.layout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

        // Clear depth at the start of the frame (1.0 = Far Plane)
        depthAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // Store is usually DONT_CARE if we don't need it later, but STORE is safer for debug
        depthAtt.storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.clearValue = { 1.0f, 0 }; // Depth 1.0, Stencil 0

        renderInfo.depthAttachment = depthAtt;

        cmd->BeginRendering( renderInfo );

        cmd->SetViewport( 0.0f, 0.0f, ( float )extent.width, ( float )extent.height, 0.0f, 1.0f );
        cmd->SetScissor( 0, 0, extent.width, extent.height );

        m_simPass->Draw( cmd, scene );

        cmd->EndRendering();

        m_ctx->EndFrame( waitSems, waitVals );
    }

    void Renderer::OnUpdate( float dt )
    {
        if( m_active )
            m_camera->OnUpdate( dt );
    }

    void Renderer::OnResize( uint32_t w, uint32_t h )
    {
        if( m_active )
        {
            m_ctx->OnResize( w, h );
            m_camera->OnResize( w, h );
        }
    }
} // namespace DigitalTwin