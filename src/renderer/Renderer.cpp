#include "renderer/Renderer.hpp"

#include "core/Log.hpp"
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

        m_gui = CreateScope<ImGuiLayer>( engine.GetDevice(), engine.GetWindow(), m_ctx->GetColorFormat() );

        // AgentRenderPass uses Viewport format (RGBA8), not Swapchain format
        m_simPass = CreateScope<AgentRenderPass>( engine.GetDevice(), m_resManager );
        m_simPass->Init( VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT );

        m_camera = CreateScope<Camera>( 45.0f, 1.77f, 0.1f, 1000.0f );

        UpdateImGuiTextures();
    }

    Renderer::~Renderer()
    {
        if( m_ctx )
            m_ctx->Shutdown();
        m_gui.reset();
    }

    void Renderer::UpdateImGuiTextures()
    {
        if( !m_gui || !m_ctx )
            return;

        // Clean up old descriptors to prevent pool exhaustion (Memory Leak Fix)
        for( auto id: m_viewportTextureIDs )
        {
            m_gui->RemoveTexture( id );
        }
        m_viewportTextureIDs.clear();

        // Register new textures
        const auto& textures = m_ctx->GetAllViewportTextures();
        auto        sampler  = m_ctx->GetViewportSampler();

        for( const auto& tex: textures )
        {
            m_viewportTextureIDs.push_back( m_gui->AddTexture( tex, sampler ) );
        }
    }

    ImTextureID Renderer::GetViewportTextureID()
    {
        uint32_t idx = m_ctx->GetCurrentFrameIndex();
        if( idx < m_viewportTextureIDs.size() )
            return m_viewportTextureIDs[ idx ];
        return ( ImTextureID )0;
    }

    void Renderer::ResizeViewport( uint32_t width, uint32_t height )
    {
        if( m_ctx->OnResizeViewport( width, height ) )
        {
            UpdateImGuiTextures();
            m_camera->OnResize( width, height );
        }
    }

    void Renderer::RenderSimulation( const Scene& scene )
    {
        if( !m_active )
            return;

        // Retrieve the active command buffer wrapper (Ref<CommandBuffer>)
        auto cmd = m_ctx->GetActiveCommandBuffer();

        auto colorTex = m_ctx->GetViewportTexture();
        auto depthTex = m_ctx->GetViewportDepth();
        auto extent   = colorTex->GetExtent();

        // 1. Transition Offscreen Texture to Color Attachment Layout
        // This is required before we can render into it.
        // The Texture class handles the specific barrier details.
        colorTex->TransitionLayout( cmd, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL );

        // --- Render to Offscreen Texture ---
        RenderingAttachmentInfo colorAtt;
        colorAtt.imageView  = colorTex->GetView();
        colorAtt.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.clearValue = { 0.1f, 0.1f, 0.1f, 1.0f };

        RenderingAttachmentInfo depthAtt;
        depthAtt.imageView  = depthTex->GetView();
        depthAtt.layout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAtt.loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAtt.storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
        depthAtt.clearValue = { 1.0f, 0 };

        RenderingInfo renderInfo     = {};
        renderInfo.renderArea.extent = { extent.width, extent.height };
        renderInfo.renderArea.offset = { 0, 0 };
        renderInfo.colorAttachments.push_back( colorAtt );
        renderInfo.depthAttachment = depthAtt;
        renderInfo.useDepth        = true;

        cmd->BeginRendering( renderInfo );

        // Update Dynamic State
        cmd->SetViewport( 0.0f, 0.0f, static_cast<float>( extent.width ), static_cast<float>( extent.height ), 0.0f, 1.0f );
        cmd->SetScissor( 0, 0, extent.width, extent.height );

        // Draw Scene
        m_simPass->Draw( cmd.get(), scene );

        cmd->EndRendering();

        // 2. Transition Offscreen Texture to Shader Read Only Layout
        // This prepares the texture to be sampled by ImGui in the next pass.
        colorTex->TransitionLayout( cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
    }

    void Renderer::RenderUI( const std::vector<VkSemaphore>& waitSems, const std::vector<uint64_t>& waitVals )
    {
        if( !m_active )
            return;

        // Retrieve raw command buffer for internal calls, and Ref for transitions if needed
        auto cmd       = m_ctx->GetActiveCommandBuffer();
        auto swapchain = m_ctx->GetSwapchain();
        auto extent    = swapchain->GetExtent();

        // Get the current swapchain image index and the raw image handle
        uint32_t imgIndex  = m_ctx->GetCurrentImageIndex();
        VkImage  swapImage = swapchain->GetImage( imgIndex );

        // --- BARRIER 1: Undefined -> Color Attachment ---
        // We must explicitly transition the swapchain image from its initial state (Undefined)
        // to Color Attachment Optimal so we can render the UI onto it.
        cmd->TransitionImageLayout( swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL );

        // --- Render UI to Swapchain ---
        RenderingAttachmentInfo colorAtt;
        colorAtt.imageView  = swapchain->GetImageView( imgIndex );
        colorAtt.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAtt.loadOp     = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAtt.storeOp    = VK_ATTACHMENT_STORE_OP_STORE;
        colorAtt.clearValue = { 0.0f, 0.0f, 0.0f, 1.0f };

        RenderingInfo renderInfo     = {};
        renderInfo.renderArea.extent = extent;
        renderInfo.renderArea.offset = { 0, 0 };
        renderInfo.colorAttachments.push_back( colorAtt );

        cmd->BeginRendering( renderInfo );

        if( m_gui )
            m_gui->End( cmd.get() );

        cmd->EndRendering();

        // --- BARRIER 2: Color Attachment -> Present Src ---
        // CRITICAL: Transition the image to Present Src KHR layout before handing it over to the presentation engine.
        // Failing to do this causes validation errors (VUID-VkPresentInfoKHR-pImageIndices-01430).
        cmd->TransitionImageLayout( swapImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR );

        // Submit and Present
        m_ctx->EndFrame( waitSems, waitVals );
    }

    void Renderer::OnUpdate( float dt )
    {
        if( m_camera )
            m_camera->OnUpdate( dt );
    }
} // namespace DigitalTwin