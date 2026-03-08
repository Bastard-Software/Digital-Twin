#include "renderer/Renderer.h"

#include "core/Log.h"
#include "platform/Window.h"
#include "resources/ResourceManager.h"
#include "resources/StreamingManager.h"
#include "rhi/Buffer.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/GPUProfiler.h"
#include "rhi/Sampler.h"
#include "rhi/Swapchain.h"
#include "rhi/Texture.h"
#include "simulation/SimulationState.h"
#include <volk.h>

// Rendering Systems
#include "renderer/Camera.h"
#include "renderer/RenderTarget.h"
#include "renderer/Scene.h"
#include "renderer/passes/GeometryPass.h"

// ImGui
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

#if defined( max )
#    undef max
#endif

namespace DigitalTwin
{
    Renderer::Renderer( Device* device, Swapchain* swapchain, ResourceManager* rm, StreamingManager* sm )
        : m_device( device )
        , m_swapchain( swapchain )
        , m_resourceManager( rm )
        , m_streamingManager( sm )
    {
    }

    Renderer::~Renderer()
    {
    }

    Result Renderer::Create()
    {
        DT_INFO( "Initializing Renderer" );

        // 1. Create Default Sampler
        SamplerDesc samplerDesc;
        samplerDesc.minFilter = VK_FILTER_LINEAR;
        samplerDesc.magFilter = VK_FILTER_LINEAR;
        m_defaultSampler      = m_resourceManager->CreateSampler( samplerDesc );

        // 2. Create Descriptor Pool for ImGui
        VkDescriptorPoolSize pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
                                              { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
                                              { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
                                              { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
                                              { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
                                              { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
                                              { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
                                              { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
                                              { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
                                              { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
                                              { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 } };

        VkDescriptorPoolCreateInfo pool_info = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets                    = 1000;
        pool_info.poolSizeCount              = ( uint32_t )std::size( pool_sizes );
        pool_info.pPoolSizes                 = pool_sizes;

        const auto&      api = m_device->GetAPI();
        VkDescriptorPool pool;
        if( api.vkCreateDescriptorPool( m_device->GetHandle(), &pool_info, nullptr, &pool ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to create ImGui Descriptor Pool." );
            return Result::FAIL;
        }
        m_imguiDescriptorPool = pool;

        // 3. Init ImGui Context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        ImGui::StyleColorsDark();

        if( !ImGui_ImplGlfw_InitForVulkan( ( GLFWwindow* )m_swapchain->GetWindow()->GetNativeWindow(), true ) )
        {
            DT_ERROR( "Failed to init ImGui for glfw." );
            return Result::FAIL;
        }

        // Load Vulkan functions for ImGui
        VkInstance instance = m_device->GetInstance();
        ImGui_ImplVulkan_LoadFunctions(
            VK_API_VERSION_1_3, []( const char* name, void* ud ) { return vkGetInstanceProcAddr( ( VkInstance )ud, name ); }, instance );

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance                  = m_device->GetInstance();
        init_info.PhysicalDevice            = m_device->GetPhysicalDevice();
        init_info.Device                    = m_device->GetHandle();
        init_info.QueueFamily               = m_device->GetGraphicsQueue()->GetFamilyIndex();
        init_info.Queue                     = m_device->GetGraphicsQueue()->GetHandle();
        init_info.DescriptorPool            = ( VkDescriptorPool )m_imguiDescriptorPool;
        init_info.MinImageCount             = 2;
        init_info.ImageCount                = 2;
        init_info.ApiVersion                = VK_API_VERSION_1_3;
        init_info.UseDynamicRendering       = true;

        VkFormat                         format                = m_swapchain->GetFormat();
        VkPipelineRenderingCreateInfoKHR pipelineInfo          = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
        pipelineInfo.colorAttachmentCount                      = 1;
        pipelineInfo.pColorAttachmentFormats                   = &format;
        pipelineInfo.depthAttachmentFormat                     = VK_FORMAT_UNDEFINED;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineInfo;

        if( !ImGui_ImplVulkan_Init( &init_info ) )
        {
            DT_ERROR( "Failed to init ImGui for Vulkan." );
            return Result::FAIL;
        }

        // Note: Font upload is handled automatically by ImGui 1.91+ in NewFrame()

        // 4. Initialize Per-Frame Resources (Render Targets and UBOs)
        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            m_renderTargets[ i ] = CreateScope<RenderTarget>( m_resourceManager, m_viewportWidth, m_viewportHeight );

            // Uniform Buffer for Camera Matrices (Host Visible so we can update it easily)
            BufferDesc uboDesc{ sizeof( glm::mat4 ), BufferType::UNIFORM };
            m_cameraUBOs[ i ] = m_resourceManager->CreateBuffer( uboDesc );
        }

        // 5. Initialize Render Passes
        m_geometryPass = CreateScope<GeometryPass>( m_device, m_resourceManager );
        if( m_geometryPass->Initialize() != Result::SUCCESS )
        {
            DT_ERROR( "Failed to initialize GeometryPass." );
            return Result::FAIL;
        }

        return Result::SUCCESS;
    }

    void Renderer::Destroy()
    {
        m_device->WaitIdle();

        // 1. Cleanup Per-Frame Resources
        for( uint32_t i = 0; i < FRAMES_IN_FLIGHT; ++i )
        {
            // Remove cached ImGui textures properly
            if( m_cachedImGuiTextures[ i ] )
            {
                ImGui_ImplVulkan_RemoveTexture( ( VkDescriptorSet )m_cachedImGuiTextures[ i ] );
                m_cachedImGuiTextures[ i ] = nullptr;
            }

            m_resourceManager->DestroyBuffer( m_cameraUBOs[ i ] );
            m_renderTargets[ i ].reset();
        }

        // 2. Cleanup Passes
        if( m_geometryPass )
        {
            m_geometryPass->Shutdown();
            m_geometryPass.reset();
        }

        // 3. Cleanup ImGui
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        const auto& api = m_device->GetAPI();
        if( m_imguiDescriptorPool )
        {
            api.vkResetDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, 0 );
            api.vkDestroyDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, nullptr );
        }

        m_resourceManager->DestroySampler( m_defaultSampler );
    }

    void Renderer::BeginUI()
    {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Setup a full-screen dockspace
        ImGuiDockNodeFlags dockFlags = ImGuiDockNodeFlags_PassthruCentralNode;
        ImGui::DockSpaceOverViewport( 0, ImGui::GetMainViewport(), dockFlags );
    }

    void Renderer::EndUI()
    {
        ImGui::Render();
    }

    void Renderer::RenderUI( std::function<void()> callback )
    {
        if( callback )
            callback();
    }

    void Renderer::SetViewportSize( uint32_t width, uint32_t height )
    {
        // Prevent zero-sized viewports
        m_viewportWidth  = std::max( width, 1u );
        m_viewportHeight = std::max( height, 1u );
    }

    void* Renderer::GetSceneTextureID( uint32_t flightIndex )
    {
        // 1. Resize RenderTarget if requested
        if( m_renderTargets[ flightIndex ]->NeedsResize( m_viewportWidth, m_viewportHeight ) )
        {
            // Remove the old descriptor from ImGui to prevent leaks
            if( m_cachedImGuiTextures[ flightIndex ] )
            {
                ImGui_ImplVulkan_RemoveTexture( ( VkDescriptorSet )m_cachedImGuiTextures[ flightIndex ] );
            }

            // Recreate color/depth textures
            m_renderTargets[ flightIndex ]->Resize( m_viewportWidth, m_viewportHeight );

            // Add new descriptor
            Texture* tex     = m_resourceManager->GetTexture( m_renderTargets[ flightIndex ]->GetColorTexture() );
            Sampler* sampler = m_resourceManager->GetSampler( m_defaultSampler );
            m_cachedImGuiTextures[ flightIndex ] =
                ImGui_ImplVulkan_AddTexture( sampler->GetHandle(), tex->GetView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
        }
        else if( !m_cachedImGuiTextures[ flightIndex ] )
        {
            // First time initialization
            Texture* tex     = m_resourceManager->GetTexture( m_renderTargets[ flightIndex ]->GetColorTexture() );
            Sampler* sampler = m_resourceManager->GetSampler( m_defaultSampler );
            m_cachedImGuiTextures[ flightIndex ] =
                ImGui_ImplVulkan_AddTexture( sampler->GetHandle(), tex->GetView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
        }

        return m_cachedImGuiTextures[ flightIndex ];
    }

    void* Renderer::GetImGuiContext()
    {
        return ImGui::GetCurrentContext();
    }

    void Renderer::RenderSimulation( CommandBuffer* cmd, SimulationState* state, Camera* camera, uint32_t flightIndex )
    {
        GPUProfiler* profiler = m_device->GetProfiler();
        if( profiler )
            profiler->BeginFrame( cmd, flightIndex );

        // 1. Update Camera Uniform Buffer
        Buffer*   ubo      = m_resourceManager->GetBuffer( m_cameraUBOs[ flightIndex ] );
        glm::mat4 viewProj = camera->GetViewProjection();
        ubo->Write( &viewProj, sizeof( glm::mat4 ) );

        // 2. Prepare RenderTarget textures for writing
        m_renderTargets[ flightIndex ]->TransitionForRendering( cmd );

        // 3. Build Rendering Info Safely (Local variables, pointers valid during cmd call)
        Texture* colorTex = m_resourceManager->GetTexture( m_renderTargets[ flightIndex ]->GetColorTexture() );
        Texture* depthTex = m_resourceManager->GetTexture( m_renderTargets[ flightIndex ]->GetDepthTexture() );

        VkRenderingAttachmentInfo colorAttachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView                 = colorTex->GetView();
        colorAttachment.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue                = { { 0.1f, 0.1f, 0.15f, 1.0f } };

        VkRenderingAttachmentInfo depthAttachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        depthAttachment.imageView                 = depthTex->GetView();
        depthAttachment.imageLayout               = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.clearValue.depthStencil   = { 1.0f, 0 };

        VkRenderingInfo renderInfo      = { VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderInfo.renderArea           = { { 0, 0 }, { m_renderTargets[ flightIndex ]->GetWidth(), m_renderTargets[ flightIndex ]->GetHeight() } };
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;
        renderInfo.pDepthAttachment     = &depthAttachment;

        // 4. Begin Dynamic Rendering
        cmd->BeginRendering( renderInfo );

        cmd->SetViewport( 0.0f, 0.0f, ( float )m_renderTargets[ flightIndex ]->GetWidth(), ( float )m_renderTargets[ flightIndex ]->GetHeight() );
        cmd->SetScissor( 0, 0, m_renderTargets[ flightIndex ]->GetWidth(), m_renderTargets[ flightIndex ]->GetHeight() );

        // 4. Execute passes
        if( state != nullptr && state->IsValid() )
        {
            // TODO: Frustum and occlusion culling that create scene
            Scene renderScene;
            renderScene.vertexBuffer      = state->vertexBuffer;
            renderScene.indexBuffer       = state->indexBuffer;
            renderScene.indirectCmdBuffer = state->indirectCmdBuffer;
            renderScene.groupDataBuffer   = state->groupDataBuffer;
            renderScene.agentBuffers[ 0 ] = state->agentBuffers[ 0 ];
            renderScene.agentBuffers[ 1 ] = state->agentBuffers[ 0 ];
            renderScene.drawCount         = state->groupCount;

            if( profiler )
                profiler->BeginZone( cmd, flightIndex, "Geometry Pass" );
            m_geometryPass->Execute( cmd, m_cameraUBOs[ flightIndex ], &renderScene, flightIndex );
            if( profiler )
                profiler->EndZone( cmd, flightIndex, "Geometry Pass" );
        }

        cmd->EndRendering();

        // 5. Prepare RenderTarget color texture for reading in ImGui
        m_renderTargets[ flightIndex ]->TransitionForSampling( cmd );
    }

    void Renderer::RecordUIPass( CommandBuffer* cmd, Texture* backbuffer )
    {
        // 1. Setup UI Rendering Attachment
        VkRenderingAttachmentInfo colorAttachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView                 = backbuffer->GetView();
        colorAttachment.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue                = { { 0.05f, 0.05f, 0.05f, 1.0f } };

        VkRenderingInfo renderInfo      = { VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderInfo.renderArea           = { { 0, 0 }, { backbuffer->GetExtent().width, backbuffer->GetExtent().height } };
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;

        // 2. Transition Swapchain Image to Color Attachment Optimal
        VkImageMemoryBarrier2 barrier1 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier1.srcStageMask          = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        barrier1.srcAccessMask         = 0;
        barrier1.dstStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier1.dstAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier1.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier1.newLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier1.image                 = backbuffer->GetHandle();
        barrier1.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier1 );

        // 3. Record Draw Commands
        cmd->BeginRendering( renderInfo );

        ImDrawData* draw_data = ImGui::GetDrawData();
        if( draw_data )
        {
            ImGui_ImplVulkan_RenderDrawData( draw_data, cmd->GetHandle() );
        }

        cmd->EndRendering();

        // 4. Transition Swapchain Image to Present
        VkImageMemoryBarrier2 barrier2 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        barrier2.srcStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        barrier2.srcAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        barrier2.dstStageMask          = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        barrier2.dstAccessMask         = 0;
        barrier2.oldLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        barrier2.newLayout             = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier2.image                 = backbuffer->GetHandle();
        barrier2.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        cmd->PipelineBarrier( 0, 0, 0, 0, nullptr, 0, nullptr, 1, &barrier2 );
    }

} // namespace DigitalTwin