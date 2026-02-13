#include "renderer/Renderer.h"

#include "core/Log.h"
#include "platform/Window.h"
#include "resources/ResourceManager.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Device.h"
#include "rhi/Sampler.h"
#include "rhi/Swapchain.h"
#include "rhi/Texture.h"
#include <Volk/volk.h>

// ImGui
#include <GLFW/glfw3.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <imgui.h>

namespace DigitalTwin
{

    Renderer::Renderer( Device* device, Swapchain* swapchain, ResourceManager* rm )
        : m_device( device )
        , m_swapchain( swapchain )
        , m_resourceManager( rm )
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

        // 2. Create Descriptor Pool
        VkDescriptorPoolSize       pool_sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
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
        VkDescriptorPoolCreateInfo pool_info    = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        pool_info.flags                         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool_info.maxSets                       = 1000;
        pool_info.poolSizeCount                 = ( uint32_t )std::size( pool_sizes );
        pool_info.pPoolSizes                    = pool_sizes;

        const auto&      api = m_device->GetAPI();
        VkDescriptorPool pool;

        if( api.vkCreateDescriptorPool( m_device->GetHandle(), &pool_info, nullptr, &pool ) != VK_SUCCESS )
        {
            DT_ERROR( "Failed to create ImGui Descriptor Pool." );
            return Result::FAIL;
        }
        m_imguiDescriptorPool = pool;

        // 3. Init ImGui
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

        ImGui::StyleColorsDark();

        if( !ImGui_ImplGlfw_InitForVulkan( ( GLFWwindow* )m_swapchain->GetWindow()->GetNativeWindow(), true ) )
        {
            DT_ASSERT( false, "" );
            DT_ERROR( "Failed to init ImGui for glfw." );
            api.vkResetDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, 0 );
            api.vkDestroyDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, nullptr );
            return Result::FAIL;
        }

        VkInstance instance = m_device->GetInstance();
        if( !ImGui_ImplVulkan_LoadFunctions(
                VK_API_VERSION_1_3,
                []( const char* function_name, void* user_data ) {
                    VkInstance instance = static_cast<VkInstance>( user_data );
                    return vkGetInstanceProcAddr( instance, function_name ); // Use volk directly
                },
                instance ) )
        {
            DT_ASSERT( false, "Failed to load ImGui Vulkan functions." );
            ImGui_ImplGlfw_Shutdown();
            api.vkResetDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, 0 );
            api.vkDestroyDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, nullptr );
            return Result::FAIL;
        }

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance                  = m_device->GetInstance();
        init_info.PhysicalDevice            = m_device->GetPhysicalDevice();
        init_info.Device                    = m_device->GetHandle();
        init_info.QueueFamily               = m_device->GetGraphicsQueue()->GetFamilyIndex();
        init_info.Queue                     = m_device->GetGraphicsQueue()->GetHandle();
        init_info.DescriptorPool            = ( VkDescriptorPool )m_imguiDescriptorPool;
        init_info.MinImageCount             = 2;
        init_info.ImageCount                = 2;

        // Set API Version to 1.3 to ensure dynamic rendering functions are loaded correctly
        init_info.ApiVersion = VK_API_VERSION_1_3;

        // --- Dynamic Rendering Setup ---
        init_info.UseDynamicRendering = true;

        // Define the output format (Swapchain format)
        VkFormat                         format       = m_swapchain->GetFormat();
        VkPipelineRenderingCreateInfoKHR pipelineInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
        pipelineInfo.colorAttachmentCount             = 1;
        pipelineInfo.pColorAttachmentFormats          = &format;
        // No depth buffer for UI pass
        pipelineInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

        // Apply structure nesting (ImGui 2025 API update)
        init_info.PipelineInfoMain.MSAASamples                 = VK_SAMPLE_COUNT_1_BIT;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineInfo;

        if( !ImGui_ImplVulkan_Init( &init_info ) )
        {
            DT_ASSERT( false, "Failed to init ImGui for Vulkan." );
            ImGui_ImplGlfw_Shutdown();
            api.vkResetDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, 0 );
            api.vkDestroyDescriptorPool( m_device->GetHandle(), ( VkDescriptorPool )m_imguiDescriptorPool, nullptr );
            return Result::FAIL;
        }

        // Note: ImGui 1.91+ handles font upload automatically in NewFrame.
        // No manual CreateFontsTexture call needed here.
        return Result::SUCCESS;
    }

    void Renderer::Destroy()
    {
        m_device->WaitIdle();
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

        // Creates the main docking space
        ImGui::DockSpaceOverViewport( 0, ImGui::GetMainViewport(), ImGuiDockNodeFlags_PassthruCentralNode );
    }

    void Renderer::EndUI()
    {
        ImGui::Render();
    }

    void Renderer::RecordUIPass( CommandBuffer* cmd, Texture* backbuffer )
    {
        // 1. Barrier: Undefined -> Color Attachment
        {
            VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask          = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            barrier.srcAccessMask         = 0;
            barrier.dstStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.dstAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.oldLayout             = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.image                 = backbuffer->GetHandle();
            barrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            cmd->PipelineBarrier( VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1,
                                  &barrier );
        }

        // 2. Begin Dynamic Rendering
        VkRenderingAttachmentInfo colorAttachment = { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        colorAttachment.imageView                 = backbuffer->GetView();
        colorAttachment.imageLayout               = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp                    = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp                   = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.clearValue                = { 0.05f, 0.05f, 0.05f, 1.0f };

        VkRenderingInfo renderInfo      = { VK_STRUCTURE_TYPE_RENDERING_INFO };
        renderInfo.renderArea           = { { 0, 0 }, backbuffer->GetExtent().width, backbuffer->GetExtent().height };
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;

        cmd->BeginRendering( renderInfo );

        // 3. Draw ImGui
        ImDrawData* draw_data = ImGui::GetDrawData();
        if( draw_data )
        {
            ImGui_ImplVulkan_RenderDrawData( draw_data, cmd->GetHandle() );
        }

        cmd->EndRendering();

        // 4. Barrier: Color Attachment -> Present
        {
            VkImageMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            barrier.srcStageMask          = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            barrier.srcAccessMask         = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            barrier.dstStageMask          = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            barrier.dstAccessMask         = 0;
            barrier.oldLayout             = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            barrier.newLayout             = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            barrier.image                 = backbuffer->GetHandle();
            barrier.subresourceRange      = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            cmd->PipelineBarrier( VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr,
                                  1, &barrier );
        }
    }

    void Renderer::RenderUI( std::function<void()> callback )
    {
        if( callback )
            callback();
    }

    void* Renderer::GetImGuiTextureID( TextureHandle handle )
    {
        Texture* tex = m_resourceManager->GetTexture( handle );
        if( !tex )
            return nullptr;

        Sampler* sampler = m_resourceManager->GetSampler( m_defaultSampler );
        return ImGui_ImplVulkan_AddTexture( sampler->GetHandle(), tex->GetView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );
    }
    void* Renderer::GetImGuiContext()
    {
        return ImGui::GetCurrentContext();
    }
} // namespace DigitalTwin