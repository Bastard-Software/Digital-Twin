#include "renderer/ImGuiLayer.hpp"

#include "core/Log.hpp"
#include "rhi/RHI.hpp"

// Define macros to use Volk with ImGui implementation
#define VK_NO_PROTOTYPES
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

namespace DigitalTwin
{
    ImGuiLayer::ImGuiLayer( Ref<Device> device, Window* window, VkFormat swapchainFormat )
        : m_device( device )
        , m_window( window )
    {
        DT_CORE_INFO( "[ImGuiLayer] Initializing ImGui (Dynamic Rendering)..." );

        // 1. Create a dedicated Descriptor Pool for ImGui
        // ImGui needs a lot of descriptors for fonts and textures.
        std::vector<VkDescriptorPoolSize> poolSizes = { { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
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

        // Use the helper method from Device to create a raw pool
        m_pool = m_device->CreateDescriptorPool( 1000, poolSizes );

        // 2. Initialize ImGui Context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;     // Enable Docking
        // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable; // Optional: Multi-Viewport

        ImGui::StyleColorsDark();

        // 3. Initialize GLFW Backend
        ImGui_ImplGlfw_InitForVulkan( ( GLFWwindow* )m_window->GetNativeWindow(), true );
        ImGui_ImplVulkan_LoadFunctions(
            VK_API_VERSION_1_3, []( const char* function_name, void* ) { return vkGetInstanceProcAddr( RHI::GetInstance(), function_name ); },
            nullptr );

        // 4. Initialize Vulkan Backend
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance                  = RHI::GetInstance();
        init_info.PhysicalDevice            = m_device->GetPhysicalDevice();
        init_info.Device                    = m_device->GetHandle();
        init_info.QueueFamily               = m_device->GetGraphicsQueue()->GetFamilyIndex();
        init_info.Queue                     = m_device->GetGraphicsQueue()->GetHandle();
        init_info.DescriptorPool            = m_pool;
        init_info.MinImageCount             = 3; // Triple buffering
        init_info.ImageCount                = 3;

        // Set API Version to 1.3 to ensure dynamic rendering functions are loaded correctly
        init_info.ApiVersion = VK_API_VERSION_1_3;

        // --- Dynamic Rendering Setup ---
        init_info.UseDynamicRendering = true;

        // Define the output format (Swapchain format)
        VkPipelineRenderingCreateInfoKHR pipelineInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR };
        pipelineInfo.colorAttachmentCount             = 1;
        pipelineInfo.pColorAttachmentFormats          = &swapchainFormat;
        // No depth buffer for UI pass
        pipelineInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

        // Apply structure nesting (ImGui 2025 API update)
        init_info.PipelineInfoMain.MSAASamples                 = VK_SAMPLE_COUNT_1_BIT;
        init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipelineInfo;

        ImGui_ImplVulkan_Init( &init_info );

        // Note: ImGui 1.91+ handles font upload automatically in NewFrame.
        // No manual CreateFontsTexture call needed here.
    }

    ImGuiLayer::~ImGuiLayer()
    {
        DT_CORE_INFO( "[ImGuiLayer] Shutting down..." );
        m_device->WaitIdle();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        m_device->DestroyDescriptorPool( m_pool );
    }

    void ImGuiLayer::Begin()
    {
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void ImGuiLayer::BeginDockspace()
    {
        // Setup a full-screen dockspace
        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos( viewport->Pos );
        ImGui::SetNextWindowSize( viewport->Size );
        ImGui::SetNextWindowViewport( viewport->ID );

        // Window flags to make it behave like a background container
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                        ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
                                        ImGuiWindowFlags_NoNavFocus;

        ImGui::PushStyleVar( ImGuiStyleVar_WindowRounding, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowBorderSize, 0.0f );
        ImGui::PushStyleVar( ImGuiStyleVar_WindowPadding, ImVec2( 0.0f, 0.0f ) );

        ImGui::Begin( "MainDockSpace", nullptr, window_flags );
        ImGui::PopStyleVar( 3 );

        ImGuiID dockspace_id = ImGui::GetID( "MyDockSpace" );
        ImGui::DockSpace( dockspace_id, ImVec2( 0.0f, 0.0f ), ImGuiDockNodeFlags_None );
    }

    void ImGuiLayer::EndDockspace()
    {
        ImGui::End(); // End MainDockSpace
    }

    void ImGuiLayer::End( CommandBuffer* cmd )
    {
        ImGui::Render();
        ImDrawData* drawData = ImGui::GetDrawData();

        // Record draw commands into the provided Command Buffer
        if( drawData )
        {
            ImGui_ImplVulkan_RenderDrawData( drawData, cmd->GetHandle() );
        }
    }

    ImTextureID ImGuiLayer::AddTexture( Ref<Texture> texture, Ref<Sampler> sampler )
    {
        if( !texture || !sampler )
            return ( ImTextureID )0;

        // Register texture with the provided sampler
        VkDescriptorSet set = ImGui_ImplVulkan_AddTexture( sampler->GetHandle(), texture->GetView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL );

        // Explicit cast to ImTextureID (ImU64) to fix compilation error
        return ( ImTextureID )set;
    }

    void ImGuiLayer::RemoveTexture( ImTextureID id )
    {
        if( id )
        {
            // Release the descriptor set back to the pool
            ImGui_ImplVulkan_RemoveTexture( ( VkDescriptorSet )id );
        }
    }

} // namespace DigitalTwin