#pragma once

#include "platform/Window.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/Device.hpp"
#include "rhi/Sampler.hpp" // Required for passing samplers to ImGui
#include "rhi/Texture.hpp"
#include <imgui.h>

namespace DigitalTwin
{
    class ImGuiLayer
    {
    public:
        /**
         * @brief Initializes ImGui with Vulkan backend (Dynamic Rendering).
         * @param device Reference to the RHI Device.
         * @param window Pointer to the OS Window wrapper.
         * @param swapchainFormat The format of the swapchain images (needed for pipeline creation).
         */
        ImGuiLayer( Ref<Device> device, Window* window, VkFormat swapchainFormat );
        ~ImGuiLayer();

        // --- Frame Lifecycle ---

        // Call at the very beginning of the frame (ImGui::NewFrame)
        void Begin();

        // Call inside a Render Pass to record ImGui draw commands
        void End( CommandBuffer* cmd );

        // --- Docking Support ---

        // Sets up the main dockspace over the viewport
        void BeginDockspace();
        void EndDockspace();

        // --- Resource Management ---

        /**
         * @brief Registers a texture with ImGui to be used in Image() widgets.
         * @param texture The texture to display.
         * @param sampler The sampler to use (should be ClampToEdge for UI).
         * @return ImTextureID Handle to be passed to ImGui::Image().
         */
        ImTextureID AddTexture( Ref<Texture> texture, Ref<Sampler> sampler );
        // Frees the descriptor set associated with the texture ID
        void RemoveTexture( ImTextureID id );

        // --- Input Queries ---

        // Returns true if ImGui wants to capture mouse input (block game camera)
        bool BlockEvents() const { return m_blockEvents; }

    private:
        Ref<Device>      m_device;
        Window*          m_window;
        VkDescriptorPool m_pool        = VK_NULL_HANDLE;
        bool             m_blockEvents = true;
    };
} // namespace DigitalTwin