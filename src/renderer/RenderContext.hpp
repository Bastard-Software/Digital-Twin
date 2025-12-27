#pragma once
#include "core/Base.hpp"
#include "platform/Window.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/Device.hpp"
#include "rhi/Sampler.hpp"
#include "rhi/Swapchain.hpp"
#include "rhi/Texture.hpp"
#include <vector>

namespace DigitalTwin
{
    /**
     * @brief Manages rendering targets: Swapchain (Screen) and Viewport (Offscreen).
     * Uses Timeline Semaphores for CPU-GPU synchronization.
     */
    class RenderContext
    {
    public:
        RenderContext( Ref<Device> device, Window* window );
        ~RenderContext();

        Result Init();
        void   Shutdown();

        // --- Frame Lifecycle ---

        // Waits for the GPU to finish the previous cycle of this frame, then resets the command buffer.
        Ref<CommandBuffer> BeginFrame();

        // Submits the frame and updates synchronization values.
        void EndFrame( const std::vector<VkSemaphore>& waitSemaphores = {}, const std::vector<uint64_t>& waitValues = {} );

        // --- Resizing ---
        void OnResizeSwapchain( uint32_t width, uint32_t height );

        // Returns true if viewport resources were recreated (size changed)
        bool OnResizeViewport( uint32_t width, uint32_t height );

        // --- Getters ---
        Ref<Swapchain> GetSwapchain() const { return m_swapchain; }
        uint32_t       GetCurrentImageIndex() const { return m_imageIndex; }
        uint32_t       GetCurrentFrameIndex() const { return m_frameIndex; }
        VkFormat       GetColorFormat() const { return m_swapchain->GetFormat(); }

        // Multi-buffered resources getters
        Ref<Texture>                     GetViewportTexture() const { return m_viewportColors[ m_frameIndex ]; }
        Ref<Texture>                     GetViewportDepth() const { return m_viewportDepths[ m_frameIndex ]; }
        Ref<Sampler>                     GetViewportSampler() const { return m_viewportSampler; }
        const std::vector<Ref<Texture>>& GetAllViewportTextures() const { return m_viewportColors; }

        Ref<CommandBuffer> GetActiveCommandBuffer() const { return m_frames[ m_frameIndex ].cmd; }

        static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

    private:
        void RecreateSwapchain();
        void CreateViewportResources( uint32_t width, uint32_t height );

    private:
        Ref<Device>    m_device;
        Window*        m_window;
        Ref<Swapchain> m_swapchain;

        // --- Viewport Resources ---
        std::vector<Ref<Texture>> m_viewportColors;
        std::vector<Ref<Texture>> m_viewportDepths;
        Ref<Sampler>              m_viewportSampler;

        uint32_t m_imageIndex = 0;
        uint32_t m_frameIndex = 0;

        struct FrameData
        {
            Ref<CommandBuffer> cmd;

            // Binary semaphore for Present engine
            VkSemaphore renderFinished = VK_NULL_HANDLE;

            // Handle to semaphore from Swapchain (we don't own this)
            VkSemaphore currentImageAvailable = VK_NULL_HANDLE;

            // The value on the Graphics Queue Timeline we must wait for
            // before re-using this frame's command buffer.
            uint64_t timelineValue = 0;
        };
        std::vector<FrameData> m_frames;
    };
} // namespace DigitalTwin