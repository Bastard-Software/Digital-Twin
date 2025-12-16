#pragma once
#include "core/Base.hpp"
#include "platform/Window.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/Device.hpp"
#include "rhi/Swapchain.hpp"
#include "rhi/Texture.hpp"
#include <vector>

namespace DigitalTwin
{

    class RenderContext
    {
    public:
        RenderContext( Ref<Device> device, Window* window );
        ~RenderContext();

        Result Init();
        void   Shutdown();

        // --- Frame Lifecycle ---
        CommandBuffer* BeginFrame();

        void EndFrame( const std::vector<VkSemaphore>& waitSemaphores = {}, const std::vector<uint64_t>& waitValues = {} );

        void OnResize( uint32_t width, uint32_t height );

        // --- Getters ---
        Ref<Swapchain> GetSwapchain() const { return m_swapchain; }
        uint32_t       GetCurrentImageIndex() const { return m_imageIndex; }
        VkFormat       GetColorFormat() const { return m_swapchain->GetFormat(); }
        VkFormat       GetDepthFormat() const { return VK_FORMAT_D32_SFLOAT; }
        Ref<Texture>   GetDepthTexture() const { return m_depthTexture; }

    private:
        void RecreateSwapchain();
        void CreateDepthResources();

    private:
        Ref<Device>    m_device;
        Window*        m_window;
        Ref<Swapchain> m_swapchain;
        Ref<Texture>   m_depthTexture;

        static constexpr uint32_t FRAMES_IN_FLIGHT = 2;

        struct FrameData
        {
            Ref<CommandBuffer> cmd;

            // Semaphore signalled when Rendering is done (created by Context)
            VkSemaphore renderFinished = VK_NULL_HANDLE;

            // Semaphore signalled when Image is available (owned by Swapchain, just a handle here)
            VkSemaphore currentImageAvailable = VK_NULL_HANDLE;

            // Timeline value to wait for on CPU before reusing this frame slot
            uint64_t timelineValue = 0;
        };

        FrameData m_frames[ FRAMES_IN_FLIGHT ];
        uint32_t  m_frameIndex = 0;
        uint32_t  m_imageIndex = 0;
    };
} // namespace DigitalTwin