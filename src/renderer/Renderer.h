#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>
#include <functional>
#include <memory>
#include <vector>

namespace DigitalTwin
{
    class ResourceManager;
    class StreamingManager;
    class RenderTarget;
    class GeometryPass;
    class Scene;
    class Camera;

    /**
     * @brief High-level rendering manager.
     * Orchestrates rendering passes (Geometry, UI) and manages frame-specific resources like RenderTargets and UBOs.
     */
    class Renderer
    {
    public:
        Renderer( Device* device, Swapchain* swapchain, ResourceManager* resourceManager, StreamingManager* streamingManager );
        ~Renderer();

        Result Create();
        void   Destroy();

        void BeginUI();
        void EndUI();
        void RenderUI( std::function<void()> callback );
        void RecordUIPass( CommandBuffer* cmd, Texture* backbuffer );

        /**
         * @brief Sets the desired resolution for the offscreen RenderTarget.
         * Usually called by the UI panel that displays the scene viewport.
         */
        void SetViewportSize( uint32_t width, uint32_t height );
        void GetViewportSize( uint32_t& width, uint32_t& height ) const
        {
            width  = m_viewportWidth;
            height = m_viewportHeight;
        }

        /**
         * @brief Records the scene drawing commands (GeometryPass, etc.) into the command buffer.
         * @param cmd Command buffer in recording state.
         * @param scene The simulation scene containing data buffers.
         * @param camera The active camera for view/projection matrices.
         * @param flightIndex Current frame-in-flight index (for ping-pong resources).
         */
        void RecordScenePass( CommandBuffer* cmd, Scene* scene, Camera* camera, uint32_t flightIndex );

        /**
         * @brief Retrieves the ImGui texture ID for the rendered scene.
         * Lazily resizes the RenderTarget if the viewport size has changed.
         * @param flightIndex Current frame-in-flight index.
         * @return void* Pointer casted to ImTextureID for ImGui::Image.
         */
        void* GetSceneTextureID( uint32_t flightIndex );

        /**
         * @brief Retrieves the ImGui context created by the renderer.
         */
        void* GetImGuiContext();

        SamplerHandle GetDefaultSampler() const { return m_defaultSampler; }

    private:
        Device*           m_device;
        Swapchain*        m_swapchain;
        ResourceManager*  m_resourceManager;
        StreamingManager* m_streamingManager;

        // Sub-systems / Passes
        Scope<GeometryPass> m_geometryPass;

        // --- Per-Frame Resources ---
        static const uint32_t FRAMES_IN_FLIGHT = 2;

        /// Offscreen render targets (Color + Depth) per frame
        Scope<RenderTarget> m_renderTargets[ FRAMES_IN_FLIGHT ];

        /// Uniform buffers containing Camera matrices per frame
        BufferHandle m_cameraUBOs[ FRAMES_IN_FLIGHT ];

        /// Cached ImGui texture descriptors to avoid recreating them every frame
        void* m_cachedImGuiTextures[ FRAMES_IN_FLIGHT ] = { nullptr, nullptr };

        // Desired viewport size for the scene
        uint32_t m_viewportWidth  = 800;
        uint32_t m_viewportHeight = 600;

        // Default global resources
        SamplerHandle m_defaultSampler;
        void*         m_imguiDescriptorPool = nullptr;
    };

} // namespace DigitalTwin