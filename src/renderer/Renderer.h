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
    class GridVisualizationPass;
    class Scene;
    class Camera;

    struct SimulationState;

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
         * @brief Records the simulation drawing commands (GeometryPass, etc.) into the command buffer.
         * @param cmd Command buffer in recording state.
         * @param state The simulation state containing data buffers.
         * @param camera The active camera for view/projection matrices.
         * @param flightIndex Current frame-in-flight index (for ping-pong resources).
         */
        void RenderSimulation( CommandBuffer* cmd, SimulationState* state, Camera* camera, uint32_t flightIndex );

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

        void                             SetGridVisualization( const GridVisualizationSettings& settings ) { m_gridVisSettings = settings; }
        const GridVisualizationSettings& GetGridVisualization() const { return m_gridVisSettings; }

    private:
        Device*           m_device;
        Swapchain*        m_swapchain;
        ResourceManager*  m_resourceManager;
        StreamingManager* m_streamingManager;

        // Settings
        GridVisualizationSettings m_gridVisSettings;

        // Sub-systems / Passes
        Scope<GeometryPass>          m_geometryPass;
        Scope<GridVisualizationPass> m_gridVisPass;

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