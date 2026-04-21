#pragma once
#include "DigitalTwinTypes.h"

#include "core/Core.h"
#include "platform/Window.h"
#include "simulation/SimulationBlueprint.h"
#include "simulation/SimulationValidator.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <string>

namespace DigitalTwin
{

    class DT_API DigitalTwin
    {
    public:
        DigitalTwin();
        ~DigitalTwin();

        Result Initialize( const DigitalTwinConfig& config );
        void   Shutdown();

        /**
         * @brief Sets the blueprint definition for the simulation without allocating GPU resources.
         */
        void SetBlueprint( const SimulationBlueprint& blueprint );

        const FrameContext& BeginFrame();
        void                EndFrame();
        void                Step(); // TODO: For future use

        void Play();
        void Pause();
        void Stop();

        /**
         * @brief Updates behaviour parameters on a live simulation without rebuilding GPU buffers.
         * No-op when in RESET state.
         */
        void HotReload( const SimulationBlueprint& blueprint );

        EngineState             GetState() const;
        const ValidationResult& GetLastValidationResult() const;

        bool IsWindowClosed();

        // Rendering
        /**
         * @brief Hot-swap MSAA sample count. Safe to call at any engine state.
         * @param samples 1 = Off, 4 = 4x MSAA. Unsupported values fall back to 1.
         */
        void     SetMSAA( uint32_t samples );
        uint32_t GetMSAA() const;
        /// Returns the maximum MSAA sample count supported by the current GPU (1 or 4).
        uint32_t GetMaxMSAA() const;

        void SetVSync( bool vsync );
        bool GetVSync() const;

        void                             SetGridVisualization( const GridVisualizationSettings& settings );
        const GridVisualizationSettings& GetGridVisualization() const;
        void                             RenderUI( std::function<void()> uiCallback );
        void                             SetShowStatsOverlay( bool show );
        bool                             IsShowingStatsOverlay() const;
        void                             SetViewportSize( uint32_t width, uint32_t height );
        void                             GetViewportSize( uint32_t& outWidth, uint32_t& outHeight ) const;
        void*                            GetSceneTextureID() const;
        void*                            GetImGuiTextureID( TextureHandle handle );
        void*                            GetImGuiContext();

        // --- Group Visibility ---
        void SetGroupVisible( int groupIndex, bool visible );

        // --- Phase 2.6.5.c.2 Step D — dynamic-topology debug visualization ---
        // Bit flags are forwarded to the voronoi_fan pipeline as a push
        // constant each frame. Zero = normal rendering. Non-vessel demos
        // (without dynamic topology) ignore these completely.
        static constexpr uint32_t DYNAMIC_TOPOLOGY_DEBUG_WIREFRAME        = 1u << 0;
        static constexpr uint32_t DYNAMIC_TOPOLOGY_DEBUG_VERTEX_COUNT     = 1u << 1;
        static constexpr uint32_t DYNAMIC_TOPOLOGY_DEBUG_HULL_MARKERS     = 1u << 2; // Step D.2
        static constexpr uint32_t DYNAMIC_TOPOLOGY_DEBUG_POLARITY_VECTORS = 1u << 3; // Step D.3
        static constexpr uint32_t DYNAMIC_TOPOLOGY_DEBUG_DRIFT_LINES      = 1u << 4; // Step D.3
        void     SetDynamicTopologyDebugFlags( uint32_t flags );
        uint32_t GetDynamicTopologyDebugFlags() const;

        // --- Camera API ---
        // Relative (interactive editor input from mouse deltas):
        void OrbitCamera( float pixelDX, float pixelDY );
        void PanCamera( float pixelDX, float pixelDY );
        void ZoomCamera( float scrollAmount );
        // Absolute (scripted demos / Python cinematic sequences):
        void        SetCameraFocus( const glm::vec3& point );
        void        SetCameraDistance( float distance );
        void        SetCameraOrientation( const glm::quat& orientation );
        void        FocusCameraOnDomain();
        CameraState GetCameraState() const;

    public:
        FileSystem* GetFileSystem() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };

} // namespace DigitalTwin