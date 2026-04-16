#pragma once

#include "core/Core.h"
#include "core/Handle.h"
#include "platform/Window.h"
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace DigitalTwin
{
    class Log;
    class FileSystem;

    DEFINE_HANDLE( TextureHandle );
    DEFINE_HANDLE( BufferHandle );
    DEFINE_HANDLE( ShaderHandle );
    DEFINE_HANDLE( SamplerHandle );
    DEFINE_HANDLE( MeshHandle );
    DEFINE_HANDLE( ComputePipelineHandle );
    DEFINE_HANDLE( GraphicsPipelineHandle );
    DEFINE_HANDLE( ThreadContextHandle );
    DEFINE_HANDLE( CommandBufferHandle );
    DEFINE_HANDLE( BindingGroupHandle );

    enum class GPUType
    {
        DEFAULT,
        DISCRETE,
        INTEGRATED,
    };

    struct DigitalTwinConfig
    {
        GPUType     gpuType       = GPUType::DEFAULT;
        bool_t      headless      = true;
        const char* rootDirectory = nullptr;
        bool_t      debugMode     = false;
        WindowDesc  windowDesc;
        uint32_t    msaaSamples   = 1; ///< Initial MSAA sample count. Valid: 1 (Off) or 4 (4x MSAA).
    };

    enum class EngineState
    {
        RESET,   // Editor mode. GPU buffers are freed. Blueprint is being edited.
        PLAYING, // Simulation is running. Compute and Graphics are active.
        PAUSED,  // Simulation is paused. Compute is halted, Graphics are active.
    };

    struct FrameContext
    {
        CommandBufferHandle graphicsCmd;
        CommandBufferHandle computeCmd;

        TextureHandle sceneTexture;

        BufferHandle agentBufferRead;
        BufferHandle agentBufferWrite;

        uint32_t frameIndex;
    };

    // Visualization
    enum class GridVisualizationMode
    {
        VOLUMETRIC_CLOUD = 0,
        SLICE_2D         = 1
    };

    enum class Colormap : int
    {
        JET    = 0, // Blue → cyan → green → yellow → red (classic rainbow)
        OXYGEN = 1, // Dark blue (hypoxic) → cyan → yellow → red (normoxic)
        HOT    = 2, // Black → red → yellow → white (fluorescence standard / Fiji Fire)
        PLASMA = 3, // Dark purple → magenta → orange → yellow (perceptually uniform)
        VEGF   = 4, // Near-black → yellow → orange → magenta (VEGF-specific)
        CUSTOM = 5, // User-defined 3-stop linear gradient
    };

    struct GridFieldVisualization
    {
        float     minValue    = 0.0f;
        float     maxValue    = 100.0f;
        float     alphaCutoff = 0.01f; // Normalized values below this are fully transparent
        float     gamma       = 1.0f;  // Nonlinear ramp: t = pow(t, gamma). <1 lifts weak gradients; >1 focuses on peaks
        Colormap  colormap    = Colormap::JET;
        glm::vec3 customLow   = { 0.0f, 0.0f, 0.5f };  // Color at t=0
        glm::vec3 customMid   = { 0.0f, 0.8f, 0.2f };  // Color at t=0.5
        glm::vec3 customHigh  = { 1.0f, 0.9f, 0.0f };  // Color at t=1
    };

    struct GridVisualizationSettings
    {
        bool                  active       = false;
        int                   fieldIndex   = 0;    // Index of the GridField (0 = Oxygen, 1 = VEGF, etc.)
        GridVisualizationMode mode         = GridVisualizationMode::SLICE_2D;
        float                 sliceZ       = 0.5f; // Normalized Z depth [0.0 - 1.0] for the 2D slice
        float                 opacitySlice = 0.4f; // Ideal for Heatmap
        float                 opacityCloud = 0.04f; // Ideal for Raymarching
        GridFieldVisualization fieldVis;            // Per-field colormap / range settings
    };

    struct VesselVisualizationSettings
    {
        bool      active    = false;
        glm::vec4 lineColor = glm::vec4( 1.0f, 0.85f, 0.0f, 1.0f ); // Gold
    };

    struct CameraState
    {
        glm::vec3 position;
        glm::vec3 focalPoint;
        glm::quat orientation;
        float     distance;
        float     fov;
    };

} // namespace DigitalTwin