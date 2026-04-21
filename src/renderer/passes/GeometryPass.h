#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>

namespace DigitalTwin
{
    class Device;
    class ResourceManager;
    class CommandBuffer;
    class Scene;

    class GeometryPass
    {
    public:
        GeometryPass( Device* device, ResourceManager* rm );
        ~GeometryPass();

        Result Initialize( VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT );
        void   Shutdown();

        void Execute( CommandBuffer* cmd, BufferHandle cameraUBO, Scene* scene, uint32_t flightIndex );

    private:
        static const uint32_t FRAMES_IN_FLIGHT = 2;

        Device*          m_device;
        ResourceManager* m_resourceManager;

        // Static mesh pipeline (geometry.vert/frag) — existing rhombus/CurvedTile path.
        ShaderHandle           m_vertShader;
        ShaderHandle           m_fragShader;
        GraphicsPipelineHandle m_pipeline;
        BindingGroupHandle     m_bindingGroups[ FRAMES_IN_FLIGHT ];

        // Phase 2.6.5.c — dynamic-topology pipeline (voronoi_fan.vert + geometry.frag).
        // Reads per-cell polygon from PolygonBuffer instead of a shared vertex buffer;
        // emits a 12-triangle fan per instance with degenerate-triangle slots for
        // cells whose polygon has < 12 vertices.
        ShaderHandle           m_voronoiVertShader;
        ShaderHandle           m_voronoiFragShader;   // Phase 2.6.5.c.2 Step D — dedicated frag with debug overlay
        GraphicsPipelineHandle m_voronoiPipeline;
        BindingGroupHandle     m_voronoiBindingGroups[ FRAMES_IN_FLIGHT ];

        // Phase 2.6.5.c.2 Step D.2 — debug markers pipeline.
        // LINE_LIST topology. 16 verts/instance × N_agents. Renders radial
        // lines from each cell center to each of 8 contact-hull points,
        // exposing where JKR thinks cells are and how the rigid-body rhombus
        // hull is oriented. Only dispatched when a debug flag is set.
        ShaderHandle           m_debugMarkersVertShader;
        ShaderHandle           m_debugMarkersFragShader;
        GraphicsPipelineHandle m_debugMarkersPipeline;
        BindingGroupHandle     m_debugMarkersBindingGroups[ FRAMES_IN_FLIGHT ];

        // Phase 2.6.5.c.2 Step D.3 — debug vectors pipeline (polarity / drift).
        // LINE_LIST topology, 2 verts/instance × N_agents. One shader, two
        // draw modes selected via push constant.
        ShaderHandle           m_debugVectorsVertShader;
        ShaderHandle           m_debugVectorsFragShader;
        GraphicsPipelineHandle m_debugVectorsPipeline;
        BindingGroupHandle     m_debugVectorsBindingGroups[ FRAMES_IN_FLIGHT ];
    };
} // namespace DigitalTwin