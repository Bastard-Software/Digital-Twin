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
        GraphicsPipelineHandle m_voronoiPipeline;
        BindingGroupHandle     m_voronoiBindingGroups[ FRAMES_IN_FLIGHT ];
    };
} // namespace DigitalTwin