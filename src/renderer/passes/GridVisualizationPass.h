#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>
#include <glm/glm.hpp>

namespace DigitalTwin
{
    class Device;
    class ResourceManager;
    class CommandBuffer;
    struct GridFieldState;
    struct GridVisualizationSettings;

    class GridVisualizationPass
    {
    public:
        GridVisualizationPass( Device* device, ResourceManager* rm );
        ~GridVisualizationPass();

        Result Initialize( VkFormat colorFormat, VkFormat depthFormat );
        void   Shutdown();

        void Execute( CommandBuffer* cmd, BufferHandle cameraUBO, const GridFieldState* gridState, const GridVisualizationSettings& settings,
                      const glm::vec3& domainSize, uint32_t flightIndex );

    private:
        static const uint32_t FRAMES_IN_FLIGHT = 2;

        Device*          m_device;
        ResourceManager* m_resourceManager;

        SamplerHandle m_linearSampler;

        ShaderHandle           m_vertShader;
        ShaderHandle           m_fragShader;
        GraphicsPipelineHandle m_pipeline;
        BindingGroupHandle     m_bindingGroups[ FRAMES_IN_FLIGHT ];
    };
} // namespace DigitalTwin