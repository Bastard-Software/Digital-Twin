#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>
#include <glm/glm.hpp>

namespace DigitalTwin
{
    class Device;
    class ResourceManager;
    class CommandBuffer;
    struct SimulationState;
    struct VesselVisualizationSettings;

    class VesselVisualizationPass
    {
    public:
        VesselVisualizationPass( Device* device, ResourceManager* rm );
        ~VesselVisualizationPass();

        Result Initialize( VkFormat colorFormat, VkFormat depthFormat, VkSampleCountFlagBits sampleCount = VK_SAMPLE_COUNT_1_BIT );
        void   Shutdown();

        void Execute( CommandBuffer* cmd, BufferHandle cameraUBO, const SimulationState* state,
                      const VesselVisualizationSettings& settings, uint32_t flightIndex );

    private:
        static const uint32_t FRAMES_IN_FLIGHT = 2;

        Device*          m_device;
        ResourceManager* m_resourceManager;

        ShaderHandle           m_vertShader;
        ShaderHandle           m_fragShader;
        GraphicsPipelineHandle m_pipeline;
        BindingGroupHandle     m_bindingGroups[ FRAMES_IN_FLIGHT ];
    };
} // namespace DigitalTwin
