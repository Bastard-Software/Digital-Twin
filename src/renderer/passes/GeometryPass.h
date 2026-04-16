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

        ShaderHandle           m_vertShader;
        ShaderHandle           m_fragShader;
        GraphicsPipelineHandle m_pipeline;
        BindingGroupHandle     m_bindingGroups[ FRAMES_IN_FLIGHT ];
    };
} // namespace DigitalTwin