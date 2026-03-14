#pragma once
#include "rhi/RHITypes.h"

#include <DigitalTwinTypes.h>

namespace DigitalTwin
{
    class Device;
    class ResourceManager;
    class CommandBuffer;
    class Scene;

    /**
     * @brief A pre-geometry compute pass executed by the Renderer.
     * It reads the raw AgentCountBuffer from the simulation and dynamically
     * overwrites the 'instanceCount' field in the IndirectCommandBuffer.
     */
    class BuildIndirectPass
    {
    public:
        BuildIndirectPass( Device* device, ResourceManager* rm );
        ~BuildIndirectPass();

        Result Initialize();
        void   Shutdown();

        void Execute( CommandBuffer* cmd, Scene* scene, uint32_t flightIndex );

    private:
        static const uint32_t FRAMES_IN_FLIGHT = 2;

        Device*          m_device;
        ResourceManager* m_resourceManager;

        ShaderHandle          m_compShader;
        ComputePipelineHandle m_pipeline;
        BindingGroupHandle    m_bindingGroups[ FRAMES_IN_FLIGHT ];
    };
} // namespace DigitalTwin