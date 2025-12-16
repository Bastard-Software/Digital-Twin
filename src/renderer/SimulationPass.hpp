#pragma once
#include "core/Base.hpp"
#include "renderer/Scene.hpp"
#include "rhi/CommandBuffer.hpp"
#include "rhi/Device.hpp"
#include "rhi/Pipeline.hpp"

namespace DigitalTwin
{
    class ResourceManager;
    /**
     * @brief Responsible for drawing the Simulation Agents (Cells).
     * Manages Graphics Pipeline and Shaders.
     */
    class SimulationPass
    {
    public:
        SimulationPass( Ref<Device> device, Ref<ResourceManager> resManager );
        ~SimulationPass();

        void Init( VkFormat colorFormat, VkFormat depthFormat );

        /**
         * @brief Records draw commands into the provided command buffer.
         */
        void Draw( CommandBuffer* cmd, const Scene& scene );

    private:
        Ref<Device>           m_device;
        Ref<GraphicsPipeline> m_pipeline;
        Ref<ResourceManager>  m_resManager;
    };
} // namespace DigitalTwin