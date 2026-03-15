#pragma once
#include "rhi/RHITypes.h"

#include <glm/glm.hpp>

namespace DigitalTwin
{

    /**
     * @brief Raw data mapped directly to Vulkan Push Constants in the compute shader.
     */
    /**
     * @brief Standardized 64-byte payload mapped to Vulkan Push Constants.
     * All compute shaders MUST define this exact memory layout.
     */
    struct ComputePushConstants
    {
        float      dt;
        float      totalTime;
        float      fParam0;
        float      fParam1;
        float      fParam2;
        float      fParam3;
        float      fParam4;
        float      fParam5;
        uint32_t   offset;
        uint32_t   maxCapacity;
        uint32_t   uParam0;
        uint32_t   uParam1;
        glm::vec4  domainSize;
        glm::uvec4 gridSize;
    };

    /**
     * @brief Private internal class representing a single dispatch operation on the GPU.
     */
    class ComputeTask
    {
    public:
        ComputeTask( ComputePipeline* pipeline, BindingGroup* bgRead0, BindingGroup* bgRead1, float targetHz, const ComputePushConstants& pc,
                     glm::uvec3 dispatchSize );

        /**
         * @brief Checks the time accumulator against targetHz to see if it should run this frame.
         */
        bool ShouldExecute( float dt );

        /**
         * @brief Binds pipeline, descriptors (depend on active ndx), pushes constants, and dispatches.
         */
        void Record( CommandBuffer* cmd, float totalTime, uint32_t activeIndex );

    private:
        ComputePipeline* m_pipeline;
        BindingGroup*    m_bindings[ 2 ]; // Array holding Ping and Pong binding groups
        float            m_targetHz;
        float            m_timeAccumulator = 0.0f;

        ComputePushConstants m_pc;
        glm::uvec3           m_dispatchSize;
    };

} // namespace DigitalTwin