#pragma once
#include "rhi/RHITypes.h"

namespace DigitalTwin
{

    /**
     * @brief Raw data mapped directly to Vulkan Push Constants in the compute shader.
     */
    struct ComputePushConstants
    {
        float    dt;
        float    totalTime;
        float    speed;
        uint32_t offset;
        uint32_t count;
    };

    /**
     * @brief Private internal class representing a single dispatch operation on the GPU.
     */
    class ComputeTask
    {
    public:
        ComputeTask( ComputePipeline* pipeline, BindingGroup* bgRead0, BindingGroup* bgRead1, float targetHz, const ComputePushConstants& pc );

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
    };

} // namespace DigitalTwin