#include "compute/ComputeTask.h"

#include "core/Log.h"
#include "rhi/BindingGroup.h"
#include "rhi/CommandBuffer.h"
#include "rhi/Pipeline.h"
#include <cmath>

namespace DigitalTwin
{
    ComputeTask::ComputeTask( ComputePipeline* pipeline, BindingGroup* bgRead0, BindingGroup* bgRead1, float targetHz, const ComputePushConstants& pc,
                              glm::uvec3 dispatchSize )
        : m_pipeline( pipeline )
        , m_targetHz( targetHz )
        , m_pc( pc )
        , m_dispatchSize( dispatchSize )
    {
        m_bindings[ 0 ] = bgRead0;
        m_bindings[ 1 ] = bgRead1;
    }

    bool ComputeTask::ShouldExecute( float dt )
    {
        if( m_targetHz <= 0.0f )
            return true; // Continuous execution

        m_timeAccumulator += dt;
        float threshold = 1.0f / m_targetHz;

        if( m_timeAccumulator >= threshold )
        {
            m_timeAccumulator = std::fmod( m_timeAccumulator, threshold );
            m_pc.dt           = threshold; // Ensure physics step is deterministic
            return true;
        }
        return false;
    }

    void ComputeTask::Record( CommandBuffer* cmd, float totalTime, uint32_t activeIndex )
    {
        DT_ASSERT( activeIndex < 2, "Active index out of range" );

        cmd->SetPipeline( m_pipeline );
        cmd->SetBindingGroup( m_bindings[ activeIndex ], m_pipeline->GetLayout(), VK_PIPELINE_BIND_POINT_COMPUTE );

        m_pc.totalTime = totalTime;
        cmd->PushConstants( m_pipeline->GetLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof( ComputePushConstants ), &m_pc );

        cmd->Dispatch( m_dispatchSize.x, m_dispatchSize.y, m_dispatchSize.z );
    }
} // namespace DigitalTwin