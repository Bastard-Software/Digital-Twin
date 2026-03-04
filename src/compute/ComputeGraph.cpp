#include "compute/ComputeGraph.h"

#include "rhi/CommandBuffer.h"
#include <volk.h>

namespace DigitalTwin
{
    void ComputeGraph::Execute( CommandBuffer* cmd, float dt, float totalTime, uint32_t activeIndex )
    {
        bool didExecuteAny = false;

        for( auto& task: m_tasks )
        {
            if( task.ShouldExecute( dt ) )
            {
                task.Record( cmd, totalTime, activeIndex );
                didExecuteAny = true;
            }
        }

        // Automatic memory barrier to ensure Compute results are visible to Graphics/Compute
        if( didExecuteAny )
        {
            VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.srcAccessMask    = VK_ACCESS_2_SHADER_WRITE_BIT;
            barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            barrier.dstAccessMask    = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

            VkDependencyInfo depInfo   = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers    = &barrier;

            cmd->PipelineBarrier( &depInfo );
        }
    }
} // namespace DigitalTwin