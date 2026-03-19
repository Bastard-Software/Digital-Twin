#include "compute/ComputeGraph.h"

#include "rhi/CommandBuffer.h"
#include <volk.h>

namespace DigitalTwin
{
    uint32_t ComputeGraph::Execute( CommandBuffer* cmd, float dt, float totalTime, uint32_t activeIndex )
    {
        uint32_t localActive = activeIndex;

        for( auto& task: m_tasks )
        {
            if( task.ShouldExecute( dt ) )
            {
                task.Record( cmd, totalTime, localActive );

                VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.srcAccessMask    = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
                barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.dstAccessMask    = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

                VkDependencyInfo depInfo   = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers    = &barrier;

                cmd->PipelineBarrier( &depInfo );

                if( task.GetChainFlip() )
                    localActive = 1u - localActive;
            }
        }

        return localActive;
    }
    ComputeTask* ComputeGraph::FindTask( const std::string& tag )
    {
        for( auto& task: m_tasks )
        {
            if( task.GetTag() == tag )
                return &task;
        }
        return nullptr;
    }
} // namespace DigitalTwin