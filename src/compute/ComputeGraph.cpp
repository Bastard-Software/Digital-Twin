#include "compute/ComputeGraph.h"

#include "rhi/CommandBuffer.h"
#include "rhi/GPUProfiler.h"
#include <volk.h>

namespace DigitalTwin
{
    uint32_t ComputeGraph::Execute( CommandBuffer* cmd, float dt, float totalTime, uint32_t activeIndex,
                                    GPUProfiler* profiler, uint32_t flightIndex )
    {
        uint32_t    localActive  = activeIndex;
        std::string currentPhase;

        for( auto& task: m_tasks )
        {
            if( task.ShouldExecute( dt ) )
            {
                // Close previous phase zone and open the new one on phase transition
                if( profiler )
                {
                    const std::string& taskPhase = task.GetPhaseName();
                    if( taskPhase != currentPhase )
                    {
                        if( !currentPhase.empty() )
                            profiler->EndZone( cmd, flightIndex, currentPhase, false );
                        currentPhase = taskPhase;
                        if( !currentPhase.empty() )
                            profiler->BeginZone( cmd, flightIndex, currentPhase, false );
                    }
                }

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

        // Close the last open zone
        if( profiler && !currentPhase.empty() )
            profiler->EndZone( cmd, flightIndex, currentPhase, false );

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