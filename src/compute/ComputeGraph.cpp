#include "compute/ComputeGraph.h"

#include "rhi/CommandBuffer.h"
#include "rhi/GPUProfiler.h"
#include <volk.h>

namespace DigitalTwin
{
    uint32_t ComputeGraph::Execute( CommandBuffer* cmd, float dt, float totalTime, uint32_t activeIndex,
                                    GPUProfiler* profiler, uint32_t flightIndex )
    {
        // Two independent ping-pong domains, one index each:
        //   agentActive  — threaded in via activeIndex, flipped by every ChainFlip position task,
        //                  returned to the caller (agent buffer continuity).
        //   m_fieldActive — persists on the graph, advanced ONCE per frame, used only by field
        //                  (diffusion PDE) tasks. Decoupling these means the field solver evolves
        //                  correctly no matter how many position tasks flip the agent index — the
        //                  fix for "a group with two Chemotaxis behaviours corrupts the sim".
        // Field READERS (chemotaxis / antigen_kill sampling a field texture) bind via agentActive
        // and so may sample the field one frame stale; harmless for a slowly-diffusing field.
        uint32_t    agentActive  = activeIndex;
        std::string currentPhase;
        bool        anyFieldTask = false;

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

                task.Record( cmd, totalTime, task.IsFieldTask() ? m_fieldActive : agentActive );

                VkMemoryBarrier2 barrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                barrier.srcStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.srcAccessMask    = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
                barrier.dstStageMask     = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                barrier.dstAccessMask    = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;

                VkDependencyInfo depInfo   = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                depInfo.memoryBarrierCount = 1;
                depInfo.pMemoryBarriers    = &barrier;

                cmd->PipelineBarrier( &depInfo );

                if( task.IsFieldTask() )
                    anyFieldTask = true;
                else if( task.GetChainFlip() )
                    agentActive = 1u - agentActive;
            }
        }

        // Close the last open zone
        if( profiler && !currentPhase.empty() )
            profiler->EndZone( cmd, flightIndex, currentPhase, false );

        // Advance the field ping-pong once per frame (all field PDEs this frame shared m_fieldActive;
        // afterwards it points at the just-written textures). Only when a field task actually ran.
        if( anyFieldTask )
            m_fieldActive = 1u - m_fieldActive;

        return agentActive;
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