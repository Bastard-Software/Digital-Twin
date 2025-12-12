#include "compute/ComputeGraph.hpp"

namespace DigitalTwin
{

    void ComputeGraph::AddTask( Ref<ComputeKernel> kernel, Ref<BindingGroup> bindings )
    {
        m_tasks.push_back( { kernel, bindings } );
    }

    void ComputeGraph::Record( CommandBuffer& cmd, uint32_t agentCount )
    {
        for( size_t i = 0; i < m_tasks.size(); ++i )
        {
            // 1. Execute Task
            m_tasks[ i ].kernel->Dispatch( cmd, m_tasks[ i ].bindings, agentCount );

            // 2. Insert Barrier (Simple global barrier for safety between kernels)
            if( i < m_tasks.size() - 1 )
            {
                VkMemoryBarrier memBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
                memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
                memBarrier.dstAccessMask   = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

                cmd.PipelineBarrier( VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, { memBarrier }, {}, {} );
            }
        }
    }
} // namespace DigitalTwin