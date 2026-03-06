#include "compute/GraphDispatcher.h"

#include "compute/ComputeGraph.h"
#include "rhi/CommandBuffer.h"

namespace DigitalTwin
{
    void GraphDispatcher::Dispatch( ComputeGraph* graph, CommandBuffer* computeCmd, CommandBuffer* graphicsCmd, float dt, float totalTime,
                                    uint32_t activeIndex )
    {
        ( void )graphicsCmd;

        if( !graph || graph->IsEmpty() )
            return;

        // =========================================================================
        // FUTURE HEURISTICS HERE:
        // Analyze graph dependencies.
        // Tasks touching geometry buffers might go to graphicsCmd.
        // Heavy math tasks go to computeCmd.
        // =========================================================================

        // For now: We record all compute tasks sequentially into the dedicated compute command buffer.
        graph->Execute( computeCmd, dt, totalTime, activeIndex );
    }
} // namespace DigitalTwin