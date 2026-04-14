#include "compute/GraphDispatcher.h"

#include "compute/ComputeGraph.h"
#include "rhi/CommandBuffer.h"

namespace DigitalTwin
{
    uint32_t GraphDispatcher::Dispatch( ComputeGraph* graph, CommandBuffer* computeCmd, CommandBuffer* graphicsCmd, float dt, float totalTime,
                                        uint32_t activeIndex, GPUProfiler* profiler, uint32_t flightIndex )
    {
        ( void )graphicsCmd;

        if( !graph || graph->IsEmpty() )
            return activeIndex;

        // =========================================================================
        // FUTURE HEURISTICS HERE:
        // Analyze graph dependencies.
        // Tasks touching geometry buffers might go to graphicsCmd.
        // Heavy math tasks go to computeCmd.
        // =========================================================================

        // For now: We record all compute tasks sequentially into the dedicated compute command buffer.
        // Returns the final active index after all chain-flipping position-writing tasks.
        return graph->Execute( computeCmd, dt, totalTime, activeIndex, profiler, flightIndex );
    }
} // namespace DigitalTwin