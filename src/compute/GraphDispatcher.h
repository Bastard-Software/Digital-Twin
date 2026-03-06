#pragma once

#include <cstdint>

namespace DigitalTwin
{
    class ComputeGraph;
    class CommandBuffer;

    /**
     * @brief Responsible for analyzing the ComputeGraph and dispatching it to GPU queues.
     * In the future, this class will contain heuristics to split workloads between
     * the Compute Command Buffer and Graphics Command Buffer.
     */
    class GraphDispatcher
    {
    public:
        static void Dispatch( ComputeGraph* graph, CommandBuffer* computeCmd, CommandBuffer* graphicsCmd, float dt, float totalTime,
                              uint32_t activeIndex );
    };
} // namespace DigitalTwin